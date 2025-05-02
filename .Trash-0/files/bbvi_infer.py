'''Infer soft states'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.optim import AdamW
from tqdm.auto import trange
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 23
torch.manual_seed(seed)  # Set PyTorch seed
torch.cuda.manual_seed_all(seed)  # Set CUDA seed (if available)
np.random.seed(seed)  # Set NumPy seed
random.seed(seed)  # Set Python random seed
torch.backends.cudnn.deterministic = True  # Ensure CuDNN is deterministic
torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking


### GS-BBVI algorithm
def fit_bbvi_schedule(model, ys, xs0=None, num_iters=100, learning=True, n_samples=10, 
                     base_lr=1e-2, warmup_iters=30, tau_max=0.1):
    device = ys.device
    tau = torch.tensor(tau_max, device=device)
    elbo_history = []
    convergence_window = 10  ## determines strictness of convergence
    
    # --- Initialize Variational Distribution ---
    T, N = ys.shape
    K, D = model.K, model.D
    '''ADDED variational q(z)'''
    variational_z = GSVariational(D, K).to(device)
    variational_x = LDSVariational(D, K, T).to(device)

    # --- Optimizer Setup ---
    '''ADDED params for q(z)'''
    params = (model.params + variational_x.params + variational_z.params) if learning else (variational_x.params + variational_z.params)
    optimizer = AdamW(params, lr=base_lr, weight_decay=1e-4)  # Start at min_lr
    
    # --- Simple 2-Phase Schedule ---
    warmup = LinearLR(optimizer, 
                     start_factor=0.5,  # Starts at 0.5 * base_lr (0.005)
                     end_factor=1.0,     # Linearly increases to full LR
                     total_iters=warmup_iters, last_epoch=-1)
    
    #decay = CosineAnnealingLR(optimizer,
    #                        T_max= warmup_iters*3,  # Duration of decay
    #                        eta_min=1e-3)  # Minimum LR
    decay = LinearLR(optimizer,start_factor=1.0, end_factor=0.5,total_iters=3*warmup_iters)
    hold = ConstantLR(optimizer,factor=1.0,total_iters=float('inf'))
    
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup, decay, hold],
                           milestones=[warmup_iters, 3*warmup_iters])

    
    # --- Training Loop ---
    pbar = trange(num_iters)
    for itr in pbar:
        optimizer.zero_grad()
        
        # --- Samples ---
        if itr == 0:  ## warmup
            if xs0 is not None:
                xs_init = xs0.unsqueeze(0).expand(n_samples, -1, -1)
            else:
                xs_init = model.initialize_ms(ys.unsqueeze(0).expand(n_samples, -1, -1))  
        else:
            xs_init = variational_x.mean_x(zs.detach()) #xs.detach() 
        '''ADDED sample zs from xs_init'''    
        zs = variational_z.sample_q_z(xs_init, tau)
        xs = variational_x.sample_q_x(zs.detach())  
        
        # --- ELBO COMPUTATION ---
        def ELBO_objective(params, itr, zs, xs, tau):
            log_pz = model.log_transition_likelihoods(zs, xs, tau)
            log_px = model.dynamics_log_likelihoods(xs, zs)
            log_py = model.log_emission_likelihood(ys.unsqueeze(0).expand(n_samples, -1, -1), xs)
            '''ADDED allow q(z) inference'''
            log_qz = variational_z.log_density(zs, xs.detach(), tau)
            log_qx = variational_x.log_density(xs, zs.detach())
            state_counts = zs.mean(dim=(0,1))  # [K] average usage per state
            state_entropy = -(state_counts * torch.log(state_counts + 1e-8)).sum()
            ### Optional plotting
            if (itr % 20 == 0 or itr == num_iters-1):
                plot_states(torch.argmax(torch.mean(zs, dim=0), dim=1).cpu().numpy())
                plot_trajectory(
                    torch.argmax(torch.mean(zs, dim=0), dim=1).cpu().numpy(),
                    variational_x.mean_x(zs).mean(dim=0).detach().cpu().numpy(), 
                    ls="-"
                )
            return (log_pz + log_px + log_py - log_qx - log_qz).mean()+ state_entropy
        
        # --- Gradient Handling ---
        elbo = ELBO_objective(params, itr, zs, xs, tau)
        loss = -elbo
        loss.backward()
        
        # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(params, max_norm=1)
        optimizer.step()
        scheduler.step()
       
        
        # --- Monitoring ---
        elbo_history.append(-loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"ELBO: {elbo_history[-1]:.1f}, LR: {current_lr:.5f}, Tau: {tau.item():.3f}")
        
        # --- EARLY STOPPING ---
        if len(elbo_history) > convergence_window:
            elbo_improvement =  np.mean(elbo_history[-1] -np.array(elbo_history[-convergence_window:-1] ))
            if elbo_improvement < 1e-1:  ## avg ELBO not increasing
                print(f"Early stopping at iteration {itr}.")
                break
            #tau = torch.max(tau * torch.exp(-torch.tensor(0.01, device=device)), 
            #              torch.tensor(0.5, device=device)) 

    return np.array(elbo_history), variational_x, variational_z



### Model likelihood
class GenerativeSLDS(nn.Module):
    def __init__(self, N, K, D, M=0, lags=1, emission_model="poisson"):
        """
        N: Number of units
        K: Number of discrete states
        D: Dimensionality of continuous state space
        M: Dimensionality of control inputs
        lags: Number of past states used in autoregression
        emission_model: "poisson" or "gaussian"
        """
        super(GenerativeSLDS, self).__init__()   
        self.N = N
        self.K = K
        self.D = D
        self.M = M
        self.lags = lags
        self.emission_model = emission_model
        
        '''Continuous latent variables x_{1:T}'''
        # Transition matrices A_k (K, D, D*lags)
        self.As = nn.Parameter(0.8 * torch.stack([
            torch.cat([self._random_rotation(D) for _ in range(lags)], dim=1)
            for _ in range(K)]).to(device)) 
        # Bias term b_k (K, D)
        self.bs = nn.Parameter(torch.zeros(K, D))  # Initialize as zeros
        # Optional control input matrices V_k (K, D, M)
        self.Vs = nn.Parameter(torch.randn(K, D, M, device=device) * 0.1) if M > 0 else None
        # Initial mean values (K, D)
        self.mu_init = torch.randn(K, D, device=device) 
        # Initial and transition covariance matrices (fixed)
        self.sqrt_Sigmas_init = torch.eye(D, device=device).repeat(K, 1, 1) 
        self.sqrt_Sigmas = torch.eye(D, device=device).repeat(K, 1, 1)  # nn.Parameter(torch.randn(K, D, D, device=device))
        
        '''Discrete latent variables z_{1:T} for state switching'''
        self.Rs = nn.Parameter(torch.randn(K, D, device=device))
        self.r = nn.Parameter(torch.tensor(0.1, device=device))
        self.logits_z1 = nn.Parameter(torch.zeros(K, device=device))
        
        '''Observation parameters'''
        self.C = nn.Parameter(torch.randn(N, D, device=device))  
        self.d = nn.Parameter(torch.randn(N, device=device))    
        # Additional parameter for Gaussian emission model
        if self.emission_model == "gaussian":
            self.log_var = nn.Parameter(-1.0 + torch.rand(1, N, device=device))
    
    @property
    def params(self):
        '''ADDED p(z) parameters'''
        params = [
            self.C, self.d,  # Observation parameters (y)
            self.Rs, self.r, self.logits_z1, # Discrete state parameters (z)
            self.As, self.bs  # Continuous state parameters (x)   
        ]
        if self.emission_model == "gaussian":
            params.append(self.log_var) 
        if self.M > 0:
            params.append(self.Vs)  # Include Vs if M > 0
        return params

    def _random_rotation(self, D):
        """Generates a random rotation matrix of size (D, D)."""
        A = torch.randn(D, D, device=device) 
        Q, _ = torch.linalg.qr(A)
        return Q

    def _compute_mus(self, xs, zs, us=None):
        """Compute autoregressive means for each state."""
        T, D = xs.shape
        mus = torch.zeros(self.K, T, D, device=xs.device)
        
        for k in range(self.K):
            for t in range(T):
                if t < self.lags:
                    mus[k, t] = self.mu_init[k]  # initial mean for first L steps
                else:
                    mean_t = self.bs[k].clone()
                    if us is not None:
                        mean_t += self.Vs[k] @ us[t]  # Control input contribution
                    for l in range(self.lags):
                        Al = self.As[k][:, l * D:(l + 1) * D]  # Extract submatrix for lag l
                        mean_t += Al @ xs[t - l - 1]  # Add contribution from past state
                    mus[k, t] = mean_t
        return mus
    
    
    def dynamics_log_likelihoods(self, xs, zs, us=None, L=1):
        n_samples = xs.shape[0]
        mus = self._compute_mus(xs.reshape(-1, self.D), zs.reshape(-1, self.K))  # Flatten batch
        mus = mus.reshape(self.K, n_samples, -1, self.D)  # [K, n_samples, T, D]
        
        # Weighted mean (zs: [n_samples, T, K], mus: [K, n_samples, T, D])
        mu_weighted = torch.einsum('nkt,kntd->ntd', zs.permute(0,2,1), mus)
        
        # Covariances (shared across samples)
        Sigmas_init = self.sqrt_Sigmas_init @ self.sqrt_Sigmas_init.transpose(-1, -2)
        Sigmas = self.sqrt_Sigmas @ self.sqrt_Sigmas.transpose(-1, -2)
        sigma_init_weighted = torch.einsum('ntk,kij->ntij', zs, Sigmas_init)
        sigma_weighted = torch.einsum('ntk,kij->ntij', zs, Sigmas)
    
        # Batch MVN log-prob
        mvn_init = dist.MultivariateNormal(mu_weighted[:, :L], sigma_init_weighted[:, :L])
        mvn = dist.MultivariateNormal(mu_weighted[:, L:], sigma_weighted[:, L:])
        return torch.cat([mvn_init.log_prob(xs[:, :L]), mvn.log_prob(xs[:, L:])], dim=1).sum(dim=1)
    
    ''' ADDED: Transition likelihood calculation'''
    def log_transition_likelihoods(self, zs, xs, temperature):
        gs_z1 = dist.RelaxedOneHotCategorical(temperature, logits=self.logits_z1)
        log_prior_z1 = gs_z1.log_prob(zs[:, 0])  
        
        # Transition Probabilities
        log_Ps = torch.einsum('ntd,kd->ntk', xs[:, :-1], self.Rs) + self.r
        gs = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        log_probs = gs.log_prob(zs[:, 1:]).sum(dim=1)  # Sum over time
        # print(log_prior_z1, log_probs)
        return log_probs + log_prior_z1  

    
    def log_emission_likelihood(self, ys, xs):
        if self.emission_model == "poisson":
            lambdas = F.softplus(xs @ self.C.T + self.d)  # [n_samples, T, N]
            return (ys * torch.log(lambdas + 1e-8) - lambdas).sum(dim=(1, 2))  # Sum over T,N
        else:  # Gaussian
            mean = xs @ self.C.T + self.d  # [n_samples, T, N]
            variance = torch.exp(self.log_var)
            return -0.5 * (((ys - mean) ** 2) / variance + torch.log(2 * torch.pi * variance)).sum(dim=(1, 2))
    
    '''ADDED initializtion of xs0 based on observations'''
    def initialize_ms(self, ys):
        n_samples, T, N = ys.shape
        D = self.D
        if self.emission_model == "gaussian":
            # Gaussian emission - linear projection
            CtC = torch.matmul(self.C.T, self.C)
            reg = 1e-4 * torch.eye(D, device=ys.device)
            CtC_inv = torch.linalg.inv(CtC + reg)
            CtC_inv_Ct = torch.matmul(CtC_inv, self.C.T)
            y_centered = ys - self.d
            y_flat = y_centered.reshape(-1, N)
            x_flat = torch.matmul(y_flat.float(), CtC_inv_Ct.T)
            
        elif self.emission_model == "poisson":
            # Poisson emission - log transform + linear projection
            epsilon = 1e-3 
            # Transform observations 
            y_transformed = torch.log(ys + epsilon) - self.d # [n_samples, T, N]
            # Compute projection matrix (same as Gaussian case)
            CtC = torch.matmul(self.C.T, self.C)
            reg = 1e-4 * torch.eye(D, device=ys.device)
            CtC_inv = torch.linalg.inv(CtC + reg)
            CtC_inv_Ct = torch.matmul(CtC_inv, self.C.T)
            y_flat = y_transformed.reshape(-1, N)
            x_flat = torch.matmul(y_flat.float(), CtC_inv_Ct.T)
        
        return x_flat.reshape(n_samples, T, D)


### Variational distributions
### q(z)
class GSVariational(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.K = K
        self.D = D
        #W = torch.tensor([[0.5, 0.1], [-0.5, -0.1], [0.1, 0.5], [-0.1, -0.5]])  # Shape: (4, 2)
        #b = torch.tensor([-1, -1, 0., 0.])  # Shape: (4,)

        # Add Gaussian noise (less aggressive boundary)
        #noise_strength = 0.1
        #W_jittered = W + torch.randn_like(W) * noise_strength
        #b_jittered = b + torch.randn_like(b) * noise_strength
        
        self.W = nn.Parameter(0.8*torch.randn(K, D, device=device)) #nn.Parameter(W_jittered.to(device)) 
        self.b = nn.Parameter(torch.zeros(K, device=device)) #nn.Parameter(b_jittered.to(device)) 
        ### Initial state 
        #epsilon = 1e-2
        #logit_p1 = torch.log(torch.tensor((1 - epsilon) / epsilon, device=device))
        #logit_p_rest = torch.log(torch.tensor(epsilon / (K - 1), device=device))
        #self.logits_z1 = torch.cat([
        #    logit_p1.unsqueeze(0), 
        #    logit_p_rest.unsqueeze(0).repeat(K - 1)])
        self.logits_z1 = nn.Parameter(torch.zeros(K, device=device))
    @property
    def params(self):
        return [self.W, self.b, self.logits_z1]
    
    @params.setter
    def params(self, value):
        self.W.data, self.b.data = value

    def log_density(self, zs, xs, temperature):
        # Initial state
        gs_z1 = dist.RelaxedOneHotCategorical(temperature, logits=self.logits_z1)
        log_prior_z1 = gs_z1.log_prob(zs[:, 0])  # (batch_size,)
        
        # Transition Probabilities
        log_Ps = torch.einsum('ntd,kd->ntk', xs[:, :-1], self.W) + self.b
        gs = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        log_probs = gs.log_prob(zs[:, 1:]).sum(dim=1)  # Sum over time
        return log_probs + log_prior_z1
    
    '''ADDED sample zs from xs''' 
    def sample_q_z(self, xs, temperature):
        n_samples, T, D = xs.shape
        gs_z1 = dist.RelaxedOneHotCategorical(temperature,logits=self.logits_z1.unsqueeze(0).expand(n_samples, -1))
        z1 = gs_z1.sample()  # [n_samples, K]
        
        log_Ps = torch.einsum('ntd,kd->ntk', xs[:, :-1], self.W) + self.b  # [n_samples, T-1, K]
        gs = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        zt = gs.sample()  
        return torch.cat([z1.unsqueeze(1), zt], dim=1)  # [n_samples, T, K]




### q(x)
class AQbFunction(nn.Module):
    def __init__(self, K, D, model=None, ys=None, initial_variance=0.01):
        super().__init__()
        self.K = K
        self.D = D
        self.initial_variance = initial_variance

        # State-specific base matrices 
        self.A_base = nn.Parameter(0.8*torch.eye(D, device="cuda").repeat(K, 1, 1))  # (K, D, D)
        self.b_base = nn.Parameter(torch.zeros(K, D))             # (K, D)
        self.Q_sqrt = nn.Parameter(torch.eye(D).repeat(K, 1, 1))
   
    def forward(self, z_t):
        # print(z_t.shape, self.A_base.shape)
        A_t = torch.einsum('ntk,kij->ntij', z_t, self.A_base)
        b_t = torch.einsum('ntk,kd->ntd', z_t, self.b_base)
        Q_t = torch.einsum('ntk,kij->ntij', z_t, self.Q_sqrt)
        return A_t, Q_t, b_t  

    
class LDSVariational(nn.Module):
    def __init__(self, D, K, T, initial_variance=0.01):
        super().__init__()
        self.D = D
        self.T = T
        self.K = K
        self.initial_variance = initial_variance
        self.AQb_fn = AQbFunction(self.K, self.D)  ## NNs cond. on z
        self.ms = nn.Parameter(torch.zeros(T, D))
        self.Ri_sqrts = nn.Parameter((1.0 / torch.sqrt(torch.tensor(self.initial_variance))) * torch.eye(D).repeat(T, 1, 1))
        
    @property
    def params(self):
        return list(self.AQb_fn.parameters()) + [self.ms, self.Ri_sqrts]

        
    def sample_q_x(self, z_samples):
        """z_samples: [n_samples, T, K] -> returns [n_samples, T, D]"""
        # Get batch parameters
        n_samples = z_samples.shape[0]
        As, Qi_sqrts, bs = self.AQb_fn(z_samples[:, :-1])  # [n_samples, T-1, D, D]
        ms = self.ms.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, T, D]
        Ri_sqrts = self.Ri_sqrts.unsqueeze(0).expand(n_samples, -1, -1, -1)  # [n_samples, T, D, D]
        
        # Batched precision matrix
        J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)
        n_samples, T, D = h.shape
        # print(J_diag.shape, J_lower_diag.shape, h.shape)
        # Batched mean computation 
        mu = block_tridiagonal_solve(J_diag, J_lower_diag, h.view(n_samples, T*D))  # [n_samples, T*D]
        mu = mu.view(n_samples, -1, D)  # [n_samples, T, D]
    
        # Batched sampling
        L_diag, L_lower_diag = block_cholesky(J_diag, J_lower_diag)
        z = torch.randn(n_samples, h.shape[1]*h.shape[2], device=h.device)
        x = batch_block_cholesky_solve(L_diag, L_lower_diag, z)  # [n_samples, T*D]
        return x.view(n_samples, -1, D) + mu  # [n_samples, T, D]
  
    def mean_x(self, z_samples):
        n_samples = z_samples.shape[0]
        As, Qi_sqrts, bs = self.AQb_fn(z_samples[:, :-1])
        ms = self.ms.unsqueeze(0).expand(n_samples, -1, -1)
        Ri_sqrts = self.Ri_sqrts.unsqueeze(0).expand(n_samples, -1, -1, -1)
        
        J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)
        n_samples, T, D = h.shape
        mu = block_tridiagonal_solve(J_diag, J_lower_diag, h.view(n_samples, T*D))
        return mu.view(n_samples, -1, D)
    
    def log_density(self, x_samples, z_samples):
        n_samples, T, D = x_samples.shape
        As, Qi_sqrts, bs = self.AQb_fn(z_samples[:, :-1])  
        ms = self.ms.unsqueeze(0).expand(n_samples, -1, -1)  ## shared
        Ri_sqrts = self.Ri_sqrts.unsqueeze(0).expand(n_samples, -1, -1, -1)
        # Batch J, h computation 
        J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)

        xtJx = 0.5 * (x_samples * torch.matmul(J_diag, x_samples.unsqueeze(-1)).squeeze(-1)).sum(dim=(1, 2))
        xtJx += (x_samples[:, 1:] * torch.matmul(J_lower_diag, x_samples[:, :-1].unsqueeze(-1)).squeeze(-1)).sum(dim=(1, 2))

        hTx = (h * x_samples).sum(dim=(1, 2))
        # Log-det 
        L_diag, L_lower_diag = block_cholesky(J_diag, J_lower_diag)
        L_diag = torch.stack(L_diag, dim=1)
        signs, logabsdets = torch.linalg.slogdet(L_diag)  # shapes: both (batch, T)
        log_det_J = 2 * logabsdets.sum(dim=1)
        normalization_term = 0.5 * T * D * torch.log(torch.tensor(2 * torch.pi, dtype=x_samples.dtype, device=x_samples.device))
        return -xtJx + hTx + 0.5 * log_det_J - normalization_term
    
    
    
### Helper functions

def convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts):
    n_samples, T_minus_1, D, _ = As.shape
    T = T_minus_1 + 1
    
    # Batched inverse covariances
    Qis = Qi_sqrts @ Qi_sqrts.transpose(-1, -2)  # [n_samples, T-1, D, D]
    Ris = Ri_sqrts @ Ri_sqrts.transpose(-1, -2)  # [n_samples, T, D, D]

    # Off-diagonal blocks
    J_lower_diag = -torch.einsum('ntij,ntjk->ntik', Qis, As)  # -Qis @ As

    # Diagonal blocks
    J_diag = torch.zeros(n_samples, T, D, D, device=As.device)
    J_diag[:, :-1] = -torch.einsum('ntji,ntjk->ntik', As, J_lower_diag)  # -As^T @ J_lower_diag
    J_diag[:, 1:] += Qis  # Add Q_{t-1}^{-1} (shifted)
    J_diag += Ris  # Add R_t^{-1}

    # Linear term h
    h = torch.zeros(n_samples, T, D, device=As.device)
    h[:, :-1] = torch.einsum('ntij,ntj->nti', J_lower_diag, bs)  # J_lower_diag @ bs
    h[:, 1:] += torch.einsum('ntij,ntj->nti', Qis, bs)  # Qis @ bs (shifted)
    # print(Ris.shape, ms.shape)
    h += torch.einsum('ntij,ntj->nti', Ris, ms)  # Ris @ ms

    return J_diag, J_lower_diag, h


def block_cholesky(J_diag, J_lower_diag, epsilon=1e-2):
    n_samples, T, D, _ = J_diag.shape
    device = J_diag.device

    L_diag = []
    L_lower_diag = []

    # First block (batched Cholesky)
    L_diag.append(torch.linalg.cholesky(J_diag[:, 0] + epsilon * torch.eye(D, device=device)))

    for t in range(1, T):
        # Batched solve: L_lower_diag[t-1] = J_lower_diag[t-1] @ L_diag[t-1]⁻ᵀ
        L_lower_t = torch.linalg.solve(
            L_diag[t-1].transpose(-1, -2),  # [n_samples, D, D]
            J_lower_diag[:, t-1].transpose(-1, -2)  # [n_samples, D, D]
        ).transpose(-1, -2)
        L_lower_diag.append(L_lower_t)

        # Batched Schur complement
        Schur = J_diag[:, t] - L_lower_t @ L_lower_t.transpose(-1, -2)
        L_diag.append(torch.linalg.cholesky(Schur + epsilon * torch.eye(D, device=device)))

    return L_diag, L_lower_diag




def block_tridiagonal_solve(J_diag, J_lower_diag, h, epsilon=1e-2):
    batch_size, T, D, _ = J_diag.shape
    device = J_diag.device
    
    # Forward pass storage
    c = torch.zeros(batch_size, T-1, D, D, device=device)
    d = torch.zeros(batch_size, T, D, device=device)
    
    # --- Forward pass ---
    # First block
    J0 = J_diag[:, 0] + epsilon * torch.eye(D, device=device)
    c[:, 0] = torch.linalg.solve(J0, J_lower_diag[:, 0].transpose(-1, -2)).transpose(-1, -2)
    d[:, 0] = torch.linalg.solve(J0, h[:, :D])
    
    # Middle blocks
    for t in range(1, T-1):
        Jt = (J_diag[:, t] - torch.einsum('nij,njk->nik', J_lower_diag[:, t-1], c[:, t-1].clone()))  # Clone c
        Jt += epsilon * torch.eye(D, device=device)
    
        c[:, t] = torch.linalg.solve(Jt, J_lower_diag[:, t].transpose(-1, -2)).transpose(-1, -2)
        rhs_d = h[:, t*D:(t+1)*D] - torch.einsum('nij,nj->ni', 
                                               J_lower_diag[:, t-1], 
                                               d[:, t-1].clone())  # Clone d
        d[:, t] = torch.linalg.solve(Jt, rhs_d)
    
    # Last block
    JT = (J_diag[:, -1] - torch.einsum('nij,njk->nik', 
                                      J_lower_diag[:, -1], 
                                      c[:, -1].clone()))  # Clone c
    JT += epsilon * torch.eye(D, device=device)
    
    rhs_last = h[:, -D:] - torch.einsum('nij,nj->ni', 
                                      J_lower_diag[:, -1], 
                                      d[:, -2].clone())  # Clone d
    d[:, -1] = torch.linalg.solve(JT, rhs_last)
    
    # --- Backward pass ---
    x = torch.zeros_like(h)
    x[:, -D:] = d[:, -1]  
    
    for t in range(T-2, -1, -1):
        x_next = x[:, (t+1)*D:(t+2)*D].clone()
        x_t = d[:, t] - torch.einsum('nij,nj->ni', c[:, t].clone(), x_next)  # Clone c
        x[:, t*D:(t+1)*D] = x_t  # Safe assignment
    
    return x


def batch_block_cholesky_solve(L_diag, L_lower_diag, z, epsilon=1e-4):
    batch_size = z.shape[0]
    T = len(L_diag)
    D = L_diag[0].shape[-1]
    device = z.device
    x = torch.zeros_like(z)
    
    # Pre-compute regularized blocks 
    reg_L_diag = [L + epsilon * torch.eye(D, device=device).unsqueeze(0) 
                for L in L_diag]
    
    # First block 
    first_block = torch.linalg.solve(reg_L_diag[0].transpose(-1,-2), 
                                   z[:, :D].unsqueeze(-1)).squeeze(-1)
    x[:, :D] = first_block  # Safe, creates new tensor
    
    # Subsequent blocks
    for t in range(1, T):
        x_prev = x[:, (t-1)*D:t*D].clone()  
        
        # RHS computation 
        rhs = z[:, t*D:(t+1)*D] - (L_lower_diag[t-1].transpose(-1,-2) @ 
                                  x_prev.unsqueeze(-1)).squeeze(-1)
        x[:, t*D:(t+1)*D] = torch.linalg.solve(reg_L_diag[t].transpose(-1,-2), 
                                    rhs.unsqueeze(-1)).squeeze(-1)  
    return x


### Plotting utilities
def plot_states(z, ls="-", lw=1, alpha=1.0):
    T = z.size
    time = np.arange(T)  # Time axis
    colors = ['red','blue','green','gold']   
    # Plot each state segment
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))  # Change points
    for start, stop in zip(zcps[:-1], zcps[1:]):
        plt.plot(time[start:stop], z[start:stop],
                 lw=lw, ls=ls,
                 color=colors[z[start] % len(colors)],
                 alpha=alpha)
    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Discrete States Over Time")
    plt.show()

def plot_trajectory(z, x, ls="-"):
    colors = ['red','blue','green','gold']   
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    for start, stop in zip(zcps[:-1], zcps[1:]):
        plt.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    # Add labels and legend
    plt.title("Trajectories per states")
    plt.show()