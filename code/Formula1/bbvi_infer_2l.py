'''GS-BBVI for 2-level GDM'''
import torch
import math
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)  # Set PyTorch seed
torch.cuda.manual_seed_all(seed)  # Set CUDA seed (if available)
np.random.seed(seed)  # Set NumPy seed
random.seed(seed)  # Set Python random seed
torch.backends.cudnn.deterministic = True  # Ensure CuDNN is deterministic
torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking


### GS-BBVI algorithm
def fit_bbvi_schedule(model, ys, num_iters=100, learning=True, n_samples=10, 
                     base_lr=1e-2, warmup_iters=30, tau_min=0.99, tau_max=0.99, verbose=False):
    device = ys.device
    tau = torch.tensor(tau_max, device=device)
    elbo_history = []
    convergence_window = 100  ## determines strictness of convergence
    
    # --- Initialize Variational Distribution ---
    T, N = ys.shape
    K, D = model.K, model.D
    variational_z = GSVariational(N, K).to(device)

    # --- Optimizer Setup ---
    params = (model.params + variational_z.params) if learning else (variational_z.params)
    optimizer = AdamW(params, lr=base_lr, weight_decay=1e-4)  # Start at min_lr

    warmup = ConstantLR(optimizer,factor=1,total_iters=warmup_iters)
    decay = LinearLR(optimizer,start_factor=1.0, end_factor=0.1,total_iters=2*warmup_iters)
    hold = ConstantLR(optimizer,factor=0.1,total_iters=float('inf'))
    
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup, decay, hold],
                           milestones=[warmup_iters, 3*warmup_iters])
    
    temp_scheduler = DelayedCosineTempSchedule(temp_init=tau_min,temp_max=tau_max,delay_epochs=warmup_iters, rise_epochs=2*warmup_iters)
    
    # --- Training Loop ---
    pbar = trange(num_iters)
    for itr in pbar:
        optimizer.zero_grad()
        
        # --- Samples ---
        ys_batch = ys.unsqueeze(0).expand(n_samples, -1, -1)
        zs = variational_z.sample_q_z(ys_batch, tau) 
        # --- ELBO COMPUTATION ---
        def ELBO_objective(params, itr, zs, tau):
            log_pz = model.log_transition_likelihoods(zs, ys_batch, tau)   
            log_py = model.log_emission_likelihood(ys, zs)
            log_qz = variational_z.log_density(zs, ys_batch, tau)
            ## encourage diverse states
            state_counts = zs.mean(dim=(0,1))  # [K] average usage per state
            state_entropy = -(state_counts * torch.log(state_counts + 1e-8)).sum()
            
            ### Optional plotting
            if verbose and (itr % 200 == 0 or itr == num_iters-1):
                plot_states(torch.argmax(torch.mean(zs, dim=0), dim=1).cpu().numpy())
                ## Latent trajectory
                plot_trajectory(
                    torch.argmax(torch.mean(zs, dim=0), dim=1).cpu().numpy(),
                    torch.einsum('tn,dn->td', ys, model.F).detach().cpu().numpy(), 
                    ls="-", title='μ'
                )
                ## Predicted observation
                y_preds0=model.smooth(ys, zs)
                plot_observations(torch.argmax(torch.mean(zs, dim=0), dim=1).cpu().numpy(), 
                                  y_preds0.mean(dim=0)[:400,:1].detach().cpu().numpy())
                print("R2:", train_metrics(ys, y_preds0.mean(dim=0)))
            return (log_pz + log_py - log_qz).mean() + state_entropy # - 10*scale_loss
        
        # --- Gradient Handling ---
        elbo = ELBO_objective(params, itr, zs, tau)
        loss = -elbo
        loss.backward()
        
        # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(params, max_norm=1)
        optimizer.step()
        scheduler.step()
        tau = temp_scheduler.step()
        tau = torch.tensor(tau, device=device)
        
        # --- Monitoring ---
        elbo_history.append(-loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"ELBO: {elbo_history[-1]:.1f}, LR: {current_lr:.5f}, Tau: {tau.item():.3f}")
        
        # --- EARLY STOPPING ---
        if len(elbo_history) > convergence_window:
            elbo_improvement =  np.mean(np.diff(elbo_history[-convergence_window:]))
            if elbo_improvement < 1e-4:  ## avg ELBO not increasing
                print(f"Early stopping at iteration {itr}.")
                break
    return np.array(elbo_history), variational_z



### Model likelihood
class GenerativeSLDS(nn.Module):
    def __init__(self, N, K, D, M=0, lags=1, emission_model="gaussian"):
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

        # Optional control input matrices V_k (K, N, M)
        self.Vs = nn.Parameter(torch.randn(K, N, M, device=device) * 0.1) if M > 0 else None
        # Initial mean values (K, D)
        self.mu_init = nn.Parameter(torch.randn(K, N, device=device)) 
             
        '''Discrete latent variables z_{1:T} for state switching'''
        self.F = nn.Parameter(torch.randn(D, N, device=device) * 0.1)  
        self.Rs = nn.Parameter(torch.randn(K, D, device=device))
        self.r = nn.Parameter(torch.zeros(K, device=device))
        '''Stickiness parameter gamma'''
        self.gamma = nn.Parameter(torch.tensor(0.1, device=device))  ## stickiness 
        self.logits_z1 = nn.Parameter(torch.zeros(K, device=device))
        
        '''Observation parameters'''
        # Random projection matrix
        C = torch.randn(N, D, device=device) 
        # Random transition matrices A_k (K, D, D*lags)
        As = torch.stack([torch.stack([self._random_rotation(D) for _ in range(lags)], dim=0) for _ in range(K)],dim=0).to(device)
        self.Ss = nn.Parameter(torch.einsum("nd,kldm->klnm", C, As))
        # Bias term b_k (K, D)
        self.bs = nn.Parameter(torch.zeros(K, N, device=device))  # Initialize as zeros
        # Additional parameter for Gaussian emission model
        if self.emission_model == "gaussian":
            self.log_var = nn.Parameter(-1.0 + torch.rand(1, N, device=device))
            # self.L = nn.Parameter(torch.eye(N, device=device).unsqueeze(0).expand(1, N, N))
    
    @property
    def params(self):
        params = [
            self.mu_init, self.Ss, self.bs,  # Observation parameters (y)
            self.Rs, self.r, self.gamma, self.logits_z1 # Discrete state parameters (z)
        ]
        if self.emission_model == "gaussian":
            params.append(self.log_var)
            #params.append(self.L) 
        if self.M > 0:
            params.append(self.Vs)  # Include Vs if M > 0
        return params
    
    
    def _random_rotation(self, D):
        A = torch.randn(D, D, device=device) 
        Q, _ = torch.linalg.qr(A)
        return Q

    def _compute_mus(self, projected_ys, us=None):
        T, D = projected_ys.shape
        device = projected_ys.device
        mus = torch.zeros(self.K, T, self.N, device=device)
        # Initial lags
        mus[:, :self.lags] = self.mu_init.unsqueeze(1)  # [K, lags, N]
        if T <= self.lags:
            return mus  
        lagged_ys = torch.stack([projected_ys[self.lags - l - 1:T - l - 1] for l in range(self.lags)], dim=1)
        contributions = torch.einsum('klnd,tld->ktn', self.Ss, lagged_ys)
        mus[:, self.lags:] = contributions + self.bs.unsqueeze(1)  # [K, T-lags, N]
    
        # Add control input if provided
        if us is not None:
            control_contrib = torch.einsum('knm,tm->ktn', self.Vs, us[self.lags:])
            mus[:, self.lags:] += control_contrib  
        return mus
    

    def log_transition_likelihoods(self, zs, ys, temperature):
        zs_normalized = zs / zs.sum(dim=-1, keepdim=True)
        gs_z1 = dist.RelaxedOneHotCategorical(temperature, logits=self.logits_z1)
        log_prior_z1 = gs_z1.log_prob(zs_normalized[:, 0])  
        # Transition Probabilities
        ''' NEW: Modified transition probabilities using observations'''
        projected_ys = torch.einsum('mtn,dn->mtd', ys[:, :-1], self.F)  # [n_samples, T-1, D]
        
        ''' NEW: Add stickiness'''
        trans = torch.einsum('mtd,kd->mtk', projected_ys, self.Rs) + self.r
        sticky = zs[:,:-1]      ## no stickiness gamma=0    
        log_Ps = (1 - self.gamma) * trans + self.gamma * sticky
        #log_Ps = torch.einsum('mtd,kd->mtk', projected_ys, self.Rs) + self.r  # [n_samples,T-1, K]    
        gs = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        log_probs = gs.log_prob(zs_normalized[:, 1:]).sum(dim=1)  # Sum over time
        return log_probs + log_prior_z1

    

    def log_emission_likelihood(self, ys, zs):
        n_samples = zs.shape[0]
        projected_ys = torch.einsum('tn,dn->td', ys, self.F)  # [T, D]
        mus = self._compute_mus(projected_ys) 
        # mus = mus.reshape(self.K, n_samples, -1, self.N)  # [K, n_samples, T, N]
        # Weighted mean (zs: [n_samples, T, K], mus: [K, n_samples, T, N])
        mean = torch.einsum('mtk,ktn->mtn', zs, mus)
        
        if self.emission_model == "poisson":
            lambdas = F.softplus(mean)
            log_lik = ys.unsqueeze(0)*torch.log(lambdas+1e-8) - lambdas
        else:
            variance = torch.exp(self.log_var)
            log_lik = -0.5 * (((ys.unsqueeze(0) - mean) ** 2) / variance + torch.log(2 * torch.pi * variance))
        return log_lik.sum(dim=(1,2))
   
    
    def smooth(self, ys, expected_states):
        projected_ys = torch.einsum('tn,dn->td', ys, self.F)  #[T, D]
        mus = self._compute_mus(projected_ys) 
        yhat = torch.einsum('mtk,ktn->mtn', expected_states, mus)  #[n_samples, T, N]
        return yhat


### Variational distributions
### q(z)
class GSVariational(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.K = K
        self.N = N      
        self.W = nn.Parameter(0.8 * torch.randn(K, N, device=device)) 
        self.b = nn.Parameter(torch.zeros(K, device=device)) 

    @property
    def params(self):
        return [self.W, self.b]

    def log_density(self, zs, ys, temperature):
        # Normalize z samples
        zs_normalized = zs / (zs.sum(dim=-1, keepdim=True) + 1e-8)  
        # Logits for q(z | y)
        log_Ps = torch.einsum('mtn,kn->mtk', ys, self.W) + self.b  # (n_samples, T, K)
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        # log q(z | y)
        log_probs = q_dist.log_prob(zs_normalized).sum(dim=1)  # (n_samples,)
        return log_probs

    def sample_q_z(self, ys, temperature):
        n_samples, T, N = ys.shape
        log_Ps = torch.einsum('mtn,kn->mtk', ys, self.W) + self.b  # (n_samples, T, K)
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        zs = q_dist.rsample()  # reparameterized sample that allows gradient flow
        return zs  # (n_samples, T, K)

    
### RNN q(z)
class RNNGSVariational(nn.Module):
    def __init__(self, N, K, hidden_dim=64, bidirectional=True):
        super().__init__()
        self.K = K
        self.N = N
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # RNN encoder over y_t
        self.rnn = nn.GRU(
            input_size=N,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        # Map RNN hidden states to variational logits over z_t
        self.out = nn.Linear(rnn_out_dim, K)

    @property
    def params(self):
        return list(self.parameters())

    def _logits(self, ys):
        h, _ = self.rnn(ys)          # (n_samples, T, rnn_out_dim)
        logits = self.out(h)         # (n_samples, T, K)
        return logits

    def log_density(self, zs, ys, temperature):
        # Normalize z samples
        zs_normalized = zs / (zs.sum(dim=-1, keepdim=True) + 1e-8)
        logits = self._logits(ys)   # (n_samples, T, K)
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        # log q(z|y)
        log_probs = q_dist.log_prob(zs_normalized).sum(dim=1)  # (n_samples,)
        return log_probs

    def sample_q_z(self, ys, temperature):
        logits = self._logits(ys)    # (n_samples, T, K)
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        zs = q_dist.rsample()        # reparameterized sample (n_samples, T, K)
        return zs

###---------------------------------------------- Helper function -----------------------------------------------------###    
### Plotting utilities
## discrete states
def plot_states(z, ls="-", lw=1, alpha=1.0):
    T = z.size
    time = np.arange(T)  # Time axis
    n_states = z.max() + 1
    cmap_obj = plt.get_cmap("tab10", n_states)
    # Plot each state segment
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))  # Change points
    for start, stop in zip(zcps[:-1], zcps[1:]):
        plt.plot(time[start:stop], z[start:stop],
                 lw=lw, ls=ls,
                 color=cmap_obj.colors[z[start]],
                 alpha=alpha)
    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Discrete States Over Time")
    plt.show()

## soft states    
def plot_gs_states(zs):
    T, K = zs.shape
    cmap_obj = plt.get_cmap("Accent", K)
    fig, axes = plt.subplots(K, 1, figsize=(12, 2 * K), sharex=True)
    for k in range(K):
        axes[k].plot(zs[:, k], label=f"State {k+1}", color=cmap_obj.colors[k])
        axes[k].set_ylabel(f"$z_t^{k+1}$")
        axes[k].legend(loc="upper right")
        axes[k].grid(True)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Gumbel-Softmax State Proportions Over Time", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
## latent trajectory
def plot_trajectory(z, x, ls="-", title=''):
    n_states = z.max() + 1
    cmap_obj = plt.get_cmap("Accent", n_states)   
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    for start, stop in zip(zcps[:-1], zcps[1:]):
        plt.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=cmap_obj.colors[z[start]],
                alpha=1.0)
    # Add labels and legend
    plt.title(title)
    plt.show()
    
    
def plot_observations(z, y, ls="-", lw=1, embed=True):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    n_states = z.max() + 1
    cmap_obj = plt.get_cmap("Accent", n_states)
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            plt.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=cmap_obj.colors[z[start]],
                    alpha=1.0)
    if embed:
        plt.title("Predicted observations over time")
        plt.show()
    
### Temperature annealing
class DelayedCosineTempSchedule:
    def __init__(self, temp_init=0.5, temp_max=0.99, delay_epochs=100, rise_epochs=100):
        """
        Args:
            temp_init: Initial τ (0.5)
            temp_max: Maximum τ (0.99)
            delay_epochs: Hold τ=0.5 for start epochs
            rise_epochs: Cosine-increase duration after start
        """
        self.temp = temp_init
        self.epoch = 0
        self.temp_init = temp_init
        self.temp_max = temp_max
        self.delay_epochs = delay_epochs
        self.rise_epochs = rise_epochs

    def step(self):
        if self.epoch < self.delay_epochs:
            # Phase 1: Hold at τ=0.5
            self.temp = self.temp_init
        elif self.epoch < self.delay_epochs + self.rise_epochs:
            # Phase 2: Cosine-increase to τ=0.99
            progress = (self.epoch - self.delay_epochs) / self.rise_epochs
            self.temp = self.temp_init + 0.5 * (self.temp_max - self.temp_init) * \
                        (1 - math.cos(math.pi * progress))  # Smooth rise
        else:
            # Phase 3: Hold at τ=0.99 indefinitely
            self.temp = self.temp_max
        
        self.epoch += 1
        return self.temp

    
###---------------------------------------------- Model performance -----------------------------------------------------###    

def compute_r2(ys_pred, ys_test):
    ss_total = np.sum((ys_test - ys_test.mean(axis=0, keepdims=True))**2, axis=0)
    ss_residual = np.sum((ys_test - ys_pred)**2, axis=0)
    r2_per_unit = 1 - (ss_residual / (ss_total + 1e-8))
    return np.mean(r2_per_unit)


def train_metrics(ys, pred_ys0, y_preds=None, k_max=0):
    if torch.is_tensor(ys): ys = ys.detach().cpu().numpy()
    if torch.is_tensor(pred_ys0): pred_ys0 = pred_ys0.detach().cpu().numpy()
    if y_preds is not None and torch.is_tensor(y_preds): 
        y_preds = y_preds.detach().cpu().numpy()
    r2_scores = []
    for k in range(k_max + 1):
        if k == 0:
            r2 = compute_r2(pred_ys0, ys)
        else:
            r2 = compute_r2(y_preds[k:, k - 1], ys[k:])
        r2_scores.append(r2)
    return r2_scores


def predict_k_step_more(k_max, model, variational_z, ys, n_trajectory=20, n_samples=100, temperature=0.99):
    ys_tensor = torch.tensor(ys).to('cuda').float()
    T, N = ys_tensor.shape
    device = ys_tensor.device
    y_preds = torch.zeros((T, k_max, N), device=device)  ## prediction results
    y_input = ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1).clone()   ## intermediate predictions
    z_preds = variational_z.sample_q_z(y_input, temperature)  # intermediate states
    
    for k in range(1, k_max + 1):  ## start from 1-step forward
        projected = torch.einsum('mtn,dn->mtd', y_input, model.F)  # [n_trajectory,T,D]
        #gs_z1 = dist.RelaxedOneHotCategorical(temperature,logits=model.logits_z1.unsqueeze(0).expand(n_trajectory, -1))
        #z1 = gs_z1.sample()
        # Predict next z from x
        trans = torch.einsum('ntd,kd->ntk', projected[:,:-1], model.Rs) + model.r
        sticky = z_preds[:,:-1]      ## no stickiness gamma=0    
        logits_z = (1 - model.gamma) * trans + model.gamma * sticky  
        logits_exp = logits_z.unsqueeze(0).expand(n_samples, -1, -1, -1)
        z_full = dist.RelaxedOneHotCategorical(temperature, logits=logits_exp).sample()  # [n_samples, n_trajectory, T-1, K]
        z_t = z_full.mean(dim=0)  
        z_preds[:,1:] = z_t
        
        y_next = model.smooth(y_input.mean(dim=0), z_preds)  # [n_trajectory,T,N]                 
        y_preds[k:, k-1] = y_next[:,k:].mean(dim=0)
        ## update y_input
        y_input[:,k:] = y_next[:,k:]
    return y_preds.detach().cpu().numpy(), z_preds


@torch.no_grad()
def predict_k_steps_full(model, variational_z, ys, k=2, n_trajectory=10, temperature=0.99):
    ys_tensor = torch.tensor(ys, device='cuda').float()  # [T, N]
    T, N = ys_tensor.shape
    K, D = model.K, model.D
    z_t = variational_z.sample_q_z(ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1), temperature)
    y_input = ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1).clone()   ## intermediate predictions
    y_preds = []
    for step in range(1, k):  ## 1-step ahead
        projected = torch.einsum('mtn,dn->mtd', y_input, model.F)  # [n_paths,T,D]
        trans = torch.einsum('ntd,kd->ntk', projected[:,:-1], model.Rs) + model.r  # [n_paths, T-1, K]
        z_prev = z_t[:, :-1]                 # [n_paths, T-1, K]
        logits_next = (1 - model.gamma) * trans + model.gamma * z_prev   # [n_paths, T-1, K]    
        # Expand to all possibilities 
        logits_exp = logits_next.unsqueeze(0).expand(n_trajectory, -1, -1, -1)  # [n_trajectory, n_paths, T-1, K]
        z_next_full = dist.RelaxedOneHotCategorical(temperature, logits=logits_exp).sample()  # [n_trajectory, n_paths, T-1, K]           
        # Flatten out the possibilities
        z_next = z_next_full.reshape(-1, T-1, K)        # [n_paths*n_trajectory, T-1, K]
        # Repeat previous trajectories to match branching
        z_t = z_t.repeat_interleave(n_trajectory, dim=0)   # [n_paths*n_trajectory, T, K]
        z_t[:,1:] = z_next
        ## predictions
        y_t = model.smooth(y_input.mean(dim=0), z_t)
        y_preds.append(y_t)                # store full predicted trajectories
        y_input = y_t.clone()
    return y_preds, z_t



@torch.no_grad()
def predict_k_steps_full(model, variational_z, ys, k=2, n_trajectory=10, temperature=0.99):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ys_tensor = torch.tensor(ys, device=device).float()  # [T, N]
    T, N = ys_tensor.shape
    K, D = model.K, model.D

    # Initial sample of z (n_trajectory independent starts)
    z_paths = variational_z.sample_q_z(
        ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1), temperature
    )  # [n_trajectory, T, K]

    # Initial input Ys for each current path: start with true observed ys replicated
    y_inputs = ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1).clone()  # [n_trajectory, T, N]

    # Compute initial predicted outputs per initial path (uses true ys to compute mus)
    # smooth expects (ys_single, expected_states) where expected_states can be [m,T,K];
    # we can pass z_paths of shape [n_trajectory, T, K] directly and get [n_trajectory, T, N]
    y_pred_init = model.smooth(ys_tensor, z_paths)  # [n_trajectory, T, N]
    y_preds = [y_pred_init.detach()]                # store initial predictions
    # update y_inputs to the predicted ones (so next step uses predicted ys per path)
    y_inputs = y_pred_init.clone()                  # [n_trajectory, T, N]

    for step in range(1, k):
        n_paths = z_paths.shape[0]  # current flattened number of trajectories

        # Project current y_inputs -> [n_paths, T, D]
        projected = torch.einsum('mtn,dn->mtd', y_inputs, model.F)

        # Transition logits per path: [n_paths, T-1, K]
        trans = torch.einsum('ntd,kd->ntk', projected[:, :-1], model.Rs) + model.r
        z_prev = z_paths[:, :-1]  # [n_paths, T-1, K]
        logits_next = (1 - model.gamma) * trans + model.gamma * z_prev  # [n_paths, T-1, K]

        # Expand each current path into n_trajectory children
        # logits_exp shape: [n_paths, n_trajectory, T-1, K]
        logits_exp = logits_next.unsqueeze(1).expand(-1, n_trajectory, -1, -1)
        z_next_full = dist.RelaxedOneHotCategorical(temperature, logits=logits_exp).sample()
        # z_next_full: [n_paths, n_trajectory, T-1, K]

        # Build children z: shape [n_paths, n_trajectory, T, K]
        z_children = z_paths.unsqueeze(1).expand(-1, n_trajectory, -1, -1).clone()
        z_children[:, :, 1:] = z_next_full
        # Flatten children into shape [n_paths * n_trajectory, T, K]
        z_flat = z_children.reshape(-1, T, K)

        # Prepare corresponding y_inputs for each flattened child:
        # parent y_inputs has shape [n_paths, T, N]; repeat/expand to [n_paths, n_trajectory, T, N] then flatten
        y_inputs_rep = y_inputs.unsqueeze(1).expand(-1, n_trajectory, -1, -1).reshape(-1, T, N)
        # Now for each flattened child, call smooth with that child's own ys (parent ys initially),
        # and the child's z (shape [T,K], we pass as [1,T,K] so smooth returns [1,T,N]).
        y_pred_list = []
        for i in range(z_flat.shape[0]):
            # Use the parent's ys (which we just repeated); smooth will compute mus from that ys
            # and combine with the child's expected_states (z_flat[i:i+1]) to produce [1,T,N]
            y_hat_i = model.smooth(y_inputs_rep[i], z_flat[i:i+1])  # [1, T, N]
            y_pred_list.append(y_hat_i.squeeze(0))                 # [T, N]

        # Stack predicted ys for all children: [n_paths * n_trajectory, T, N]
        y_pred_flat = torch.stack(y_pred_list, dim=0)

        # Store and update for next iteration (flattened form)
        y_preds.append(y_pred_flat.detach())
        z_paths = z_flat
        y_inputs = y_pred_flat.clone()

    # After loop:
    # y_preds is a list: y_preds[0] shape [n_trajectory, T, N], y_preds[1] shape [n_trajectory^2, T, N], ...
    # z_paths is final flattened latent trajectories: [n_trajectory**(k), T, K] if you ran k-1 expansions
    return y_preds, z_paths
