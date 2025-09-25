'''Batched RNN version GS-BBVI for 2-level GDM'''
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
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)  # Set PyTorch seed
torch.cuda.manual_seed_all(seed)  # Set CUDA seed (if available)
np.random.seed(seed)  # Set NumPy seed
random.seed(seed)  # Set Python random seed
torch.backends.cudnn.deterministic = True  # Ensure CuDNN is deterministic
torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking


### GS-BBVI algorithm
def fit_bbvi_schedule(model, trials, num_iters=100, learning=True, n_samples=10, 
                     base_lr=1e-2, warmup_iters=30, tau_min=0.99, tau_max=0.99, 
                     batch_size=2, verbose=False):
    
    device = next(model.parameters()).device
    tau = torch.tensor(tau_max, device=device)
    elbo_history = []
    convergence_window = 1000
    
    # --- Variational distribution ---
    N = trials[0].shape[1]  # assume consistent across trials
    K, D = model.K, model.D
    variational_z = GSVariational(N, K).to(device)

    # --- Optimizer setup ---
    params = (model.params + variational_z.params) if learning else (variational_z.params)
    optimizer = AdamW(params, lr=base_lr, weight_decay=1e-4)

    warmup = ConstantLR(optimizer, factor=1, total_iters=warmup_iters)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=2*warmup_iters)
    hold = ConstantLR(optimizer, factor=0.1, total_iters=float('inf'))
    
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup, decay, hold],
                           milestones=[warmup_iters, 3*warmup_iters])
    
    temp_scheduler = DelayedCosineTempSchedule(temp_init=tau_min,temp_max=tau_max,delay_epochs=warmup_iters, rise_epochs=2*warmup_iters)
    # --- Dataset Loader ---
    dataset = TrialsDataset(trials)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    pbar = trange(num_iters)
    # --- Training Loop ---
    for itr in pbar:
        for ys_batch, mask, lengths in loader:  ## mini-batches
            ys_batch, mask = ys_batch.to(device), mask.to(device)   # [B, T, N], [B, T]
        
            # --- Expand for n_samples ---
            ys_batch = ys_batch.unsqueeze(0).expand(n_samples, -1, -1, -1)   # [M, B, T, N]
            mask = mask.unsqueeze(0).expand(n_samples, -1, -1)               # [M, B, T]
            optimizer.zero_grad()
            
            # --- Sample q(z|y) ---
            zs = variational_z.sample_q_z(ys_batch, tau)   # [M, B, T, K]
        
            # --- ELBO ---
            def ELBO_objective(zs, tau):
                log_pz = model.log_transition_likelihoods(zs, ys_batch, tau, mask)
                log_py = model.log_emission_likelihood(ys_batch, zs, mask)
                log_qz = variational_z.log_density(zs, ys_batch, tau, mask)

                # Encourage diverse state usage
                state_counts = zs.mean(dim=(0,1,2))  # [K] averaged across M,B,T
                state_entropy = -(state_counts * torch.log(state_counts + 1e-8)).sum()

                if verbose and (itr % 200 == 0 or itr == num_iters-1):
                    inferred_states = torch.argmax(torch.mean(zs, dim=0), dim=-1)  # [B,T]
                    plot_states(inferred_states[0].cpu().numpy())  # plot first trial
                        
                    with torch.no_grad():
                        # Use first trial (B=0)
                        ys_first = ys_batch[0, 0]   # [T,N], just one trial
                        zs_first = zs[:, 0]               # [M,T,K], samples for first trial
                        # Latent trajectory
                        plot_trajectory(
                            torch.argmax(zs_first.mean(dim=0), dim=1).cpu().numpy(),   # [T]
                            torch.einsum('tn,dn->td', ys_first, model.F).cpu().numpy(),
                            ls="-", title="μ (first trial)"
                        )
                    
                        # Predicted observation
                        y_preds0 = model.smooth(ys_first, zs_first)
                        plot_observations(
                            torch.argmax(zs_first.mean(dim=0), dim=1).cpu().numpy(),
                            y_preds0.mean(dim=0)[:400, :1].cpu().numpy())

                        print("R2:", train_metrics(torch.tensor(ys_first), y_preds0.mean(dim=0).cpu()))
                    
                return (log_pz + log_py - log_qz).mean() + state_entropy

            elbo = ELBO_objective(zs, tau)
            loss = -elbo
            loss.backward()
           
            torch.nn.utils.clip_grad_norm_(params, max_norm=1)
            optimizer.step()
            scheduler.step()
            tau = torch.tensor(temp_scheduler.step(), device=device)

            elbo_history.append(-loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"ELBO: {elbo_history[-1]:.1f}, LR: {current_lr:.5f}, Tau: {tau.item():.3f}")

            # --- Early stopping ---
            if len(elbo_history) > convergence_window:
                elbo_improvement = np.mean(np.diff(elbo_history[-convergence_window:]))
                if elbo_improvement < 1e-4:
                    print(f"Early stopping at iteration {itr}.")
                    return np.array(elbo_history), variational_z

    return np.array(elbo_history), variational_z



class TrialsDataset(Dataset):
    def __init__(self, trials):
        # trials = list of tensors, each [T_i, N]
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]

def pad_collate(batch):
    # batch = list of [T_i, N] tensors
    lengths = [b.shape[0] for b in batch]
    N = batch[0].shape[1]
    max_len = max(lengths)

    padded = torch.zeros(len(batch), max_len, N, device=device)
    mask = torch.zeros(len(batch), max_len, device=device)

    for i, b in enumerate(batch):
        T = b.shape[0]
        padded[i, :T] = b
        mask[i, :T] = 1.0

    return padded, mask, lengths


### Model likelihood
class GenerativeSLDS(nn.Module):
    def __init__(self, N, K, D, M=0, lags=1, emission_model="gaussian", hidden_dim=16):
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
        self.trans_rnn = nn.GRU(
            input_size=D,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.trans_out = nn.Linear(hidden_dim, K)
        self.logits_z1 = nn.Parameter(torch.zeros(K, device=device)) 
        self.F = nn.Parameter(torch.randn(D, N, device=device) * 0.1)  
        
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
        params = [self.mu_init, self.F, self.Ss, self.bs, self.logits_z1]
        if self.emission_model == "gaussian":
            params.append(self.log_var)
        if self.M > 0:
            params.append(self.Vs)
        params += list(self.trans_rnn.parameters()) + list(self.trans_out.parameters())
        return params
    
    
    def _random_rotation(self, D):
        A = torch.randn(D, D, device=device) 
        Q, _ = torch.linalg.qr(A)
        return Q

    def _compute_mus(self, projected_ys, us=None, mask=None):
        # single-trial (unbatched)
        if projected_ys.dim() == 2:
            T, D = projected_ys.shape
            device = projected_ys.device
            mus = torch.zeros(self.K, T, self.N, device=device) 
            # initial lags
            if self.lags > 0:
                mus[:, :self.lags] = self.mu_init[:, None, :].expand(-1, self.lags, -1)  # [K, lags, N]
            if T > self.lags:
                # lagged_ys: [T-lags, lags, D]
                lagged_ys = torch.stack([
                    projected_ys[self.lags - l - 1 : T - l - 1]
                    for l in range(self.lags)
                ], dim=1)
                # contributions: [K, T-lags, N]
                contributions = torch.einsum('klnd,tld->ktn', self.Ss, lagged_ys)
                mus[:, self.lags:] = contributions + self.bs.unsqueeze(1)  # [K, T-lags, N]
    
            # control input (single-trial)
            if us is not None:
                control_contrib = torch.einsum('knm,tm->ktn', self.Vs, us[self.lags:])
                mus[:, self.lags:] += control_contrib
    
            # mask (single-trial) mask shape expected [T] or [1,T]
            if mask is not None:
                mask_vec = mask.squeeze()  # [T]
                mus = mus * mask_vec[None, :, None]
    
            return mus  # [K, T, N]
    
        # batched version
        elif projected_ys.dim() == 3:
            B, T, D = projected_ys.shape
            device = projected_ys.device
            mus = torch.zeros(self.K, B, T, self.N, device=device)  # [K,B,T,N]
    
            # initial lags: expand mu_init across batch and lags
            if self.lags > 0:
                mus[:, :, :self.lags] = self.mu_init[:, None, None, :].expand(-1, B, self.lags, -1)
    
            if T > self.lags:
                # lagged_ys: [B, T-lags, lags, D]
                lagged_ys = torch.stack([
                    projected_ys[:, self.lags - l - 1 : T - l - 1]
                    for l in range(self.lags)
                ], dim=2)  # [B, T-lags, lags, D]
    
                # contributions: 'klnd,btld->kbtn' -> [K,B,T-lags,N]
                contributions = torch.einsum('klnd,btld->kbtn', self.Ss, lagged_ys)
                mus[:, :, self.lags:] = contributions + self.bs[:, None, None, :]
    
            # control input (batched)
            if us is not None:
                # us expected shape [B, T, M]
                control_contrib = torch.einsum('knm,btm->kbtn', self.Vs, us[:, self.lags:])
                mus[:, :, self.lags:] += control_contrib
    
            # mask (batched) expected shape [B, T]
            if mask is not None:
                mus = mus * mask.unsqueeze(0).unsqueeze(-1)  # [K,B,T,N]
    
            return mus  # [K, B, T, N]
        
    

    def log_transition_likelihoods(self, zs, ys, temperature, mask=None):
        zs_normalized = zs / (zs.sum(dim=-1, keepdim=True) + 1e-8)

        # Prior over z1
        gs_z1 = dist.RelaxedOneHotCategorical(temperature, logits=self.logits_z1)
        log_prior_z1 = gs_z1.log_prob(zs_normalized[:, :, 0])  # [M, B]

        # Project ys for RNN transition
        projected_ys = torch.einsum('mbtn,dn->mbtd', ys[:, :, :-1], self.F)  # [M, B, T-1, D]
        h, _ = self.trans_rnn(projected_ys.reshape(-1, projected_ys.shape[2], projected_ys.shape[3]))
        h = h.view(zs.shape[0], zs.shape[1], -1, h.shape[-1])  # [M,B,T-1,H]
        logits = self.trans_out(h)  # [M,B,T-1,K]

        gs = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        log_probs = gs.log_prob(zs_normalized[:, :, 1:])  # [M,B,T-1]

        # Mask out padded timesteps
        if mask is not None:
            log_probs = log_probs * mask[:, :, 1:]

        return log_prior_z1 + log_probs.sum(dim=2)  # [M,B]


    
    
    def log_emission_likelihood(self, ys, zs, mask=None):
        M, B, T, N = ys.shape
        # Project observations (same projection for all M samples) 
        projected_all = torch.einsum('mbtn,dn->mbtd', ys, self.F)  # [M,B,T,D] 
        # Use the first MC slice to compute mus per batch
        projected_b = projected_all[0]  # [B, T, D] 
        # compute mus shape [K, B, T, N]
        mus = self._compute_mus(projected_b, us=None, mask=(mask[0] if mask is not None else None))
        # Now combine zs (M,B,T,K) with mus (K,B,T,N) -> mean [M,B,T,N]
        # einsum indices: m b t k, k b t n -> m b t n
        mean = torch.einsum('mbtk,kbtn->mbtn', zs, mus)
    
        if self.emission_model == "poisson":
            lambdas = F.softplus(mean)
            log_lik = ys * torch.log(lambdas + 1e-8) - lambdas
        else:
            variance = torch.exp(self.log_var)  # shape [N] or [1,N]
            # broadcast variance across M,B,T
            log_lik = -0.5 * (((ys - mean) ** 2) / variance + torch.log(2 * math.pi * variance))
    
        # Sum over observation dims -> [M,B,T]
        log_lik = log_lik.sum(dim=-1)
    
        # Apply mask if provided (mask has shape [M,B,T])
        if mask is not None:
            log_lik = log_lik * mask
    
        # Sum over time -> [M,B]
        return log_lik.sum(dim=-1)


    
    def smooth(self, ys, expected_states):
        # single-trial ys [T,N]
        if ys.dim() == 2:
            T, N = ys.shape
            # expected_states: [m,T,K]
            if expected_states.dim() == 3:
                projected_ys = torch.einsum('tn,dn->td', ys, self.F)  # [T, D]
                mus = self._compute_mus(projected_ys)                  # [K, T, N]
                yhat = torch.einsum('mtk,ktn->mtn', expected_states, mus)  # [m, T, N]
                return yhat
    
            # expected_states: [M,B,T,K] or [M,T,K] (MC dim)
            if expected_states.dim() == 4:
                M, B, T_e, K = expected_states.shape
                assert T_e == T
                # make ys into [M,B,T,N]
                ys_rep = ys.unsqueeze(0).unsqueeze(0).expand(M, B, T, N)
                return self._smooth_batched(ys_rep, expected_states)
    
            if expected_states.dim() == 2:
                # unexpected (e.g., [T,K]) - treat as [1,T,K]
                return self.smooth(ys, expected_states.unsqueeze(0))
    
        # batched ys [B,T,N]
        elif ys.dim() == 3:
            B, T, N = ys.shape
            # expected_states can be [M,B,T,K] or [B,T,K] (no MC)
            if expected_states.dim() == 3:
                # add MC dim
                expected_states = expected_states.unsqueeze(0)  # [1,B,T,K]
    
            # expected_states is [M,B,T,K], ys -> need to expand to [M,B,T,N]
            M = expected_states.shape[0]
            ys_mb = ys.unsqueeze(0).expand(M, -1, -1, -1)  # [M,B,T,N]
            return self._smooth_batched(ys_mb, expected_states)
        else:
            raise ValueError("Unexpected ys shape in smooth()")
    
    
    def _smooth_batched(self, ys_mb, expected_states_mb):
        """
        Helper: ys_mb [M,B,T,N], expected_states_mb [M,B,T,K] -> returns [M,B,T,N]
        """
        M, B, T, N = ys_mb.shape
        # projected: [M,B,T,D]
        projected = torch.einsum('mbtn,dn->mbtd', ys_mb, self.F)
    
        # Compute mus once per batch (they don't depend on MC noise)
        mus = self._compute_mus(projected[0])   # either [K,T,N] or [K,B,T,N] depending on projected[0]
        # if mus is [K,T,N] (happens when projected[0] is 2D) convert to [K,B,T,N]
        if mus.dim() == 3:
            mus = mus[:, None, :, :].expand(-1, B, -1, -1)  # [K,B,T,N]
    
        # combine: expected_states_mb [M,B,T,K], mus [K,B,T,N] -> yhat [M,B,T,N]
        yhat = torch.einsum('mbtk,kbtn->mbtn', expected_states_mb, mus)
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

    
    def log_density(self, zs, ys, temperature, mask=None):
        zs_normalized = zs / (zs.sum(dim=-1, keepdim=True) + 1e-8)
        log_Ps = torch.einsum('mbtn,kn->mbtk', ys, self.W) + self.b  # [M,B,T,K]
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        log_probs = q_dist.log_prob(zs_normalized)  # [M,B,T]

        if mask is not None:
            log_probs = log_probs * mask
        return log_probs.sum(dim=2)  # [M,B]
    
    

    def sample_q_z(self, ys, temperature):   
        M, B, T, N = ys.shape
        log_Ps = torch.einsum('mbtn,kn->mbtk', ys, self.W) + self.b  # [M,B,T,K]
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=log_Ps)
        zs = q_dist.rsample()  # [M,B,T,K]
        return zs
    
    
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

    def log_density(self, zs, ys, temperature, mask=None):
        zs_normalized = zs / (zs.sum(dim=-1, keepdim=True) + 1e-8)
        M,B,T,N = ys.shape
        h, _ = self.rnn(ys.reshape(-1, T, N))      # [M*B,T,H]
        logits = self.out(h).view(M,B,T,-1)        # [M,B,T,K]
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        log_probs = q_dist.log_prob(zs_normalized) # [M,B,T]

        if mask is not None:
            log_probs = log_probs * mask
        return log_probs.sum(dim=2)  # [M,B]

    def sample_q_z(self, ys, temperature):
        M, B, T, N = ys.shape
        ys_flat = ys.reshape(M * B, T, N)  # merge batch for RNN
        h, _ = self.rnn(ys_flat)           # [M*B,T,H]
        logits = self.out(h).reshape(M, B, T, self.K)  # [M,B,T,K]
        q_dist = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        zs = q_dist.rsample()              # [M,B,T,K]
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

def compute_r2_multi(ys_pred, ys_true):
    ys_pred = np.asarray(ys_pred)
    ys_true = np.asarray(ys_true)

    if ys_pred.ndim == 2:  # [T,N], single trial
        ys_pred, ys_true = ys_pred[None, ...], ys_true[None, ...]  # add trial axis

    B, T, N = ys_true.shape
    r2_scores = []

    for b in range(B):
        ss_total = np.sum((ys_true[b] - ys_true[b].mean(axis=0, keepdims=True))**2, axis=0)
        ss_residual = np.sum((ys_true[b] - ys_pred[b])**2, axis=0)
        r2_per_unit = 1 - (ss_residual / (ss_total + 1e-8))
        r2_scores.append(np.mean(r2_per_unit))  # average across units for trial b

    return float(np.mean(r2_scores))  # average across trials



def train_metrics(ys, pred_ys0, y_preds=None, k_max=0):
    if torch.is_tensor(ys): ys = ys.detach().cpu().numpy()
    if torch.is_tensor(pred_ys0): pred_ys0 = pred_ys0.detach().cpu().numpy()
    if y_preds is not None and torch.is_tensor(y_preds): 
        y_preds = y_preds.detach().cpu().numpy()

    r2_scores = []
    for k in range(k_max + 1):
        if k == 0:
            r2 = compute_r2_multi(pred_ys0, ys)
        else:
            # careful: for k>0, ys and preds are shifted in time
            r2 = compute_r2_multi(y_preds[:, k:, k - 1], ys[:, k:])
        r2_scores.append(r2)

    return r2_scores



def predict_k_step_more(k_max, model, variational_z, ys, 
                        n_trajectory=20, n_samples=100, temperature=0.99):
    device = next(model.parameters()).device
    ys_tensor = torch.as_tensor(ys, dtype=torch.float32, device=device)  # [B,T,N]
    B, T, N = ys_tensor.shape

    # Initialize prediction storage
    y_preds = torch.zeros((B, T, k_max, N), device=device)

    # Expand inputs for trajectories
    y_input = ys_tensor.unsqueeze(0).expand(n_trajectory, -1, -1, -1).clone()  # [M,B,T,N]

    # Sample initial states
    z_preds = variational_z.sample_q_z(y_input, temperature)  # [M,B,T,K]

    for k in range(1, k_max + 1):  # start from 1-step forward
        # Project into latent space
        projected = torch.einsum('mbtn,dn->mbtd', y_input, model.F)  # [M,B,T,D]

        # Transition logits
        trans = torch.einsum('mbtd,kd->mbtk', projected[:, :, :-1], model.Rs) + model.r
        sticky = z_preds[:, :, :-1]  # [M,B,T-1,K]
        logits_z = (1 - model.gamma) * trans + model.gamma * sticky

        # Sample future z
        logits_exp = logits_z.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)  # [n_samples,M,B,T-1,K]
        z_full = dist.RelaxedOneHotCategorical(temperature, logits=logits_exp).sample()
        z_t = z_full.mean(dim=0)  # [M,B,T-1,K]
        z_preds[:, :, 1:] = z_t

        # Predict next y
        y_next = model.smooth(y_input[0], z_preds)  # -> [M,B,T,N]
        y_preds[:, k:, k-1] = y_next[:, :, k:].mean(dim=0)  # fill k-step prediction

        # Update inputs with predicted ys
        y_input[:, :, k:] = y_next[:, :, k:]

    return y_preds.detach().cpu().numpy(), z_preds.detach().cpu().numpy()


