import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

## Gumbel-Softmax sample
def sample_relaxed_one_hot(logits, temperature):
    gumbel = -np.log(-np.log(npr.rand(*logits.shape) + 1e-8) + 1e-8)
    y = (logits + gumbel) / temperature
    return np.exp(y - logsumexp(y))  # softmax

def generate_nascar_soft(T, K, D, N=10, temp=0.99, scale=1, speed=1, seed=0, fancy=True, gamma=0.0, 
                         emission=None, setup=False, model=None):
    npr.seed(seed)
    if fancy:
        As = [np.array([[0, np.pi/24], [-np.pi/24, 0]]),
          np.array([[0, np.pi/24], [-np.pi/24, 0]])
         ]
        As_exp = np.array([expm(A) for A in As])  # shape: [K, D, D]
        centers = [np.array([+2.0, 0.]), np.array([-2.0, 0.])]
        bs = [(np.eye(D) - A) @ center for A, center in zip(As_exp, centers)]  ## avoid drift    
        As.append(np.array([[0, 0], [0, 0]]))
        As.append(np.array([[0, 0], [0, 0]]))
        bs.append(np.array([+0.1, 0.]))  ## small speed
        bs.append(np.array([-0.25, 0.]))
    else:
        As = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]])
        bs = [np.array([0, -0.05]), np.array([0, 0.05])]
        bs += [np.array([+0.05, 0.]), np.array([-0.05, 0.])]
    As_exp = np.array([expm(A) for A in As])
    
    #  Transition weights (Rs @ x + r)
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])   # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])   # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([-1.0])    # y > 0
    w4, b4 = np.array([0.0, -1.0]), np.array([-1.0])    # y < 0
    Rs = scale*np.row_stack((100*w1, 100*w2, 100*w3,100*w4))
    r = scale*np.concatenate((100*b1, 100*b2, 100*b3, 100*b4))
    

    # Initialization
    xs = np.zeros((T+1, D))
    zs = np.zeros((T+1, K))
    zs[0] = np.array([0, 0, 1, 0])  ## first state
    xs[0] = np.array([0, 1])  ## initial point
    sp = npr.rand()*(1-speed)+speed  ## Changing speeds
    sps = []
    for t in range(1, T+1):
        last_state = np.argmax(zs[t-1])
        x_prev = xs[t - 1]
        z_prev = zs[t - 1]
        
        ## add stickiness
        trans = Rs @ x_prev + r 
        sticky = z_prev  ## no stickiness gamma=0
        
        # Sample relaxed z ~ RelaxedOneHotCategorical(Rs @ x + r)
        logits = (1 - gamma) * trans + gamma * sticky
        z_t = sample_relaxed_one_hot(logits, temperature=temp)  # shape: [K]
        zs[t] = z_t
        # Mixture dynamics weighted sum over modes
        A_t = np.tensordot(z_t, As_exp, axes=(0, 0))  # shape: [D, D]
        b_t = np.tensordot(z_t, bs, axes=(0, 0))      # shape: [D]
        # Propagate x
        xs[t] = A_t @ x_prev + sp*b_t 
        sps.append(sp)
        if last_state != np.argmax(zs[t]):
            # sample random speed
            sp = npr.rand()*(1-speed)+speed
    
    # Emissions
    zs, xs = zs[1:], xs[1:]      
    if emission is None:
        C = npr.randn(N, D)  ## emission matrix
        ys = (C @ xs.T).T + npr.randn(xs.shape[0], N)*0.01
    else:
        C = emission
        ys = (C @ xs.T).T + npr.randn(xs.shape[0], N)*0.01  
        
    if setup:  ## corr. dynamics
        model.mu_init = nn.Parameter(torch.tensor(np.tile(np.array([0, 1]), (4, 1))).to('cuda').float())
        model.As = nn.Parameter(torch.tensor(As_exp).to('cuda').float())
        model.bs = nn.Parameter(torch.tensor(np.array(bs)).to('cuda').float())
        model.Rs = nn.Parameter(torch.tensor(Rs).to('cuda').float())
        model.r = nn.Parameter(torch.tensor(r).to('cuda').float())
        model.gamma = nn.Parameter(torch.tensor(gamma, device='cuda'))
        model.logits_z1 = nn.Parameter(torch.tensor(np.array([0, 0, 10, 0])).to('cuda').float())
        model.C = nn.Parameter(torch.tensor(C).to('cuda').float())
        if model.emission_model == "gaussian":
            model.log_var = nn.Parameter(-2.0 + torch.zeros(1, N, device='cuda'))
        return zs, xs, ys, model
    return zs, xs, ys, C
    



### Plotting functions
def plot_gs_states(zs):  ## Soft states per row
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

    
def plot_soft_states(zs, title="", separate_cbar=False):  ## Soft states mixed
    T, K = zs.shape    
    cmap_obj = plt.get_cmap("Accent", K)
    base_colors = np.array([cmap_obj(i)[:3] for i in range(K)])  # RGB only
    # Blended colors per timepoint
    blended_colors = zs @ base_colors  
    alphas = zs.max(axis=1)            # confidence max prop
    # Build RGBA matrix 
    rgba = np.zeros((1, T, 4))
    rgba[0, :, :3] = blended_colors
    rgba[0, :, 3] = alphas
    
    if separate_cbar:
        fig, (ax, cax) = plt.subplots(
            1, 2, figsize=(14, 1.5),
            gridspec_kw={"width_ratios": [12, 0.1]}
        )
    else:
        # single plot without colorbar
        fig, ax = plt.subplots(figsize=(14, 1.5))
        cax = None
    ax.imshow(rgba, aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Time point")
    ax.set_title(title)
    
    if separate_cbar:
        cmap = ListedColormap(base_colors)
        bounds = np.arange(-0.5, K, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])      
        cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(K), orientation="vertical")
        cbar.set_label("Base States")
        cbar.set_ticklabels([f"State {i}" for i in range(K)])
    
    plt.tight_layout()
    plt.show()    
    
    
def plot_gs_states_stacked(zs, with_colorbar=False):  ## Soft states stacked
    T, K = zs.shape
    cmap_obj = plt.get_cmap("Accent", K)
    base_colors = np.array([cmap_obj(i)[:3] for i in range(K)])

    fig, ax = plt.subplots(figsize=(14, 4))

    for k in range(K):
        ax.plot(zs[:, k], color=base_colors[k], lw=1.5)
    ax.set_xlabel("Time step", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_title("", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    if with_colorbar:
        cmap = ListedColormap(base_colors)
        bounds = np.arange(-0.5, K, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, ticks=np.arange(K), orientation="vertical", pad=0.02)
        cbar.set_label("States", fontsize=14)
        cbar.set_ticklabels([f"State {i+1}" for i in range(K)])
        cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
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
    plt.title(title)
    plt.tick_params(axis='both', labelsize=14)
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
    