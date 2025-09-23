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
    

    
