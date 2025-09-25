import fastf1 as ff1
from fastf1 import plotting
from fastf1 import utils
import fastf1.legacy
import fastf1 as ff1
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm



### Generate full laps
def generate_full_laps(laps_winner, winner):
    pos_parts = []
    for lap_idx, lap in laps_winner.iterlaps():
        try:
            pos = lap.get_pos_data()   
        except Exception as e:
            # missing pos data for this lap
            print(f"Skipping lap {lap_idx}: {e}")
            continue
        if pos is None or pos.empty:
            # no pos data for this lap
            continue
        # keep only X/Y and add lap metadata
        keep_cols = [c for c in ['X', 'Y', 'SessionTime'] if c in pos.columns]
        pos_small = pos[keep_cols].copy()
        # store lap number
        lap_number = lap['LapNumber'] 
        pos_small['LapNumber'] = lap_number
        pos_small['Driver'] = winner
        pos_parts.append(pos_small)
    # concat into full race trajectory
    pos_winner = pd.concat(pos_parts, ignore_index=True)
    return pos_winner

### Generate states
def one_hot_states_multilap(track, corner_xy, lap_numbers):
    track = np.asarray(track)
    corner_xy = np.asarray(corner_xy)
    lap_numbers = np.asarray(lap_numbers)

    T = track.shape[0]
    M = corner_xy.shape[0]

    states = np.zeros((T, M), dtype=int)
    z = np.zeros(T, dtype=int)

    unique_laps = np.unique(lap_numbers)
    # iterate laps in ascending order
    for lap in unique_laps:
        idx = np.where(lap_numbers == lap)[0]
        if idx.size == 0:
            continue
        lap_track = track[idx] # local coords for this lap
        L = len(lap_track)

        # find the nearest index for every corner
        corner_idxs_local = np.zeros(M, dtype=int)
        for m, (cx, cy) in enumerate(corner_xy):
            d = np.linalg.norm(lap_track - np.array([cx, cy]), axis=1)
            corner_idxs_local[m] = int(np.argmin(d))

        # sort by appearance along the lap
        order = np.argsort(corner_idxs_local)  # original corner indices in lap order
        s = corner_idxs_local[order]          # sorted local indices

        # assign segments between successive corners 
        for i in range(M):
            start_local = int(s[i])
            end_local = int(s[(i + 1) % M])   # next corner (wraps to first)
            corner_id = int(order[i])       

            if end_local > start_local:
                # simple case: segment entirely inside lap
                gstart = idx[start_local]
                gend   = idx[end_local]
                states[gstart:gend, corner_id] = 1
                z[gstart:gend] = corner_id
            else:
                # wrap-around
                gstart1 = idx[start_local]
                gend1   = idx[-1]     # inclusive 
                states[gstart1:gend1 + 1, corner_id] = 1
                z[gstart1:gend1 + 1] = corner_id

                gstart2 = idx[0]
                gend2   = idx[end_local]
                if gend2 > gstart2:
                    states[gstart2:gend2, corner_id] = 1
                    z[gstart2:gend2] = corner_id
    return states, z




## latent trajectory
def plot_trajectory(z, x, ls="-", title=''):
    n_states = z.max() + 1
    cmap_obj = plt.get_cmap("tab20", n_states)   
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    for start, stop in zip(zcps[:-1], zcps[1:]):
        plt.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=cmap_obj.colors[z[start]],
                alpha=1.0)
    plt.title(title)
    plt.show()
    
## Soft latent trajectory    
def plot_soft_trajectory(zs, x, ls="-", title="", separate_cbar=True):
    T, K = zs.shape
    cmap_obj = plt.get_cmap("Accent", K)
    base_colors = np.array([cmap_obj(i)[:3] for i in range(K)])  # RGB  
    # Blended colors and confidence as alpha
    blended_colors = zs @ base_colors  # (T, 3)
    alphas = zs.max(axis=1)            # confidence as max prob
    
    if separate_cbar:
        fig, (ax, cax) = plt.subplots(
            1, 2, figsize=(10, 6),
            gridspec_kw={"width_ratios": [12, 0.5]})
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = None    
    # Plot trajectory 
    for t in range(T - 1):
        ax.plot(
            x[t:t+2, 0], x[t:t+2, 1],
            lw=1.5, ls=ls,
            color=np.append(blended_colors[t], alphas[t]))    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ## Plot colorbar
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
    offset_vector = [500, 0] 
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"
        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi 
        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y
        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = [text_x, text_y]
        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = [corner['X'], corner['Y']]  
        # Draw a circle next to the track
        plt.scatter(text_x, text_y, color='indigo', s=220)  
        # Draw a line from the track to this circle
        plt.plot([track_x, text_x], [track_y, text_y], color='black') 
        plt.text(text_x, text_y, txt,
                 va='center_baseline', ha='center', size=15, color='white')
plt.show()
    
    
def plot_observations(z, y, ls="-", lw=1, embed=True):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    n_states = z.max() + 1
    cmap_obj = plt.get_cmap("tab20", n_states)
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
    
def plot_gs_states(zs):
    T, K = zs.shape
    cmap_obj = plt.get_cmap("Accent", K)
    fig, axes = plt.subplots(K, 1, figsize=(12, 2 * K), sharex=True)
    for k in range(K):
        axes[k].plot(zs[:, k], label=f"State {k}", color=cmap_obj.colors[k])
        axes[k].set_ylabel(f"$z_t^{k}$")
        axes[k].legend(loc="upper right")
        axes[k].grid(True)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)    

def align_trajectory_affine(X_inf, X_true):
    T, D = X_inf.shape
    X_inf_aug = np.hstack([X_inf, np.ones((T, 1))])  # (T, D+1)
    # Solve least squares 
    A, _, _, _ = np.linalg.lstsq(X_inf_aug, X_true, rcond=None)
    X_aligned = X_inf_aug @ A
    return X_aligned