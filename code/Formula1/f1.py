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

### Generate full laps
def generate_full_laps(laps_winner, winner):
    pos_parts = []
    for lap_idx, lap in laps_winner.iterlaps():
        try:
            pos = lap.get_pos_data()   
        except Exception as e:
            # missing / unavailable pos data for this lap
            print(f"Skipping lap {lap_idx}: {e}")
            continue
        if pos is None or pos.empty:
            # no pos data for this lap
            continue
        # keep only X/Y and add lap metadata
        keep_cols = [c for c in ['X', 'Y', 'SessionTime'] if c in pos.columns]
        pos_small = pos[keep_cols].copy()
        # store lap number and driver so you can identify segments later
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
    # iterate laps in ascending order (chronological)
    for lap in unique_laps:
        idx = np.where(lap_numbers == lap)[0]
        if idx.size == 0:
            continue
        lap_track = track[idx]               # local coords for this lap, length L
        L = len(lap_track)

        # find the nearest index (within this lap) for every corner
        corner_idxs_local = np.zeros(M, dtype=int)
        for m, (cx, cy) in enumerate(corner_xy):
            d = np.linalg.norm(lap_track - np.array([cx, cy]), axis=1)
            corner_idxs_local[m] = int(np.argmin(d))

        # sort by appearance along the lap
        order = np.argsort(corner_idxs_local)          # original corner indices in lap order
        s = corner_idxs_local[order]                   # sorted local indices (increasing)

        # assign segments between successive corners (with wrap-around)
        for i in range(M):
            start_local = int(s[i])
            end_local = int(s[(i + 1) % M])   # next corner (wraps to first)
            corner_id = int(order[i])         # which original corner this segment belongs to

            if end_local > start_local:
                # simple case: segment entirely inside lap
                gstart = idx[start_local]
                gend   = idx[end_local]
                states[gstart:gend, corner_id] = 1
                z[gstart:gend] = corner_id
            else:
                # wrap-around: start_local..end_of_lap  AND  start_of_lap..end_local
                gstart1 = idx[start_local]
                gend1   = idx[-1]     # inclusive - last index of lap
                states[gstart1:gend1 + 1, corner_id] = 1
                z[gstart1:gend1 + 1] = corner_id

                gstart2 = idx[0]
                gend2   = idx[end_local]
                if gend2 > gstart2:
                    states[gstart2:gend2, corner_id] = 1
                    z[gstart2:gend2] = corner_id
                # if gend2 == gstart2 there's no slice to assign at the start
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
    # Add labels and legend
    plt.title(title)
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
    

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)    
