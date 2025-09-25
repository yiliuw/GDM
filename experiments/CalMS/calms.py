import json
import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

def convert_to_array(dictionary, feature_dictionary = None):
    # Convert dictionary values (lists) to numpy arrays, until depth 3.
    converted = {}
    # First key is the group name for the sequences
    for groupname in dictionary.keys():
        converted[groupname] = {}
        # Next key is the sequence id
        for sequence_id in dictionary[groupname].keys():
            converted[groupname][sequence_id] = {}
            # If not adding features, add keypoints, scores, and annotations & metadata (if available)
            if feature_dictionary is None:
                converted[groupname][sequence_id]['keypoints'] = np.array(dictionary[groupname][sequence_id]['keypoints'])
            converted[groupname][sequence_id]['scores'] = np.array(dictionary[groupname][sequence_id]['scores'])          
            if 'annotations' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['annotations'] = np.array(dictionary[groupname][sequence_id]['annotations'])                         
            if 'metadata' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['metadata'] = dictionary[groupname][sequence_id]['metadata']                  
    return converted


W, H = 1024, 570  # image width, height
def normalize_by_image(kp, W=1024, H=570):
    kp = kp.copy().astype(np.float32)
    kp[:, :, 0, :] /= W   # normalize x
    kp[:, :, 1, :] /= H   # normalize y
    return kp

def fit_scaler(data, mouse_ids, annotator="annotator-id_0", task="task1/train"):
    scaler = StandardScaler()
    trials = []

    # fit scaler on all frames 
    for mid in mouse_ids:
        key = f"{task}/mouse{mid:03d}_{task.split('/')[0]}_annotator1"
        kp = data[annotator][key]['keypoints']
        kp_norm = normalize_by_image(kp, W, H)
        X = kp_norm.reshape(kp_norm.shape[0], -1)  # (n_frames, 28)
        scaler.partial_fit(X)  # update mean/std

    # transform each trial
    for mid in mouse_ids:
        key = f"{task}/mouse{mid:03d}_{task.split('/')[0]}_annotator1"
        kp = data[annotator][key]['keypoints']
        kp_norm = normalize_by_image(kp, W, H)
        X = kp_norm.reshape(kp_norm.shape[0], -1)
        X_scaled = scaler.transform(X)
        trials.append(torch.tensor(X_scaled).to('cuda').float())
    return scaler, trials


def plot_soft_states(zs, title="", separate_cbar=False):
    T, K = zs.shape    
    cmap_obj = plt.get_cmap("Set2", K)
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
            1, 2, figsize=(8, 2),
            gridspec_kw={"width_ratios": [8, 0.08]}
        )
    else:
        # Single plot without colorbar
        fig, ax = plt.subplots(figsize=(12, 2))
        cax = None
    ax.imshow(rgba, aspect="auto")
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Time point", fontsize=12)
    
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
        cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()