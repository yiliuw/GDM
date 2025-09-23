import json
import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, BoundaryNorm


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

def plot_expert(states):
    n_states = len(np.unique(states)) 
    state_matrix = states[np.newaxis, :]
    cmap = plt.get_cmap('Set2', n_states)  # discrete colormap
    colors = [cmap(i) for i in range(n_states)]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, n_states, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(12, 2))
    plt.imshow(state_matrix, aspect='auto', cmap=cmap, norm=norm)
    plt.yticks([])
    plt.xlabel("Time point")
    plt.title("Expert Mouse State Timeline")
    cbar = plt.colorbar(ticks=np.arange(n_states))
    cbar.set_label("State")
    cbar.set_ticklabels(np.arange(n_states))
    plt.show()