import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def get_feature_importance(model, target_layer, inputs, target_index):

    model.eval()
    model.zero_grad()
    
    cache = {}
    
    def forward_hook(module, input, output):
        output.retain_grad()
        cache['activations'] = output

    hook = target_layer.register_forward_hook(forward_hook)
        
    out = model(inputs)
    
    out[0, target_index].backward()

    activations = cache['activations']
    grads = activations.grad
    
    weights = torch.mean(grads, dim=2, keepdim=True) 
    cam = torch.sum(weights * activations, dim=1) 
    cam = torch.relu(cam)
    
    cam = cam.unsqueeze(1) 
    size = inputs.shape[-1] 
    cam_resized = F.interpolate(cam, size=size, mode='linear')
    
    cam_resized = cam_resized.squeeze().detach().cpu().numpy() 

    cam_normalized = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8)
    
    hook.remove()

    return cam_normalized


def plot_feature_overlay(model, target_layer, inputs, target_index):

    cam_normalized = get_feature_importance(model, target_layer, inputs, target_index)
    
    original_signal = inputs[0, 0].detach().cpu().numpy()
    seq_length = inputs.shape[-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(original_signal, color='black', linewidth=1.5, zorder=2)
    ax.set_xlim(0, seq_length)
    ymin, ymax = ax.get_ylim()

    heatmap_data = cam_normalized.reshape(1, -1) 
    im = ax.imshow(heatmap_data, aspect='auto', cmap='jet', 
                   extent=[0, seq_length, ymin, ymax], alpha=0.5, zorder=1)


    layer_name = target_layer.__class__.__name__
    ax.set_title(f"1D Signal Grad-CAM Overlay (Layer: {layer_name})")
    ax.set_xlabel("Sequence Steps")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Feature Importance")

    plt.tight_layout()
    plt.show()
