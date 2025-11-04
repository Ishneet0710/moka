"""
Trajectory heatmap generation from keypoints using Gaussian distributions.
Converts discrete waypoints into continuous probabilistic trajectory representations.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io


def create_gaussian_heatmap(center: Tuple[float, float], sigma: float, heatmap_size: Tuple[int, int]) -> np.ndarray:
    """
    Create a 2D Gaussian heatmap centered at a point.
    
    Args:
        center: (x, y) coordinates of the Gaussian center
        sigma: Standard deviation of the Gaussian (controls spread)
        heatmap_size: (height, width) of the output heatmap
    
    Returns:
        gaussian: 2D numpy array with Gaussian distribution
    """
    x = np.arange(0, heatmap_size[1], 1)
    y = np.arange(0, heatmap_size[0], 1)
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
    return gaussian


def waypoints_to_trajectory_heatmap(waypoints: np.ndarray, image_shape: Tuple[int, int], sigma: float = 10) -> np.ndarray:
    """
    Convert discrete waypoints to continuous trajectory heatmap with Gaussian blobs.
    
    Args:
        waypoints: Array of shape (N, 2) containing (x, y) coordinates
        image_shape: (height, width) of output heatmap
        sigma: Gaussian spread parameter (larger = wider blobs)
    
    Returns:
        trajectory_heatmap: (H, W) heatmap with Gaussian blobs at each waypoint
    """
    heatmap = np.zeros(image_shape)
    
    for waypoint in waypoints:
        gaussian_blob = create_gaussian_heatmap(waypoint, sigma, image_shape)
        heatmap = np.maximum(heatmap, gaussian_blob)  # Take max to overlay
    
    return heatmap


def connect_waypoints_with_gaussians(
    waypoints: np.ndarray, 
    image_shape: Tuple[int, int], 
    sigma: float = 10, 
    num_interpolate: int = 10
) -> np.ndarray:
    """
    Connect waypoints with interpolated Gaussian distributions to create smooth trajectory.
    
    Args:
        waypoints: Array of shape (N, 2) containing (x, y) coordinates defining the trajectory
        image_shape: (height, width) of output heatmap
        sigma: Gaussian spread parameter
        num_interpolate: Number of points to interpolate between consecutive waypoints
    
    Returns:
        trajectory_heatmap: Smooth continuous trajectory as a heatmap
    """
    heatmap = np.zeros(image_shape)
    
    if len(waypoints) < 2:
        # If only one waypoint, just create a single Gaussian
        if len(waypoints) == 1:
            gaussian_blob = create_gaussian_heatmap(waypoints[0], sigma, image_shape)
            heatmap = gaussian_blob
        return heatmap
    
    # Interpolate between consecutive waypoints
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        # Linear interpolation
        t = np.linspace(0, 1, num_interpolate)
        interpolated_points = [
            (start[0] + t_i * (end[0] - start[0]),
             start[1] + t_i * (end[1] - start[1])) 
            for t_i in t
        ]
        
        # Add Gaussian at each interpolated point
        for point in interpolated_points:
            gaussian_blob = create_gaussian_heatmap(point, sigma, image_shape)
            heatmap += gaussian_blob
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def get_trajectory_heatmap_from_keypoints(
    object_vertices: Dict[str, np.ndarray], 
    image_shape: Tuple[int, int], 
    sigma: float = 15,
    num_interpolate: int = 20
) -> Dict[str, np.ndarray]:
    """
    Generate trajectory heatmaps from MOKA keypoints for all objects.
    
    Args:
        object_vertices: Dict from get_keypoints_from_segmentation() mapping object names to keypoints
        image_shape: (height, width) of the image
        sigma: Gaussian spread parameter (larger = wider trajectory)
        num_interpolate: Number of interpolation points between waypoints
    
    Returns:
        trajectory_heatmaps: Dict mapping object names to trajectory heatmaps
    """
    trajectory_heatmaps = {}
    
    for object_name, keypoints in object_vertices.items():
        # keypoints are already sorted by y-coordinate in get_keypoints_from_segmentation
        heatmap = connect_waypoints_with_gaussians(
            keypoints, 
            image_shape, 
            sigma=sigma,
            num_interpolate=num_interpolate
        )
        trajectory_heatmaps[object_name] = heatmap
    
    return trajectory_heatmaps


def plot_trajectory_heatmap(
    image: Image.Image, 
    trajectory_heatmaps: Dict[str, np.ndarray], 
    alpha: float = 0.6,
    colormap: str = 'hot',
    return_PIL: bool = True,
    fname: str = None
) -> Image.Image:
    """
    Overlay trajectory heatmaps on image with visualization.
    
    Args:
        image: PIL Image to overlay heatmaps on
        trajectory_heatmaps: Dict mapping object names to heatmaps
        alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
        colormap: Matplotlib colormap name ('hot', 'jet', 'viridis', etc.)
        return_PIL: If True, return PIL Image; if False, return matplotlib figure
        fname: Optional filename to save the figure
    
    Returns:
        PIL Image or matplotlib figure with trajectory heatmaps overlaid
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Create custom colormap with transparency
    if colormap == 'hot':
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.3), (1, 0.5, 0, 0.6), (1, 1, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('trajectory', colors)
    else:
        cmap = plt.get_cmap(colormap)
    
    # Overlay each object's trajectory
    combined_heatmap = np.zeros(trajectory_heatmaps[list(trajectory_heatmaps.keys())[0]].shape)
    for obj_name, heatmap in trajectory_heatmaps.items():
        combined_heatmap = np.maximum(combined_heatmap, heatmap)
    
    plt.imshow(combined_heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
    
    if return_PIL:
        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        return result_image
    else:
        return plt.gcf()


def plot_trajectory_heatmap_per_object(
    image: Image.Image,
    trajectory_heatmaps: Dict[str, np.ndarray],
    object_vertices: Dict[str, np.ndarray],
    alpha: float = 0.5,
    return_PIL: bool = True,
    fname: str = None
) -> Image.Image:
    """
    Plot trajectory heatmaps with different colors per object and show waypoints.
    
    Args:
        image: PIL Image
        trajectory_heatmaps: Dict mapping object names to heatmaps
        object_vertices: Dict mapping object names to keypoints
        alpha: Transparency of heatmap overlay
        return_PIL: If True, return PIL Image
        fname: Optional filename to save
    
    Returns:
        PIL Image with colored trajectory heatmaps and waypoints
    """
    plt.figure(figsize=(14, 10))
    plt.imshow(image)
    
    # Color palette for different objects
    color_list = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'YlOrBr']
    
    # Plot each object's trajectory with different color
    for idx, (obj_name, heatmap) in enumerate(trajectory_heatmaps.items()):
        cmap = plt.get_cmap(color_list[idx % len(color_list)])
        plt.imshow(heatmap, cmap=cmap, alpha=alpha)
        
        # Plot waypoints
        if obj_name in object_vertices:
            waypoints = object_vertices[obj_name]
            plt.plot(waypoints[:, 0], waypoints[:, 1], 'o-', 
                    color=cmap(0.8), markersize=8, linewidth=2, 
                    label=obj_name, alpha=0.8)
            
            # Annotate waypoints
            for i, wp in enumerate(waypoints):
                plt.annotate(f'{i}', (wp[0], wp[1]), 
                           fontsize=10, color='white',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=cmap(0.8), alpha=0.7))
    
    plt.legend(loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
    
    if return_PIL:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        return result_image
    else:
        return plt.gcf()


def extract_attention_heatmap(
    model,
    image_tensor: torch.Tensor,
    text_prompt: str,
    layer_idx: int = -1
) -> np.ndarray:
    """
    Extract attention weights from GroundingDINO transformer as heatmap.
    
    Args:
        model: GroundingDINO model
        image_tensor: Input image tensor
        text_prompt: Text query
        layer_idx: Which transformer layer to extract attention from (-1 = last layer)
    
    Returns:
        attention_heatmap: Spatial attention map
    """
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Capture attention weights from transformer
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights.append(output[1])
    
    # Register hook on transformer decoder layer
    try:
        hook = model.transformer.decoder.layers[layer_idx].register_forward_hook(hook_fn)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor, text_prompt)
        
        hook.remove()
        
        if attention_weights:
            # Average attention across heads and convert to heatmap
            attn = attention_weights[0]
            if isinstance(attn, torch.Tensor):
                attn_map = attn.mean(dim=1).squeeze().cpu().numpy()
                return attn_map
    except Exception as e:
        print(f"Warning: Could not extract attention weights: {e}")
    
    return None
