"""
Extract attention maps from GroundingDINO for task-specific area identification.
Combines attention with trajectory heatmaps for focused motion planning.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image
import io
from groundingdino.util.inference import predict
from moka.vision.segmentation import prepare_dino, load_pil_image
from moka.vision.trajectory_heatmap import create_gaussian_heatmap


def extract_grounding_dino_attention(
    model,
    image_tensor: torch.Tensor,
    text_prompt: str,
    layer_idx: int = -1,
    aggregate_method: str = 'mean'
) -> Optional[np.ndarray]:
    """
    Extract attention weights from GroundingDINO transformer.
    
    Args:
        model: GroundingDINO model
        image_tensor: Preprocessed image tensor
        text_prompt: Text query (e.g., "metal watch")
        layer_idx: Which decoder layer (-1 = last layer)
        aggregate_method: How to aggregate attention ('mean', 'max', 'sum')
    
    Returns:
        attention_map: 2D spatial attention map, or None if extraction fails
    """
    attention_weights = []
    
    def attention_hook(module, input, output):
        """Hook to capture attention weights from transformer decoder"""
        # GroundingDINO cross_attn outputs a single tensor
        if isinstance(output, torch.Tensor):
            attention_weights.append(output)
        elif isinstance(output, tuple) and len(output) > 1:
            # Fallback for other architectures
            attn = output[1]
            attention_weights.append(attn)
    
    try:
        # Register hook on decoder layer's cross-attention
        # GroundingDINO has model.transformer.decoder.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
            decoder_layer = model.transformer.decoder.layers[layer_idx]
            # Hook into cross_attn (attention between image and text)
            if hasattr(decoder_layer, 'cross_attn'):
                target_module = decoder_layer.cross_attn
            elif hasattr(decoder_layer, 'self_attn'):
                target_module = decoder_layer.self_attn
            else:
                raise AttributeError("Cannot find attention module in decoder layer")
        elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            decoder_layer = model.model.transformer.decoder.layers[layer_idx]
            target_module = decoder_layer.cross_attn if hasattr(decoder_layer, 'cross_attn') else decoder_layer.self_attn
        else:
            raise AttributeError("Cannot find transformer decoder in model")
        
        hook = target_module.register_forward_hook(attention_hook)
        
        # Run inference
        with torch.no_grad():
            BOX_THRESHOLD = 0.3
            TEXT_THRESHOLD = 0.22
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
        
        hook.remove()
        
        # Process attention weights
        if attention_weights:
            attn = attention_weights[0]  # Get captured attention
            
            if isinstance(attn, torch.Tensor):
                # GroundingDINO cross_attn output shape: [batch, num_queries, features]
                # e.g., [1, 900, 256]
                
                # Aggregate over feature dimension to get query importance
                if aggregate_method == 'mean':
                    attn_map = attn.mean(dim=-1)  # [batch, num_queries]
                elif aggregate_method == 'max':
                    attn_map = attn.max(dim=-1)[0]  # [batch, num_queries]
                elif aggregate_method == 'sum':
                    attn_map = attn.sum(dim=-1)  # [batch, num_queries]
                else:
                    attn_map = attn.mean(dim=-1)
                
                attn_map = attn_map.squeeze().cpu().numpy()  # [num_queries]
                
                # Reshape to 2D spatial map (approximate square grid)
                num_queries = attn_map.shape[0]
                grid_size = int(np.sqrt(num_queries))
                
                # Pad or trim to make it square
                if grid_size * grid_size < num_queries:
                    grid_size += 1
                
                padded = np.zeros(grid_size * grid_size)
                padded[:num_queries] = attn_map
                attn_map_2d = padded.reshape(grid_size, grid_size)
                
                # Normalize to [0, 1]
                if attn_map_2d.max() > 0:
                    attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min())
                
                return attn_map_2d
        
    except Exception as e:
        print(f"Warning: Could not extract attention weights: {e}")
        print(f"This is expected if the model architecture doesn't expose attention.")
    
    return None


def get_attention_heatmap_for_object(
    image: Image.Image,
    object_name: str,
    target_size: Tuple[int, int] = (512, 512)
) -> Optional[np.ndarray]:
    """
    Extract attention heatmap for a specific object using GroundingDINO.
    
    Args:
        image: PIL Image
        object_name: Object to focus on (e.g., "metal watch")
        target_size: (height, width) for output heatmap
    
    Returns:
        attention_heatmap: 2D attention map resized to target_size
    """
    # Load and preprocess image
    image_np, image_tensor = load_pil_image(image)
    
    # Load GroundingDINO model
    model = prepare_dino(dino_path='./ckpts')
    
    # Extract attention
    attention_map = extract_grounding_dino_attention(
        model=model,
        image_tensor=image_tensor,
        text_prompt=object_name,
        layer_idx=-1,
        aggregate_method='mean'
    )
    
    if attention_map is not None:
        # Resize to target size
        from scipy.ndimage import zoom
        h, w = target_size
        zoom_factors = (h / attention_map.shape[0], w / attention_map.shape[1])
        attention_heatmap = zoom(attention_map, zoom_factors, order=1)
        return attention_heatmap
    
    return None


def combine_attention_with_trajectory(
    trajectory_heatmap: np.ndarray,
    attention_heatmap: np.ndarray,
    alpha: float = 0.5,
    method: str = 'multiply'
) -> np.ndarray:
    """
    Combine attention map with trajectory heatmap for focused motion planning.
    
    Args:
        trajectory_heatmap: Trajectory heatmap from motion planning
        attention_heatmap: Attention map from GroundingDINO
        alpha: Weighting factor (0=only trajectory, 1=only attention)
        method: 'multiply', 'add', or 'weighted'
    
    Returns:
        combined_heatmap: Attention-weighted trajectory
    """
    # Ensure same shape
    assert trajectory_heatmap.shape == attention_heatmap.shape, \
        "Trajectory and attention heatmaps must have same shape"
    
    if method == 'multiply':
        # Element-wise multiplication emphasizes overlapping regions
        combined = trajectory_heatmap * attention_heatmap
    
    elif method == 'add':
        # Weighted addition
        combined = (1 - alpha) * trajectory_heatmap + alpha * attention_heatmap
    
    elif method == 'weighted':
        # Weighted geometric mean
        combined = (trajectory_heatmap ** (1 - alpha)) * (attention_heatmap ** alpha)
    
    else:
        combined = trajectory_heatmap * attention_heatmap
    
    # Normalize to [0, 1]
    if combined.max() > 0:
        combined = combined / combined.max()
    
    return combined


def get_smooth_object_heatmap_from_bbox(
    image: Image.Image,
    object_name: str,
    target_size: Tuple[int, int],
    sigma: float = 50.0,
    use_bbox_size: bool = True
) -> Optional[np.ndarray]:
    """
    Create smooth Gaussian heatmap from GroundingDINO bounding box.
    This provides differentiable, smooth attention without blocky artifacts.
    
    Args:
        image: PIL Image
        object_name: Object to detect (e.g., "metal watch")
        target_size: (height, width) for output heatmap
        sigma: Gaussian spread (larger = wider attention region)
        use_bbox_size: If True, scale sigma based on bbox size
    
    Returns:
        smooth_heatmap: Smooth Gaussian heatmap centered on object
    """
    from moka.vision.segmentation import prepare_dino, load_pil_image
    from groundingdino.util.inference import predict
    
    # Load image and model
    image_np, image_tensor = load_pil_image(image)
    model = prepare_dino(dino_path='./ckpts')
    
    # Get bounding box
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.22
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=object_name,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    if len(boxes) == 0:
        print(f"Warning: No object '{object_name}' detected.")
        return None
    
    # Get highest confidence detection
    best_idx = torch.argmax(logits)
    bbox = boxes[best_idx]  # [center_x, center_y, width, height] normalized [0, 1]
    
    # Convert to pixel coordinates
    h, w = target_size
    center_x = float(bbox[0]) * w
    center_y = float(bbox[1]) * h
    bbox_width = float(bbox[2]) * w
    bbox_height = float(bbox[3]) * h
    
    # Optionally scale sigma based on bbox size
    if use_bbox_size:
        # Use average of width/height as spread
        adaptive_sigma = (bbox_width + bbox_height) / 4.0
        sigma = max(sigma, adaptive_sigma)  # Use at least the provided sigma
    
    # Create smooth Gaussian heatmap
    smooth_heatmap = create_gaussian_heatmap(
        center=(center_x, center_y),
        sigma=sigma,
        heatmap_size=target_size
    )
    
    return smooth_heatmap


def get_smooth_object_heatmap_from_mask(
    image: Image.Image,
    object_name: str,
    target_size: Tuple[int, int],
    sigma: float = 20.0
) -> Optional[np.ndarray]:
    """
    Create smooth heatmap from GroundingDINO + SAM segmentation mask.
    Applies Gaussian blur to binary mask for differentiable, smooth gradients.
    
    Args:
        image: PIL Image
        object_name: Object to segment
        target_size: (height, width) for output heatmap
        sigma: Gaussian blur sigma (larger = smoother)
    
    Returns:
        smooth_heatmap: Smooth blurred segmentation mask
    """
    from scipy.ndimage import gaussian_filter, zoom
    from moka.vision.segmentation import (
        prepare_dino, load_pil_image, get_object_bboxes, get_segmentation_masks
    )
    
    # Load image and get segmentation
    image_np, image_tensor = load_pil_image(image)
    model = prepare_dino(dino_path='./ckpts')
    
    # Get bounding box
    boxes, logits, phrases = get_object_bboxes(image_tensor, [object_name])
    
    if len(boxes) == 0:
        print(f"Warning: No object '{object_name}' detected.")
        return None
    
    # Get segmentation mask
    masks = get_segmentation_masks(
        image=image,
        texts=[object_name],
        boxes=boxes,
        logits=logits,
        phrases=phrases,
        visualize=False
    )
    
    if masks is None or len(masks) == 0:
        print(f"Warning: Could not segment '{object_name}'.")
        return None
    
    # Get first mask and convert to binary
    mask = masks[0][0].cpu().numpy() if torch.is_tensor(masks[0][0]) else masks[0][0]
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Resize to target size if needed
    if binary_mask.shape != target_size:
        zoom_factors = (target_size[0] / binary_mask.shape[0], 
                       target_size[1] / binary_mask.shape[1])
        binary_mask = zoom(binary_mask, zoom_factors, order=1)
    
    # Apply Gaussian blur for smooth gradients
    smooth_mask = gaussian_filter(binary_mask, sigma=sigma)
    
    # Normalize to [0, 1]
    if smooth_mask.max() > 0:
        smooth_mask = smooth_mask / smooth_mask.max()
    
    return smooth_mask


def generate_attention_focused_trajectory(
    image: Image.Image,
    trajectory_heatmap: np.ndarray,
    object_name: str,
    alpha: float = 0.5,
    combine_method: str = 'multiply',
    attention_method: str = 'bbox'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate trajectory heatmap focused on task-specific areas using attention.
    
    Args:
        image: PIL Image
        trajectory_heatmap: Base trajectory heatmap
        object_name: Object to focus attention on
        alpha: Attention weighting
        combine_method: How to combine attention and trajectory
        attention_method: 'bbox' (smooth Gaussian from bbox), 
                         'mask' (smooth blur from segmentation),
                         'transformer' (old method, blocky)
    
    Returns:
        focused_trajectory: Attention-weighted trajectory heatmap
        attention_map: Raw attention map (for visualization)
    """
    target_size = trajectory_heatmap.shape
    
    # Get smooth attention map based on method
    if attention_method == 'bbox':
        attention_map = get_smooth_object_heatmap_from_bbox(
            image=image,
            object_name=object_name,
            target_size=target_size,
            sigma=50.0,
            use_bbox_size=True
        )
    elif attention_method == 'mask':
        attention_map = get_smooth_object_heatmap_from_mask(
            image=image,
            object_name=object_name,
            target_size=target_size,
            sigma=20.0
        )
    elif attention_method == 'transformer':
        # Old method - blocky attention from transformer
        attention_map = get_attention_heatmap_for_object(
            image=image,
            object_name=object_name,
            target_size=target_size
        )
    else:
        raise ValueError(f"Unknown attention_method: {attention_method}")
    
    if attention_map is None:
        print(f"Warning: Could not extract attention for '{object_name}'. Using original trajectory.")
        return trajectory_heatmap, None
    
    # Combine attention with trajectory
    focused_trajectory = combine_attention_with_trajectory(
        trajectory_heatmap=trajectory_heatmap,
        attention_heatmap=attention_map,
        alpha=alpha,
        method=combine_method
    )
    
    return focused_trajectory, attention_map


def visualize_attention_and_trajectory(
    image: Image.Image,
    trajectory_heatmap: np.ndarray,
    attention_map: Optional[np.ndarray],
    focused_trajectory: np.ndarray,
    waypoints: Optional[list] = None,
    title: str = "Attention-Focused Trajectory",
    return_PIL: bool = True
) -> Image.Image:
    """
    Visualize attention map, trajectory, and combined result side-by-side.
    
    Args:
        image: Original PIL Image
        trajectory_heatmap: Original trajectory
        attention_map: Attention map (can be None)
        focused_trajectory: Combined attention + trajectory
        waypoints: Optional waypoints to overlay
        title: Figure title
        return_PIL: Return as PIL Image
    
    Returns:
        Visualization image
    """
    if attention_map is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Trajectory heatmap
    axes[1].imshow(image)
    axes[1].imshow(trajectory_heatmap, cmap='hot', alpha=0.6)
    if waypoints is not None and len(waypoints) > 0:
        waypoints_arr = np.array(waypoints)
        axes[1].plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'co-', markersize=8, linewidth=2)
    axes[1].set_title('Trajectory Heatmap', fontsize=12, weight='bold')
    axes[1].axis('off')
    
    # Attention map (if available)
    if attention_map is not None:
        axes[2].imshow(image)
        axes[2].imshow(attention_map, cmap='viridis', alpha=0.6)
        axes[2].set_title('Attention Map', fontsize=12, weight='bold')
        axes[2].axis('off')
        
        # Combined
        axes[3].imshow(image)
        axes[3].imshow(focused_trajectory, cmap='hot', alpha=0.6)
        if waypoints is not None and len(waypoints) > 0:
            waypoints_arr = np.array(waypoints)
            axes[3].plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'co-', markersize=8, linewidth=2)
        axes[3].set_title('Attention-Focused Trajectory', fontsize=12, weight='bold')
        axes[3].axis('off')
    else:
        # Just show focused trajectory
        axes[2].imshow(image)
        axes[2].imshow(focused_trajectory, cmap='hot', alpha=0.6)
        if waypoints is not None and len(waypoints) > 0:
            waypoints_arr = np.array(waypoints)
            axes[2].plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'co-', markersize=8, linewidth=2)
        axes[2].set_title('Focused Trajectory', fontsize=12, weight='bold')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    
    if return_PIL:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        return result_image
    else:
        return fig


def get_task_specific_area_mask(
    image: Image.Image,
    object_name: str,
    threshold: float = 0.5,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Get binary mask of task-specific area using attention.
    
    Args:
        image: PIL Image
        object_name: Object to focus on
        threshold: Attention threshold for binary mask
        target_size: Output size
    
    Returns:
        binary_mask: Binary mask of task-specific area
    """
    attention_map = get_attention_heatmap_for_object(
        image=image,
        object_name=object_name,
        target_size=target_size
    )
    
    if attention_map is not None:
        binary_mask = (attention_map > threshold).astype(np.float32)
        return binary_mask
    
    return np.ones(target_size)  # Return all-ones if attention fails
