"""
Compositional 2D Heatmap Generation
Combines VoxPoser's compositional approach with MOKA's visual prompting.

Generates three types of heatmaps via VLM:
1. Affordance map - Where the robot should go
2. Collision map - What to avoid
3. Transition map - Preferred motion directions/pathways

Composes them using weighted linear combination (VoxPoser-style).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import Dict, List, Tuple, Optional
import json


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize heatmap to [0, 1] range without producing NaN.

    Args:
        heatmap: 2D array to normalize

    Returns:
        Normalized heatmap in [0, 1] range
    """
    denom = heatmap.max() - heatmap.min()
    if denom == 0:
        return heatmap
    return (heatmap - heatmap.min()) / denom


def tiles_to_heatmap(tiles: List[str], image_shape: Tuple[int, int],
                     grid_size: Tuple[int, int] = (5, 5),
                     sigma: float = 40.0) -> np.ndarray:
    """
    Convert tile labels (e.g., ['a3', 'b2', 'c4']) to 2D heatmap.

    Args:
        tiles: List of tile labels (e.g., 'a3' means column a, row 3)
        image_shape: (height, width) of output heatmap
        grid_size: (num_rows, num_cols) of the tile grid
        sigma: Gaussian blur sigma for smoothing (default 40.0 for smooth merging)

    Returns:
        2D heatmap with high values at specified tiles
    """
    from string import ascii_lowercase

    heatmap = np.zeros(image_shape)
    h, w = image_shape

    for tile in tiles:
        if len(tile) < 2:
            continue

        # Parse tile: 'a3' -> col='a', row='3'
        col = ascii_lowercase.index(tile[0])  # 0-4
        row = grid_size[1] - int(tile[1])      # Convert to 0-4 (top to bottom)

        # Convert tile indices to pixel coordinates (center of tile)
        y_center = int((row + 0.5) * h / grid_size[1])
        x_center = int((col + 0.5) * w / grid_size[0])

        # Set high value at tile center
        if 0 <= y_center < h and 0 <= x_center < w:
            heatmap[y_center, x_center] = 1.0

    # Apply Gaussian blur to spread influence
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return normalize_map(heatmap)


def points_to_heatmap(points: List[Tuple[float, float]], image_shape: Tuple[int, int],
                      sigma: float = 40.0) -> np.ndarray:
    """
    Convert list of (x, y) points to 2D heatmap with Gaussian blobs.

    Args:
        points: List of (x, y) coordinates
        image_shape: (height, width) of output heatmap
        sigma: Gaussian blob size (default 40.0 for smooth merging)

    Returns:
        2D heatmap with Gaussian blobs at specified points
    """
    heatmap = np.zeros(image_shape)
    h, w = image_shape

    for x, y in points:
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]

        # Gaussian blob centered at (x, y)
        gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
        heatmap += gaussian

    return normalize_map(heatmap)


def generate_affordance_map(response: Dict, image_shape: Tuple[int, int],
                           grid_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Generate affordance map from VLM response.
    Applies distance transform to create gradient flowing toward targets.

    Args:
        response: VLM response with 'affordance_tiles' or 'affordance_points'
        image_shape: (height, width) of output map
        grid_size: Tile grid dimensions

    Returns:
        Affordance heatmap with high values at goal regions
    """
    heatmap = np.zeros(image_shape)

    # Generate from tiles
    if 'affordance_tiles' in response and response['affordance_tiles']:
        heatmap = tiles_to_heatmap(response['affordance_tiles'], image_shape,
                                   grid_size, sigma=50.0)

    # Generate from points
    elif 'affordance_points' in response and response['affordance_points']:
        heatmap = points_to_heatmap(response['affordance_points'], image_shape,
                                    sigma=50.0)

    # Apply distance transform (VoxPoser technique)
    # Creates gradient flowing toward high-value regions
    if heatmap.max() > 0:
        # Invert: high values become low distances
        inverted = 1 - heatmap
        distance_map = distance_transform_edt(inverted)
        # Re-normalize: high distances become low values (attract toward targets)
        affordance_map = normalize_map(distance_map)
        # Invert again so targets have HIGH value
        affordance_map = 1 - affordance_map
    else:
        affordance_map = heatmap

    return affordance_map


def generate_collision_map(response: Dict, image_shape: Tuple[int, int],
                          grid_size: Tuple[int, int] = (5, 5),
                          sigma: float = 40.0) -> np.ndarray:
    """
    Generate collision/avoidance map from VLM response.
    High values indicate regions to AVOID.

    Args:
        response: VLM response with 'collision_tiles' or 'collision_points'
        image_shape: (height, width) of output map
        grid_size: Tile grid dimensions
        sigma: Gaussian blur for safety margins (default 40.0)

    Returns:
        Collision heatmap with high values at obstacles
    """
    heatmap = np.zeros(image_shape)

    # Generate from tiles
    if 'collision_tiles' in response and response['collision_tiles']:
        heatmap = tiles_to_heatmap(response['collision_tiles'], image_shape,
                                   grid_size, sigma=0)  # No blur yet

    # Generate from points
    elif 'collision_points' in response and response['collision_points']:
        heatmap = points_to_heatmap(response['collision_points'], image_shape,
                                    sigma=sigma)

    # Apply Gaussian smoothing for safety margins (VoxPoser technique)
    if sigma > 0 and heatmap.max() > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return normalize_map(heatmap)


def generate_transition_map(response: Dict, image_shape: Tuple[int, int],
                           grid_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Generate transition/pathway map from VLM response.
    High values indicate preferred motion pathways.

    Args:
        response: VLM response with 'transition_tiles' or 'transition_points'
        image_shape: (height, width) of output map
        grid_size: Tile grid dimensions

    Returns:
        Transition heatmap with high values at preferred pathways
    """
    heatmap = np.zeros(image_shape)

    # Generate from tiles
    if 'transition_tiles' in response and response['transition_tiles']:
        heatmap = tiles_to_heatmap(response['transition_tiles'], image_shape,
                                   grid_size, sigma=40.0)

    # Generate from points (e.g., via waypoints)
    elif 'transition_points' in response and response['transition_points']:
        heatmap = points_to_heatmap(response['transition_points'], image_shape,
                                    sigma=40.0)

    return normalize_map(heatmap)


def compose_heatmaps(affordance_map: np.ndarray,
                    collision_map: np.ndarray,
                    transition_map: Optional[np.ndarray] = None,
                    affordance_weight: float = 2.0,
                    collision_weight: float = 1.0,
                    transition_weight: float = 1.0,
                    smooth_sigma: float = 0.0) -> np.ndarray:
    """
    Compose multiple heatmaps using weighted linear combination (VoxPoser-style).

    Args:
        affordance_map: Where to go (high = good)
        collision_map: What to avoid (high = bad)
        transition_map: Preferred pathways (high = good), optional
        affordance_weight: Weight for affordance (default 2.0, VoxPoser uses 2:1 ratio)
        collision_weight: Weight for collision avoidance (default 1.0)
        transition_weight: Weight for transition preference (default 1.0)
        smooth_sigma: Additional Gaussian smoothing after composition (default 0.0 = no smoothing).
                     Increase to 30-50 for smoother, more continuous heatmaps like VoxPoser.

    Returns:
        Composed heatmap with high values indicating preferred regions
    """
    # Compose: attract to affordance, repel from collisions
    # Note: collision_map has high values at obstacles, so we SUBTRACT or use (1 - collision)
    composed = (affordance_map * affordance_weight +
                (1 - collision_map) * collision_weight)

    # Add transition preferences if provided
    if transition_map is not None:
        composed += transition_map * transition_weight

    # Apply additional smoothing to merge high-value regions (VoxPoser-style)
    if smooth_sigma > 0:
        composed = gaussian_filter(composed, sigma=smooth_sigma)

    # Final normalization
    composed = normalize_map(composed)

    return composed


def visualize_composition(image: Image.Image,
                         affordance_map: np.ndarray,
                         collision_map: np.ndarray,
                         transition_map: Optional[np.ndarray],
                         composed_map: np.ndarray,
                         title: str = "Compositional Heatmap") -> plt.Figure:
    """
    Visualize all component maps and final composition side-by-side.

    Args:
        image: Original observation image
        affordance_map: Generated affordance heatmap
        collision_map: Generated collision heatmap
        transition_map: Generated transition heatmap (optional)
        composed_map: Final composed heatmap
        title: Figure title

    Returns:
        Matplotlib figure
    """
    num_maps = 4 if transition_map is not None else 3
    fig, axes = plt.subplots(2, num_maps, figsize=(5*num_maps, 10))

    # Row 1: Individual component maps
    # Affordance
    axes[0, 0].imshow(image)
    axes[0, 0].imshow(affordance_map, cmap='Greens', alpha=0.6)
    axes[0, 0].set_title('Affordance Map\n(Where to Go)', fontsize=12, weight='bold')
    axes[0, 0].axis('off')

    # Collision
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(collision_map, cmap='Reds', alpha=0.6)
    axes[0, 1].set_title('Collision Map\n(What to Avoid)', fontsize=12, weight='bold')
    axes[0, 1].axis('off')

    # Transition (if exists)
    if transition_map is not None:
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(transition_map, cmap='Blues', alpha=0.6)
        axes[0, 2].set_title('Transition Map\n(Preferred Path)', fontsize=12, weight='bold')
        axes[0, 2].axis('off')

        # Composed
        axes[0, 3].imshow(image)
        axes[0, 3].imshow(composed_map, cmap='hot', alpha=0.7)
        axes[0, 3].set_title('Composed Heatmap\n(Final)', fontsize=12, weight='bold')
        axes[0, 3].axis('off')
    else:
        # Composed (no transition)
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(composed_map, cmap='hot', alpha=0.7)
        axes[0, 2].set_title('Composed Heatmap\n(Final)', fontsize=12, weight='bold')
        axes[0, 2].axis('off')

    # Row 2: Heatmaps only (no background)
    axes[1, 0].imshow(affordance_map, cmap='Greens')
    axes[1, 0].set_title('Affordance Only', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(collision_map, cmap='Reds')
    axes[1, 1].set_title('Collision Only', fontsize=10)
    axes[1, 1].axis('off')

    if transition_map is not None:
        axes[1, 2].imshow(transition_map, cmap='Blues')
        axes[1, 2].set_title('Transition Only', fontsize=10)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(composed_map, cmap='hot')
        axes[1, 3].set_title('Composed Only', fontsize=10)
        axes[1, 3].axis('off')
    else:
        axes[1, 2].imshow(composed_map, cmap='hot')
        axes[1, 2].set_title('Composed Only', fontsize=10)
        axes[1, 2].axis('off')

    plt.suptitle(title, fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()

    return fig


def sample_trajectory_from_heatmap(heatmap: np.ndarray,
                                  start_point: Tuple[float, float],
                                  num_steps: int = 50,
                                  step_size: float = 10.0) -> np.ndarray:
    """
    Sample trajectory by following gradient of heatmap (gradient ascent).

    Args:
        heatmap: Composed heatmap (high values = preferred regions)
        start_point: Starting (x, y) position
        num_steps: Number of trajectory steps
        step_size: Size of each step in pixels

    Returns:
        Trajectory as array of shape (num_steps, 2) with (x, y) coordinates
    """
    trajectory = [start_point]
    current = np.array(start_point, dtype=float)
    h, w = heatmap.shape

    for _ in range(num_steps - 1):
        x, y = int(current[0]), int(current[1])

        # Check bounds
        if not (1 <= x < w-1 and 1 <= y < h-1):
            break

        # Compute gradient (direction of steepest ascent)
        grad_x = (heatmap[y, min(x+1, w-1)] - heatmap[y, max(x-1, 0)]) / 2
        grad_y = (heatmap[min(y+1, h-1), x] - heatmap[max(y-1, 0), x]) / 2

        gradient = np.array([grad_x, grad_y])

        # Normalize and scale
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm * step_size
        else:
            # Random walk if gradient is zero
            gradient = np.random.randn(2) * step_size

        # Update position
        current = current + gradient
        current = np.clip(current, [0, 0], [w-1, h-1])

        trajectory.append(tuple(current))

    return np.array(trajectory)
