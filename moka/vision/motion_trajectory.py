"""
Motion trajectory generation for robot manipulation.
Generates continuous trajectory heatmaps from start position to goal (grasp/contact points).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
from moka.vision.trajectory_heatmap import (
    create_gaussian_heatmap,
    connect_waypoints_with_gaussians
)


def generate_motion_trajectory(
    start_point: Tuple[float, float],
    goal_point: Tuple[float, float],
    image_shape: Tuple[int, int],
    trajectory_type: str = 'direct',
    sigma: float = 15,
    num_interpolate: int = 50,
    approach_angle: Optional[float] = None
) -> np.ndarray:
    """
    Generate a motion trajectory from start to goal position.
    
    Args:
        start_point: (x, y) starting position (e.g., current robot position)
        goal_point: (x, y) goal position (e.g., grasp point)
        image_shape: (height, width) of output heatmap
        trajectory_type: 'direct', 'curved', 'vertical_then_horizontal', 'approach_from_above'
        sigma: Gaussian spread parameter
        num_interpolate: Number of interpolation points
        approach_angle: Optional angle for approach direction (radians)
    
    Returns:
        trajectory_heatmap: Continuous trajectory from start to goal
    """
    waypoints = []
    
    if trajectory_type == 'direct':
        # Straight line from start to goal
        waypoints = [start_point, goal_point]
    
    elif trajectory_type == 'curved':
        # Curved trajectory (parabolic arc)
        t = np.linspace(0, 1, 10)
        # Control point for curve (above midpoint)
        mid_x = (start_point[0] + goal_point[0]) / 2
        mid_y = min(start_point[1], goal_point[1]) - 50  # Arc upward
        
        for t_i in t:
            # Quadratic Bezier curve
            x = (1-t_i)**2 * start_point[0] + 2*(1-t_i)*t_i * mid_x + t_i**2 * goal_point[0]
            y = (1-t_i)**2 * start_point[1] + 2*(1-t_i)*t_i * mid_y + t_i**2 * goal_point[1]
            waypoints.append((x, y))
    
    elif trajectory_type == 'vertical_then_horizontal':
        # Move vertically first, then horizontally
        waypoints = [
            start_point,
            (start_point[0], goal_point[1]),  # Vertical move
            goal_point  # Horizontal move
        ]
    
    elif trajectory_type == 'approach_from_above':
        # Approach from above (common for pick operations)
        approach_height = 80  # pixels above goal
        waypoints = [
            start_point,
            (goal_point[0], goal_point[1] - approach_height),  # Above goal
            goal_point  # Down to goal
        ]
    
    elif trajectory_type == 'approach_with_angle':
        # Approach from a specific angle
        if approach_angle is not None:
            approach_dist = 100  # pixels
            approach_x = goal_point[0] - approach_dist * np.cos(approach_angle)
            approach_y = goal_point[1] - approach_dist * np.sin(approach_angle)
            waypoints = [
                start_point,
                (approach_x, approach_y),
                goal_point
            ]
        else:
            waypoints = [start_point, goal_point]
    
    # Convert waypoints to continuous trajectory
    waypoints = np.array(waypoints)
    trajectory_heatmap = connect_waypoints_with_gaussians(
        waypoints,
        image_shape,
        sigma=sigma,
        num_interpolate=num_interpolate
    )
    
    return trajectory_heatmap


def generate_grasp_trajectory(
    grasp_keypoint: Tuple[float, float],
    image_shape: Tuple[int, int],
    start_position: Optional[Tuple[float, float]] = None,
    motion_direction: str = 'downward',
    sigma: float = 15,
    num_interpolate: int = 50
) -> np.ndarray:
    """
    Generate trajectory for grasping motion based on motion direction.
    
    Args:
        grasp_keypoint: (x, y) grasp point on object
        image_shape: (height, width)
        start_position: Optional starting position, if None uses top/side based on direction
        motion_direction: 'downward', 'upward', 'leftward', 'rightward', 'forward'
        sigma: Gaussian spread
        num_interpolate: Interpolation density
    
    Returns:
        trajectory_heatmap: Grasp trajectory
    """
    h, w = image_shape
    
    # Determine start position based on motion direction if not provided
    if start_position is None:
        if motion_direction == 'downward':
            start_position = (grasp_keypoint[0], max(0, grasp_keypoint[1] - 150))
        elif motion_direction == 'upward':
            start_position = (grasp_keypoint[0], min(h, grasp_keypoint[1] + 150))
        elif motion_direction == 'leftward':
            start_position = (min(w, grasp_keypoint[0] + 150), grasp_keypoint[1])
        elif motion_direction == 'rightward':
            start_position = (max(0, grasp_keypoint[0] - 150), grasp_keypoint[1])
        else:  # forward or default
            start_position = (grasp_keypoint[0], max(0, grasp_keypoint[1] - 150))
    
    # Generate trajectory
    trajectory = generate_motion_trajectory(
        start_position,
        grasp_keypoint,
        image_shape,
        trajectory_type='approach_from_above' if motion_direction == 'downward' else 'direct',
        sigma=sigma,
        num_interpolate=num_interpolate
    )
    
    return trajectory


def generate_place_trajectory(
    start_keypoint: Tuple[float, float],
    target_keypoint: Tuple[float, float],
    image_shape: Tuple[int, int],
    motion_direction: str = 'downward',
    sigma: float = 15,
    num_interpolate: int = 50,
    lift_height: int = 100
) -> np.ndarray:
    """
    Generate trajectory for place motion (pick-and-place).
    
    Args:
        start_keypoint: (x, y) starting grasp point
        target_keypoint: (x, y) target place location
        image_shape: (height, width)
        motion_direction: Direction of final placement
        sigma: Gaussian spread
        num_interpolate: Interpolation density
        lift_height: How high to lift object before moving
    
    Returns:
        trajectory_heatmap: Place trajectory with lift
    """
    # Create waypoints: grasp -> lift -> move -> place
    waypoints = [
        start_keypoint,
        (start_keypoint[0], start_keypoint[1] - lift_height),  # Lift up
        (target_keypoint[0], target_keypoint[1] - lift_height),  # Move over target
        target_keypoint  # Place down
    ]
    
    waypoints = np.array(waypoints)
    trajectory_heatmap = connect_waypoints_with_gaussians(
        waypoints,
        image_shape,
        sigma=sigma,
        num_interpolate=num_interpolate
    )
    
    return trajectory_heatmap


def generate_subtask_trajectory(
    subtask: Dict,
    object_vertices: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    sigma: float = 15,
    num_interpolate: int = 50,
    start_position: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Generate trajectory for a MOKA subtask.
    
    Args:
        subtask: Dict with keys 'object_grasped', 'object_unattached', 'motion_direction'
        object_vertices: Dict mapping object names to keypoints
        image_shape: (height, width)
        sigma: Gaussian spread
        num_interpolate: Interpolation density
        start_position: Optional robot starting position
    
    Returns:
        trajectory_heatmap: Complete trajectory for subtask
        waypoints: List of key waypoints in trajectory
    """
    object_grasped = subtask.get('object_grasped', '')
    object_unattached = subtask.get('object_unattached', '')
    motion_direction = subtask.get('motion_direction', 'downward')
    
    waypoints = []
    
    # Case 1: Grasp and place (both objects specified)
    if object_grasped and object_unattached:
        # Get keypoints
        grasp_kp = object_vertices[object_grasped][0]  # First keypoint (usually center)
        target_kp = object_vertices[object_unattached][0]
        
        # Generate pick-and-place trajectory
        trajectory_heatmap = generate_place_trajectory(
            grasp_kp,
            target_kp,
            image_shape,
            motion_direction=motion_direction,
            sigma=sigma,
            num_interpolate=num_interpolate
        )
        
        waypoints = [
            grasp_kp,
            (grasp_kp[0], grasp_kp[1] - 100),  # Lift
            (target_kp[0], target_kp[1] - 100),  # Move
            target_kp  # Place
        ]
    
    # Case 2: Only interact with one object (e.g., press button)
    elif object_unattached:
        target_kp = object_vertices[object_unattached][0]
        
        # Generate approach trajectory
        trajectory_heatmap = generate_grasp_trajectory(
            target_kp,
            image_shape,
            start_position=start_position,
            motion_direction=motion_direction,
            sigma=sigma,
            num_interpolate=num_interpolate
        )
        
        if start_position:
            waypoints = [start_position, target_kp]
        else:
            # Infer start from motion direction
            h, w = image_shape
            if motion_direction == 'downward':
                waypoints = [(target_kp[0], target_kp[1] - 150), target_kp]
            else:
                waypoints = [(target_kp[0], target_kp[1] - 100), target_kp]
    
    else:
        # No valid objects, return empty
        trajectory_heatmap = np.zeros(image_shape)
        waypoints = []
    
    return trajectory_heatmap, waypoints


def plot_motion_trajectory(
    image: Image.Image,
    trajectory_heatmap: np.ndarray,
    waypoints: Optional[List[Tuple[float, float]]] = None,
    alpha: float = 0.6,
    show_waypoints: bool = True,
    show_arrows: bool = True,
    colormap: str = 'hot',
    title: str = 'Motion Trajectory',
    return_PIL: bool = True,
    fname: Optional[str] = None
) -> Image.Image:
    """
    Visualize motion trajectory with optional waypoints and direction arrows.
    
    Args:
        image: PIL Image
        trajectory_heatmap: Trajectory heatmap array
        waypoints: Optional list of waypoints to mark
        alpha: Heatmap transparency
        show_waypoints: Whether to show waypoint markers
        show_arrows: Whether to show direction arrows
        colormap: Matplotlib colormap
        title: Plot title
        return_PIL: Return PIL Image if True
        fname: Optional save filename
    
    Returns:
        PIL Image with trajectory visualization
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Create colormap with transparency
    if colormap == 'hot':
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.3), (1, 0.5, 0, 0.6), (1, 1, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('trajectory', colors)
    else:
        cmap = plt.get_cmap(colormap)
    
    # Plot trajectory heatmap
    plt.imshow(trajectory_heatmap, cmap=cmap, alpha=alpha)
    
    # Plot waypoints if provided
    if waypoints is not None and len(waypoints) > 0 and show_waypoints:
        waypoints_arr = np.array(waypoints)
        plt.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'o-', 
                color='cyan', markersize=10, linewidth=3, 
                markeredgecolor='white', markeredgewidth=2)
        
        # Annotate waypoints
        for i, wp in enumerate(waypoints_arr):
            label = 'Start' if i == 0 else ('Goal' if i == len(waypoints_arr)-1 else f'{i}')
            plt.annotate(label, (wp[0], wp[1]), 
                       fontsize=12, color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor='blue', alpha=0.7),
                       ha='center')
        
        # Draw arrows between waypoints
        if show_arrows and len(waypoints_arr) > 1:
            for i in range(len(waypoints_arr) - 1):
                dx = waypoints_arr[i+1, 0] - waypoints_arr[i, 0]
                dy = waypoints_arr[i+1, 1] - waypoints_arr[i, 1]
                plt.arrow(waypoints_arr[i, 0], waypoints_arr[i, 1], 
                         dx*0.8, dy*0.8,
                         head_width=15, head_length=20, 
                         fc='cyan', ec='white', linewidth=2, alpha=0.8)
    
    plt.title(title, fontsize=16, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if fname:
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
