"""
Enhanced motion planning with continuous trajectory heatmaps and attention weighting.
Extends MOKA's request_motion() with Gaussian trajectories and attention focus.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from moka.vision.trajectory_heatmap import connect_waypoints_with_gaussians
from moka.vision.attention_extraction import generate_attention_focused_trajectory


def request_motion_with_trajectory(
    subtask: Dict,
    obs_image: Image.Image,
    annotated_image: Image.Image,
    candidate_keypoints: List[Tuple[float, float]],
    waypoint_grid_size: int,
    prompts: Dict,
    debug: bool = False,
    sigma: float = 20,
    num_interpolate: int = 50,
    attention_alpha: float = 0.5,
    generate_heatmap: bool = True
) -> Tuple[Dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Enhanced version of request_motion() that adds continuous trajectory heatmaps.
    
    Args:
        subtask: Subtask dictionary with instruction, objects, motion_direction
        obs_image: Observation image
        annotated_image: Image with visual prompts
        candidate_keypoints: List of candidate keypoint positions
        waypoint_grid_size: Grid size for waypoint annotation
        prompts: Prompt templates
        debug: Whether to print debug info
        sigma: Gaussian spread for trajectory (15-25 recommended)
        num_interpolate: Interpolation points between waypoints (50-70 recommended)
        attention_alpha: Attention weighting (0.5 = balanced)
        generate_heatmap: Whether to generate trajectory heatmap
    
    Returns:
        context: Enhanced context with trajectory heatmaps
        trajectory_heatmap: Base trajectory heatmap (or None)
        focused_trajectory: Attention-focused trajectory (or None)
    """
    from moka.planners.visual_prompt_utils import request_motion
    
    # Call original MOKA motion selection
    context, _, _ = request_motion(
        subtask,
        obs_image,
        annotated_image,
        candidate_keypoints,
        waypoint_grid_size=waypoint_grid_size,
        prompts=prompts,
        debug=debug
    )
    
    if not generate_heatmap:
        return context, None, None
    
    # Extract waypoint positions from context
    # MOKA uses grasp_keypoint, function_keypoint, target_keypoint
    waypoint_positions = []
    
    # Build trajectory: grasp -> function -> target
    if 'grasp_keypoint' in context and context['grasp_keypoint'] is not None:
        waypoint_positions.append(context['grasp_keypoint'])
    
    if 'function_keypoint' in context and context['function_keypoint'] is not None:
        waypoint_positions.append(context['function_keypoint'])
    
    if 'target_keypoint' in context and context['target_keypoint'] is not None:
        waypoint_positions.append(context['target_keypoint'])
    
    # Fallback: check for 'waypoints' field (if exists)
    if len(waypoint_positions) == 0 and 'waypoints' in context:
        waypoint_ids = context['waypoints']
        for wp_id in waypoint_ids:
            if wp_id < len(candidate_keypoints):
                waypoint_positions.append(candidate_keypoints[wp_id])
    
    if len(waypoint_positions) == 0:
        if debug:
            print("Warning: No keypoints found in motion context (grasp/function/target)")
        return context, None, None
    
    waypoint_positions = np.array(waypoint_positions)
    
    if debug:
        print(f"\n✓ Extracted {len(waypoint_positions)} waypoints:")
        if 'grasp_keypoint' in context and context['grasp_keypoint'] is not None:
            print(f"  - Grasp: {context['grasp_keypoint']}")
        if 'function_keypoint' in context and context['function_keypoint'] is not None:
            print(f"  - Function: {context['function_keypoint']}")
        if 'target_keypoint' in context and context['target_keypoint'] is not None:
            print(f"  - Target: {context['target_keypoint']}")
    
    # Generate continuous Gaussian trajectory
    image_shape = (obs_image.height, obs_image.width)
    
    trajectory_heatmap = connect_waypoints_with_gaussians(
        waypoints=waypoint_positions,
        image_shape=image_shape,
        sigma=sigma,
        num_interpolate=num_interpolate
    )
    
    # Add attention weighting
    focus_object = subtask.get('object_grasped', '') or subtask.get('object_unattached', '')
    
    if focus_object:
        focused_trajectory, attention_map = generate_attention_focused_trajectory(
            image=obs_image,
            trajectory_heatmap=trajectory_heatmap,
            object_name=focus_object,
            alpha=attention_alpha,
            combine_method='multiply'
        )
    else:
        focused_trajectory = trajectory_heatmap
        attention_map = None
    
    # Enhance context with trajectory data
    context['trajectory_heatmap'] = trajectory_heatmap
    context['focused_trajectory'] = focused_trajectory
    context['attention_map'] = attention_map
    context['waypoint_positions'] = waypoint_positions
    
    if debug:
        print(f"\n✓ Enhanced motion context with trajectory heatmaps")
        print(f"  Waypoints: {len(waypoint_positions)}")
        print(f"  Trajectory shape: {trajectory_heatmap.shape}")
        print(f"  Attention: {'Yes' if attention_map is not None else 'No'}")
    
    return context, trajectory_heatmap, focused_trajectory


def visualize_enhanced_motion(
    context: Dict,
    obs_image: Image.Image,
    show_original: bool = True,
    show_trajectory: bool = True,
    show_attention: bool = True,
    show_focused: bool = True,
    alpha: float = 0.6
) -> Image.Image:
    """
    Visualize enhanced motion context with trajectory heatmaps.
    
    Args:
        context: Enhanced context from request_motion_with_trajectory()
        obs_image: Original observation image
        show_original: Show original image
        show_trajectory: Show trajectory heatmap
        show_attention: Show attention map
        show_focused: Show focused trajectory
        alpha: Heatmap transparency
    
    Returns:
        visualization: PIL Image with visualizations
    """
    from moka.vision.attention_extraction import visualize_attention_and_trajectory
    
    trajectory_heatmap = context.get('trajectory_heatmap')
    focused_trajectory = context.get('focused_trajectory')
    attention_map = context.get('attention_map')
    waypoint_positions = context.get('waypoint_positions')
    
    if trajectory_heatmap is None:
        print("Warning: No trajectory heatmap in context")
        return obs_image
    
    visualization = visualize_attention_and_trajectory(
        image=obs_image,
        trajectory_heatmap=trajectory_heatmap,
        attention_map=attention_map,
        focused_trajectory=focused_trajectory,
        waypoints=waypoint_positions,
        title="Enhanced Motion with Trajectory Heatmaps",
        return_PIL=True
    )
    
    return visualization


# Convenience function for quick usage
def enhance_motion_context(
    context: Dict,
    obs_image: Image.Image,
    candidate_keypoints: List[Tuple[float, float]],
    subtask: Dict,
    sigma: float = 20,
    num_interpolate: int = 50,
    attention_alpha: float = 0.5
) -> Dict:
    """
    Add trajectory heatmaps to existing motion context.
    Use this if you already called request_motion() and want to add heatmaps.
    
    Args:
        context: Existing context from request_motion()
        obs_image: Observation image
        candidate_keypoints: List of candidate keypoint positions
        subtask: Subtask dictionary
        sigma: Gaussian spread
        num_interpolate: Interpolation points
        attention_alpha: Attention weighting
    
    Returns:
        Enhanced context with trajectory heatmaps
    """
    # Extract waypoint positions from context
    waypoint_positions = []
    
    # Build trajectory: grasp -> function -> target
    if 'grasp_keypoint' in context and context['grasp_keypoint'] is not None:
        waypoint_positions.append(context['grasp_keypoint'])
    
    if 'function_keypoint' in context and context['function_keypoint'] is not None:
        waypoint_positions.append(context['function_keypoint'])
    
    if 'target_keypoint' in context and context['target_keypoint'] is not None:
        waypoint_positions.append(context['target_keypoint'])
    
    # Fallback: check for 'waypoints' field
    if len(waypoint_positions) == 0 and 'waypoints' in context:
        waypoint_ids = context['waypoints']
        waypoint_positions = [candidate_keypoints[wp_id] for wp_id in waypoint_ids 
                             if wp_id < len(candidate_keypoints)]
    
    if len(waypoint_positions) == 0:
        return context
    
    waypoint_positions = np.array(waypoint_positions)
    
    # Generate trajectory
    image_shape = (obs_image.height, obs_image.width)
    trajectory_heatmap = connect_waypoints_with_gaussians(
        waypoints=waypoint_positions,
        image_shape=image_shape,
        sigma=sigma,
        num_interpolate=num_interpolate
    )
    
    # Add attention
    focus_object = subtask.get('object_grasped', '') or subtask.get('object_unattached', '')
    if focus_object:
        focused_trajectory, attention_map = generate_attention_focused_trajectory(
            image=obs_image,
            trajectory_heatmap=trajectory_heatmap,
            object_name=focus_object,
            alpha=attention_alpha
        )
    else:
        focused_trajectory = trajectory_heatmap
        attention_map = None
    
    # Add to context
    context['trajectory_heatmap'] = trajectory_heatmap
    context['focused_trajectory'] = focused_trajectory
    context['attention_map'] = attention_map
    context['waypoint_positions'] = waypoint_positions
    
    return context
