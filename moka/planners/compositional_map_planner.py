"""
Compositional Map Planner
Integrates VoxPoser-style compositional heatmaps with MOKA's tile-based visual prompting.

This module generates affordance, collision, and transition maps using VLM (GPT-4V),
then composes them into a final heatmap for trajectory planning.
"""

import numpy as np
import json
import re
from PIL import Image
from typing import Dict, List, Tuple, Optional

from moka.gpt_utils import request_gpt, request_gemini
from moka.planners.visual_prompt_utils import annotate_visual_prompts
from moka.vision.compositional_heatmap import (
    generate_affordance_map,
    generate_collision_map,
    generate_transition_map,
    compose_heatmaps,
    normalize_map,
    visualize_composition
)


def parse_json_response(response: str) -> dict:
    """
    Parse JSON from VLM response, handling markdown code blocks.

    Args:
        response: VLM response string (may contain markdown)

    Returns:
        Parsed JSON dictionary
    """
    # Strip markdown code blocks if present
    response = response.strip()
    if response.startswith('```'):
        # Remove opening ```json or ```
        lines = response.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove closing ```
        if lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines)

    return json.loads(response)


def filter_collision_affordance_conflicts(
    collision_tiles: List[str],
    affordance_tiles: List[str],
    debug: bool = False
) -> List[str]:
    """
    Remove collision tiles that overlap with affordance tiles.

    This is a sanity check to prevent VLM hallucinations where it marks
    the goal location as a collision zone (which makes no sense).

    Args:
        collision_tiles: List of collision tile labels (e.g., ['b2', 'b3'])
        affordance_tiles: List of affordance tile labels (e.g., ['b2', 'c3'])
        debug: Whether to print debug information

    Returns:
        Filtered collision tiles with overlaps removed
    """
    if not collision_tiles or not affordance_tiles:
        return collision_tiles

    # Convert to sets for efficient comparison
    collision_set = set(collision_tiles)
    affordance_set = set(affordance_tiles)

    # Find conflicts
    conflicts = collision_set & affordance_set

    if conflicts and debug:
        print(f"\n⚠ WARNING: VLM marked goal tiles as collision zones!")
        print(f"  Conflicting tiles: {sorted(conflicts)}")
        print(f"  This is likely a VLM hallucination - removing from collision map")

    # Remove conflicts
    filtered = list(collision_set - affordance_set)

    return filtered


def request_compositional_maps(
    subtask: Dict,
    obs_image: Image.Image,
    annotated_image: Image.Image,
    candidate_keypoints: Dict,
    prompts: Dict,
    waypoint_grid_size: Tuple[int, int] = (5, 5),
    debug: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Request VLM to generate affordance, collision, and transition maps.

    Args:
        subtask: Task information dict with keys:
                 'instruction', 'object_grasped', 'object_unattached', 'motion_direction'
        obs_image: Original observation image
        annotated_image: Image with visual marks (keypoints and grid)
        candidate_keypoints: Dict mapping object names to keypoint arrays
        prompts: Dict containing prompt templates
        waypoint_grid_size: Grid dimensions (rows, cols)
        debug: Whether to print debug information

    Returns:
        Tuple of:
        - maps_dict: Dictionary with keys 'affordance', 'collision', 'transition'
                     mapping to 2D numpy arrays
        - responses_dict: Dictionary with raw VLM responses for each map type
    """
    image_shape = (obs_image.height, obs_image.width)

    # Prepare task context
    task_context = f"""
Task Information:
- Instruction: {subtask['instruction']}
- Object grasped: {subtask.get('object_grasped', '')}
- Object unattached: {subtask.get('object_unattached', '')}
- Motion direction: {subtask.get('motion_direction', '')}
"""

    maps_dict = {}
    responses_dict = {}

    # ==================== AFFORDANCE MAP ====================
    if debug:
        print("\n" + "="*70)
        print("| Generating Affordance Map")
        print("="*70)

    affordance_prompt = prompts['generate_affordance_map'] + "\n\n" + task_context
    affordance_response = request_gemini(
        affordance_prompt,
        images=[annotated_image],
        model_name="gemini-2.5-pro"
    )

    if debug:
        print("VLM Response:")
        print(affordance_response)

    # Parse response
    try:
        affordance_json = parse_json_response(affordance_response)
        responses_dict['affordance'] = affordance_json

        affordance_map = generate_affordance_map(
            affordance_json,
            image_shape,
            grid_size=waypoint_grid_size
        )
        maps_dict['affordance'] = affordance_map

        if debug:
            print(f"✓ Affordance map generated: {len(affordance_json.get('affordance_tiles', []))} tiles")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠ Failed to parse affordance response: {e}")
        maps_dict['affordance'] = np.zeros(image_shape)
        responses_dict['affordance'] = {}

    # ==================== COLLISION MAP ====================
    if debug:
        print("\n" + "="*70)
        print("| Generating Collision Map")
        print("="*70)

    collision_prompt = prompts['generate_collision_map'] + "\n\n" + task_context
    collision_response = request_gemini(
        collision_prompt,
        images=[annotated_image],
        model_name="gemini-2.5-pro"
    )

    if debug:
        print("VLM Response:")
        print(collision_response)

    # Parse response
    try:
        collision_json = parse_json_response(collision_response)

        # SANITY CHECK: Warn about collision/affordance overlaps but DON'T remove them
        # (Affordance map might hallucinate, so we keep collision zones as-is)
        if 'collision_tiles' in collision_json and 'affordance' in responses_dict:
            affordance_tiles = responses_dict['affordance'].get('affordance_tiles', [])
            original_collision_tiles = collision_json['collision_tiles']

            conflicts = set(original_collision_tiles) & set(affordance_tiles)
            if conflicts and debug:
                print(f"\n⚠ WARNING: Tiles marked as BOTH affordance AND collision: {sorted(conflicts)}")
                print(f"  This might indicate VLM confusion. Keeping collision zones as-is for safety.")

        responses_dict['collision'] = collision_json

        collision_map = generate_collision_map(
            collision_json,
            image_shape,
            grid_size=waypoint_grid_size,
            sigma=40.0  # Gaussian smoothing for safety margins
        )
        maps_dict['collision'] = collision_map

        if debug:
            print(f"✓ Collision map generated: {len(collision_json.get('collision_tiles', []))} tiles")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠ Failed to parse collision response: {e}")
        maps_dict['collision'] = np.zeros(image_shape)
        responses_dict['collision'] = {}

    # ==================== TRANSITION MAP ====================
    if debug:
        print("\n" + "="*70)
        print("| Generating Transition Map")
        print("="*70)

    transition_prompt = prompts['generate_transition_map'] + "\n\n" + task_context
    transition_response = request_gemini(
        transition_prompt,
        images=[annotated_image],
        model_name="gemini-2.5-pro"
    )

    if debug:
        print("VLM Response:")
        print(transition_response)

    # Parse response
    try:
        transition_json = parse_json_response(transition_response)
        responses_dict['transition'] = transition_json

        transition_map = generate_transition_map(
            transition_json,
            image_shape,
            grid_size=waypoint_grid_size
        )
        maps_dict['transition'] = transition_map

        if debug:
            print(f"✓ Transition map generated: {len(transition_json.get('transition_tiles', []))} tiles")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠ Failed to parse transition response: {e}")
        maps_dict['transition'] = np.zeros(image_shape)
        responses_dict['transition'] = {}

    return maps_dict, responses_dict


def generate_compositional_heatmap(
    subtask: Dict,
    obs_image: Image.Image,
    candidate_keypoints: Dict,
    prompts: Dict,
    waypoint_grid_size: Tuple[int, int] = (5, 5),
    affordance_weight: float = 2.0,
    collision_weight: float = 1.0,
    transition_weight: float = 1.0,
    smooth_sigma: float = 40.0,
    debug: bool = False,
    visualize: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Generate compositional heatmap for a subtask using VLM-guided map generation.

    This is the main entry point that:
    1. Creates annotated image with visual marks
    2. Requests VLM to generate affordance, collision, transition maps
    3. Composes maps using weighted combination
    4. Optionally visualizes the results

    Args:
        subtask: Task information dict
        obs_image: Original observation image
        candidate_keypoints: Dict mapping object names to keypoint arrays
        prompts: Dict containing prompt templates
        waypoint_grid_size: Grid dimensions (rows, cols)
        affordance_weight: Weight for affordance map (default 2.0)
        collision_weight: Weight for collision map (default 1.0)
        transition_weight: Weight for transition map (default 1.0)
        smooth_sigma: Additional Gaussian smoothing after composition (default 40.0).
                     Higher values (40-60) create smoother, more continuous VoxPoser-style heatmaps.
                     Lower values (0-20) preserve discrete MOKA-style peaks.
                     Set to 0 to disable post-composition smoothing.
        debug: Whether to print debug information
        visualize: Whether to display visualization

    Returns:
        Tuple of:
        - composed_heatmap: Final composed heatmap (2D array)
        - info_dict: Dictionary containing individual maps and VLM responses
    """
    # Step 1: Create annotated image
    annotated_image = annotate_visual_prompts(
        obs_image,
        candidate_keypoints,
        waypoint_grid_size
    )

    # Step 2: Request maps from VLM
    maps_dict, responses_dict = request_compositional_maps(
        subtask,
        obs_image,
        annotated_image,
        candidate_keypoints,
        prompts,
        waypoint_grid_size,
        debug=debug
    )

    # Step 3: Compose maps
    if debug:
        print("\n" + "="*70)
        print("| Composing Heatmaps")
        print("="*70)
        print(f"Weights: affordance={affordance_weight}, collision={collision_weight}, transition={transition_weight}")
        print(f"Post-composition smoothing: sigma={smooth_sigma}")

    composed_heatmap = compose_heatmaps(
        maps_dict['affordance'],
        maps_dict['collision'],
        maps_dict.get('transition'),
        affordance_weight=affordance_weight,
        collision_weight=collision_weight,
        transition_weight=transition_weight,
        smooth_sigma=smooth_sigma
    )

    if debug:
        print(f"✓ Composed heatmap shape: {composed_heatmap.shape}")
        print(f"  Value range: [{composed_heatmap.min():.3f}, {composed_heatmap.max():.3f}]")

    # Step 4: Visualize if requested
    if visualize:
        fig = visualize_composition(
            obs_image,
            maps_dict['affordance'],
            maps_dict['collision'],
            maps_dict.get('transition'),
            composed_heatmap,
            title=f"Compositional Heatmap: {subtask['instruction']}"
        )
        import matplotlib.pyplot as plt
        plt.show()

    # Prepare info dict
    info_dict = {
        'maps': maps_dict,
        'responses': responses_dict,
        'weights': {
            'affordance': affordance_weight,
            'collision': collision_weight,
            'transition': transition_weight
        }
    }

    return composed_heatmap, info_dict


def extract_trajectory_from_heatmap(
    heatmap: np.ndarray,
    start_point: Optional[Tuple[float, float]] = None,
    end_point: Optional[Tuple[float, float]] = None,
    num_waypoints: int = 10,
    method: str = 'gradient'
) -> np.ndarray:
    """
    Extract trajectory waypoints from composed heatmap.

    Args:
        heatmap: Composed heatmap (high values = preferred regions)
        start_point: Starting (x, y) position, if None uses highest value in bottom half
        end_point: Ending (x, y) position, if None uses highest value in top half
        num_waypoints: Number of waypoints to extract
        method: Extraction method ('gradient' for gradient ascent, 'peaks' for local maxima)

    Returns:
        Array of waypoints with shape (num_waypoints, 2) containing (x, y) coordinates
    """
    h, w = heatmap.shape

    # Determine start point if not provided
    if start_point is None:
        # Find highest value in bottom half
        bottom_half = heatmap[h//2:, :]
        max_idx = np.unravel_index(bottom_half.argmax(), bottom_half.shape)
        start_point = (max_idx[1], max_idx[0] + h//2)

    # Determine end point if not provided
    if end_point is None:
        # Find highest value in top half
        top_half = heatmap[:h//2, :]
        max_idx = np.unravel_index(top_half.argmax(), top_half.shape)
        end_point = (max_idx[1], max_idx[0])

    if method == 'gradient':
        # Use gradient ascent
        from moka.vision.compositional_heatmap import sample_trajectory_from_heatmap
        trajectory = sample_trajectory_from_heatmap(
            heatmap,
            start_point,
            num_steps=num_waypoints,
            step_size=max(h, w) / num_waypoints
        )
    elif method == 'peaks':
        # Find local maxima along path
        # Simple interpolation between start and end with local peak refinement
        t = np.linspace(0, 1, num_waypoints)
        trajectory = []
        for t_i in t:
            # Linear interpolation
            x = start_point[0] + t_i * (end_point[0] - start_point[0])
            y = start_point[1] + t_i * (end_point[1] - start_point[1])

            # Refine to local maximum in neighborhood
            x_int, y_int = int(x), int(y)
            search_radius = 10
            y_min = max(0, y_int - search_radius)
            y_max = min(h, y_int + search_radius)
            x_min = max(0, x_int - search_radius)
            x_max = min(w, x_int + search_radius)

            local_region = heatmap[y_min:y_max, x_min:x_max]
            if local_region.size > 0:
                local_max = np.unravel_index(local_region.argmax(), local_region.shape)
                x = x_min + local_max[1]
                y = y_min + local_max[0]

            trajectory.append([x, y])

        trajectory = np.array(trajectory)
    else:
        raise ValueError(f"Unknown method: {method}")

    return trajectory
