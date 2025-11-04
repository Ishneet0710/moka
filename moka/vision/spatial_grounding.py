"""
Spatial-aware grounding utilities for robust object detection.
Helps disambiguate objects when text descriptions alone are insufficient.
"""

import torch
import numpy as np
from PIL import Image
from groundingdino.util import box_ops
from .segmentation import load_pil_image, prepare_dino
from groundingdino.util.inference import predict


def get_spatially_filtered_bboxes(
    image,
    texts,
    spatial_priors=None,
    visualize=False,
    logdir=None
):
    """
    Get bounding boxes with spatial filtering to disambiguate similar objects.

    This is useful when you have multiple similar objects (e.g., two red cubes)
    and text descriptions alone aren't sufficient to distinguish them.

    Args:
        image: PIL Image
        texts: List of object descriptions (e.g., ['red cube', 'green tile'])
        spatial_priors: Dict mapping object descriptions to spatial preferences.
                       Each entry can specify:
                       - 'prefer_top': Select object with smallest y-coordinate (highest in image)
                       - 'prefer_bottom': Select object with largest y-coordinate (lowest in image)
                       - 'prefer_left': Select object with smallest x-coordinate (leftmost)
                       - 'prefer_right': Select object with largest x-coordinate (rightmost)
                       - 'min_confidence': Minimum confidence threshold (overrides default)

                       Example: {'red cube': 'prefer_top', 'green tile': 'prefer_bottom'}
        visualize: Whether to show detection visualization
        logdir: Directory to save visualization (if None, just displays)

    Returns:
        boxes: Tensor of bounding boxes [N, 4] in cxcywh format
        logits: Tensor of confidence scores [N]
        phrases: List of detected phrases [N]

    Example:
        >>> spatial_priors = {
        ...     'red cube': 'prefer_top',  # Select cube higher in the image (in gripper)
        ...     'green tile': None  # No spatial filtering, use highest confidence
        ... }
        >>> boxes, logits, phrases = get_spatially_filtered_bboxes(
        ...     image, ['red cube', 'green tile'], spatial_priors
        ... )
    """
    image_np, image_torch = load_pil_image(image)
    model = prepare_dino(dino_path='./ckpts')

    # Default thresholds
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.22

    all_boxes, all_logits, all_phrases = [], [], []

    for text in texts:
        # Get all detections for this text
        boxes, logits, phrases = predict(
            model=model,
            image=image_torch,
            caption=text,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        if len(boxes) == 0:
            print(f"Warning: No detections found for '{text}'")
            continue

        # Apply spatial filtering if specified
        spatial_prior = None
        if spatial_priors is not None and text in spatial_priors:
            spatial_prior = spatial_priors[text]

        # Select the best box based on spatial prior
        if spatial_prior is None or len(boxes) == 1:
            # No spatial filtering: use highest confidence
            id = torch.argmax(logits)
        elif spatial_prior == 'prefer_top':
            # Select box with smallest cy (highest in image)
            id = torch.argmin(boxes[:, 1])
            print(f"  '{text}': Selected top object (cy={boxes[id, 1]:.3f})")
        elif spatial_prior == 'prefer_bottom':
            # Select box with largest cy (lowest in image)
            id = torch.argmax(boxes[:, 1])
            print(f"  '{text}': Selected bottom object (cy={boxes[id, 1]:.3f})")
        elif spatial_prior == 'prefer_left':
            # Select box with smallest cx (leftmost)
            id = torch.argmin(boxes[:, 0])
            print(f"  '{text}': Selected left object (cx={boxes[id, 0]:.3f})")
        elif spatial_prior == 'prefer_right':
            # Select box with largest cx (rightmost)
            id = torch.argmax(boxes[:, 0])
            print(f"  '{text}': Selected right object (cx={boxes[id, 0]:.3f})")
        else:
            print(f"Warning: Unknown spatial prior '{spatial_prior}', using confidence")
            id = torch.argmax(logits)

        all_boxes.append(boxes[id])
        all_logits.append(logits[id])
        all_phrases.append(phrases[id])

    if len(all_boxes) == 0:
        raise RuntimeError("No objects detected! Check your text prompts or image.")

    boxes_tensor = torch.stack(all_boxes, dim=0)
    logits_tensor = torch.stack(all_logits, dim=0)

    # Visualize if requested
    if visualize:
        from groundingdino.util.inference import annotate
        import matplotlib.pyplot as plt
        import os

        H, W, _ = image_np.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_tensor) * torch.Tensor([W, H, W, H])
        annotated_frame = annotate(
            image_source=image_np,
            boxes=boxes_tensor,
            logits=logits_tensor,
            phrases=all_phrases
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        annotated_frame_pil = Image.fromarray(annotated_frame)

        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            annotated_frame_pil.save(os.path.join(logdir, 'bbox_spatial.png'))

        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_frame_pil)
        plt.axis('off')
        plt.title('Spatially Filtered Detections', fontsize=14, weight='bold')
        plt.show()

    return boxes_tensor, logits_tensor, all_phrases


def get_scene_object_bboxes_spatial(image, texts, spatial_priors=None, visualize=False, logdir=None):
    """
    Drop-in replacement for get_scene_object_bboxes with spatial filtering support.

    Args:
        image: PIL Image
        texts: List of object descriptions
        spatial_priors: Dict mapping object descriptions to spatial preferences
                       (see get_spatially_filtered_bboxes for details)
        visualize: Whether to visualize detections
        logdir: Directory to save visualizations

    Returns:
        boxes: Bounding boxes in cxcywh format
        logits: Confidence scores
        phrases: Detected phrases
    """
    return get_spatially_filtered_bboxes(image, texts, spatial_priors, visualize, logdir)
