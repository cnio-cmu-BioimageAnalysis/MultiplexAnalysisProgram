import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from typing import Dict, Callable, List, Optional, Union
from multiplex_pipeline.utils.helpers import extract_roi_number
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk


def post_process_mask(mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000) -> np.ndarray:
    """
    Fills small holes and removes small objects from a binary mask.

    Args:
        mask (np.ndarray): The binary mask to be processed.
        min_size (int): The minimum size of objects to be kept in the mask. Objects smaller than this size are removed.
        max_hole_size (int): The maximum size of holes to be filled.

    Returns:
        np.ndarray: The processed binary mask.
    """
    mask_bool = mask.astype(bool)
    
    if max_hole_size > 0:
        mask_bool = remove_small_holes(mask_bool, area_threshold=max_hole_size)
        print(f"Holes ≤ {max_hole_size} px filled.")
        
    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)
        print(f"Objects < {min_size} px removed.")
        
    return mask_bool


def post_process_mask_closing(mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000) -> np.ndarray:
    """
    Performs a large morphological closing to unify structures, ignoring size parameters.

    Args:
        mask (np.ndarray): The binary mask to be processed.
        min_size (int): Ignored in this function.
        max_hole_size (int): Ignored in this function.

    Returns:
        np.ndarray: The processed binary mask after morphological closing.
    """
    mask_bool = mask.astype(bool)
    selem = disk(20)
    closed = binary_closing(mask_bool, selem)
    print(f"Performed closing (disk radius=20).")
    
    return closed


def generate_initial_mask(
    channel_data: np.ndarray,
    score: Union[Dict[str, float], float],
    scaling_divisor: float
) -> (np.ndarray, np.ndarray):
    """
    Computes the initial binary mask by thresholding the channel data.

    Args:
        channel_data (np.ndarray): The raw channel data.
        score (Union[Dict[str, float], float]): The threshold score for each ROI or a single threshold for all.
        scaling_divisor (float): A scaling factor for thresholding.

    Returns:
        np.ndarray: The raw channel data and the binary initial mask.
    """
    m, s = channel_data.mean(), channel_data.std()
    thresh = m + (score / scaling_divisor) * 3 * s
    initial_mask = channel_data > thresh
    
    return channel_data, initial_mask


def apply_dapi_mask(
    initial_mask: np.ndarray,
    roi_num: str,
    dapi_masks: Dict[str, np.ndarray],
    require_dapi: bool,
    name: str
) -> Optional[np.ndarray]:
    """
    Intersects the initial mask with the DAPI ROI mask if required.

    Args:
        initial_mask (np.ndarray): The initial binary mask.
        roi_num (str): The ROI number.
        dapi_masks (Dict[str, np.ndarray]): A dictionary of DAPI masks.
        require_dapi (bool): Whether to apply the DAPI mask or not.
        name (str): The name of the image being processed.

    Returns:
        Optional[np.ndarray]: The masked binary mask, or None if the DAPI mask is missing and `require_dapi` is True.
    """
    if require_dapi:
        key = f"roi{roi_num}_dapi"
        dmask = dapi_masks.get(key)
        if dmask is None:
            print(f"Skipping {name}: missing DAPI mask {key}")
            return None
        return initial_mask & (dmask > 0)
    
    return initial_mask


def save_mask(
    mask: np.ndarray,
    base_folder: str,
    name: str,
    roi_num: str,
    filename: str
) -> None:
    """
    Saves the binary mask to disk under a folder named '{name} - ROI{roi_num}'.

    Args:
        mask (np.ndarray): The binary mask to save.
        base_folder (str): The base directory where masks are saved.
        name (str): The name associated with the image.
        roi_num (str): The ROI number.
        filename (str): The filename for the saved mask.
    """
    roi_folder = os.path.join(base_folder, f"{name} - ROI{roi_num}")
    os.makedirs(roi_folder, exist_ok=True)
    tifffile.imwrite(os.path.join(roi_folder, filename), mask.astype(np.uint8))
    print(f"Saved: {os.path.join(roi_folder, filename)}")


def display_masks(
    channel: np.ndarray,
    initial_mask: np.ndarray,
    processed_mask: np.ndarray,
    brightness_factor: Optional[float],
    mask_label: str,
    channel_index: int,
    name: str,
    min_size: int,
    max_hole_size: int
) -> None:
    """
    Displays the raw channel, initial mask, and post-processed mask side by side.

    Args:
        channel (np.ndarray): The raw channel data.
        initial_mask (np.ndarray): The initial binary mask.
        processed_mask (np.ndarray): The post-processed binary mask.
        brightness_factor (Optional[float]): Factor to adjust brightness of the channel.
        mask_label (str): The label for the mask.
        channel_index (int): The index of the channel being processed.
        name (str): The name of the image.
        min_size (int): The minimum object size for processing.
        max_hole_size (int): The maximum hole size for processing.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    disp = channel * brightness_factor if brightness_factor else channel
    axs[0].imshow(np.clip(disp, 0, channel.max()), cmap='gray')
    axs[0].set_title(f'{name}: {mask_label} (ch {channel_index})'); axs[0].axis('off')
    axs[1].imshow(initial_mask, cmap='gray')
    axs[1].set_title('Initial mask'); axs[1].axis('off')
    axs[2].imshow(processed_mask, cmap='gray')
    axs[2].set_title(f'Post-processed (min={min_size}, max_hole={max_hole_size})'); axs[2].axis('off')
    plt.tight_layout()
    plt.show()


def create_channel_masks(
    images_dict: Dict[str, np.ndarray],
    dapi_masks_dict: Dict[str, np.ndarray],
    channel_index: int,
    user_scores: Union[Dict[str, float], float],
    scaling_divisor: float,
    base_folder_path: str,
    min_size: int,
    max_hole_size: int,
    mask_label: str,
    mask_filename: str,
    post_process_funcs: List[Callable[[np.ndarray, int, int], np.ndarray]],
    brightness_factor: Optional[float] = None,
    require_dapi: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generates, post-processes, saves, and displays channel masks.

    Args:
        images_dict (Dict[str, np.ndarray]): A dictionary of images.
        dapi_masks_dict (Dict[str, np.ndarray]): A dictionary of DAPI masks.
        channel_index (int): The index of the channel to process.
        user_scores (Union[Dict[str, float], float]): The score to use for thresholding. [0-3] mu + x*sigma = threshold
        scaling_divisor (float): A scaling factor for thresholding.
        base_folder_path (str): The base folder to save the masks.
        min_size (int): The minimum size of objects to keep.
        max_hole_size (int): The maximum hole size to fill.
        mask_label (str): The label for the mask.
        mask_filename (str): The filename for saving the mask.
        post_process_funcs (List[Callable[[np.ndarray, int, int], np.ndarray]]): Functions for post-processing.
        brightness_factor (Optional[float]): Factor to adjust the brightness.
        require_dapi (bool): Whether to require a DAPI mask.

    Returns:
        Dict[str, np.ndarray]: A dictionary of processed masks indexed by ROI.
    """
    out_masks: Dict[str, np.ndarray] = {}

    for name, image in images_dict.items():
        roi_num = extract_roi_number(name)
        if roi_num is None:
            print(f"Skipping {name}: cannot extract ROI number.")
            continue

        # Initial mask
        chan = image[channel_index].astype(float)
        score_val = user_scores[name] if isinstance(user_scores, dict) else user_scores
        chan, initial_mask = generate_initial_mask(chan, score_val, scaling_divisor)

        # Apply DAPI mask if required
        masked = apply_dapi_mask(initial_mask, roi_num, dapi_masks_dict, require_dapi, name)
        if masked is None:
            continue

        # Post-process mask
        processed = masked.copy()
        for fn in post_process_funcs:
            processed = fn(processed, min_size=min_size, max_hole_size=max_hole_size)

        # Save and record the mask
        out_key = f"roi{roi_num}"
        out_masks[out_key] = processed
        save_mask(processed, base_folder_path, name, roi_num, mask_filename)

        # Display the masks
        display_masks(
            chan,
            masked,
            processed,
            brightness_factor,
            mask_label,
            channel_index,
            name,
            min_size,
            max_hole_size
        )

    return out_masks
