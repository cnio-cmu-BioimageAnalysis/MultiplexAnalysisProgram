


def post_process_mask(mask, min_size=0, max_hole_size=10000):
    """
    Performs post-processing on a binary mask.
    
    Parameters:
    - mask (np.ndarray): Binary mask to process.
    - min_size (int): Minimum size of objects to retain.
    - max_hole_size (int): Maximum size of holes to fill.
    
    Returns:
    - np.ndarray: Post-processed mask.
    """

    from skimage.morphology import remove_small_objects, remove_small_holes


    # Ensure that the mask is of boolean type
    mask = mask.astype(bool)
    
    # Fill small holes that are less than or equal to max_hole_size
    if max_hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=max_hole_size)
        print(f"Holes smaller than or equal to {max_hole_size} pixels have been filled.")

    # Remove small objects that are smaller than min_size
    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)
        print(f"Objects smaller than {min_size} pixels have been removed.")
    
    return mask



def post_process_mask_closing(mask, min_size=0, max_hole_size=10000):
    """
    Performs a large morphological closing on the binary mask.
    This combines a dilation followed by an erosion to fill holes
    and unify the mask.
    """
    from skimage.morphology import binary_closing, disk

    # Convert mask to boolean
    mask = mask.astype(bool)
    
    # Perform a "large" closing to fill holes and unify structures
    selem = disk(20)  # Adjust the disk radius as needed
    mask = binary_closing(mask, selem)
    
    return mask


# multiplex_pipeline/preprocessing/segmentation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from typing import Dict, Any, Callable, List, Union
from multiplex_pipeline.utils.helpers import extract_roi_number
from multiplex_pipeline.preprocessing.segmentation import (
    post_process_mask,
    post_process_mask_closing,
)
# multiplex_pipeline/preprocessing/segmentation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from typing import Dict, Any, Callable, List, Union
from multiplex_pipeline.utils.helpers import extract_roi_number
from multiplex_pipeline.preprocessing.segmentation import (
    post_process_mask,
    post_process_mask_closing,
)

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
    brightness_factor: Union[float, None] = None,
    require_dapi: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Idéntico a antes, pero ahora guarda `initial_mask` para plotearla.
    """
    out_masks: Dict[str, np.ndarray] = {}

    for name, image in images_dict.items():
        roi_num = extract_roi_number(name)
        if roi_num is None:
            print(f"Saltando {name}: no pude extraer ROI")
            continue

        # obtener score
        score = user_scores[name] if isinstance(user_scores, dict) else user_scores
        scaling = (score / scaling_divisor) * 3

        # extraer canal y calcular umbral
        chan = image[channel_index, ...].astype(float)
        m, s = chan.mean(), chan.std()
        thresh = m + scaling * s
        initial_mask = chan > thresh

        # combinar con DAPI si hace falta
        if require_dapi:
            key = f"roi{roi_num}_dapi"
            dmask = dapi_masks_dict.get(key)
            if dmask is None:
                print(f"Saltando {name}: falta DAPI {key}")
                continue
            initial_mask &= (dmask > 0)

        # post-procesado
        mask_post = initial_mask.copy()
        for fn in post_process_funcs:
            mask_post = fn(mask_post, min_size=min_size, max_hole_size=max_hole_size)

        # guardar en dict y en disco
        out_key = f"roi{roi_num}"
        out_masks[out_key] = mask_post
        roi_folder = os.path.join(base_folder_path, f"{name} - ROI{roi_num}")
        os.makedirs(roi_folder, exist_ok=True)
        tifffile.imwrite(
            os.path.join(roi_folder, mask_filename),
            mask_post.astype(np.uint8)
        )
        print(f"Guardado: {os.path.join(roi_folder, mask_filename)}")

        # plotting: canal / máscara inicial / máscara post-proc
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        disp = chan * brightness_factor if brightness_factor else chan
        if brightness_factor:
            disp = np.clip(disp, 0, chan.max())
        axs[0].imshow(disp, cmap='gray')
        axs[0].set_title(f'{name}: {mask_label} (ch {channel_index})')
        axs[0].axis('off')

        axs[1].imshow(initial_mask, cmap='gray')
        axs[1].set_title(f'{mask_label} Initial Binary Mask')
        axs[1].axis('off')

        axs[2].imshow(mask_post, cmap='gray')
        axs[2].set_title(
            f'Post-Processed Mask\n(min={min_size}, max_hole={max_hole_size})'
        )
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    return out_masks
