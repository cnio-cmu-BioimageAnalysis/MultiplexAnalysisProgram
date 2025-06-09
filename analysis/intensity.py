import numpy as np
import pandas as pd
from skimage import measure
from tqdm import tqdm
from typing import Dict, Optional

from multiplex_pipeline.config import ROI_PATTERN, DAPI_CONNECTIVITY, PIXEL_AREA
from multiplex_pipeline.utils.validation import verify_binary


def extract_roi_key(img_name: str) -> Optional[str]:
    """
    Extracts the ROI key (Region of Interest) from the image name.

    Args:
        img_name (str): The image file name.

    Returns:
        Optional[str]: The extracted ROI key from the image name, or None if not found.
    """
    m = ROI_PATTERN.search(img_name)
    return m.group(1).lower() if m else None


def label_dapi(mask: np.ndarray) -> np.ndarray:
    """
    Labels the DAPI mask, returning it if already labeled or labeling it if binary.

    Args:
        mask (np.ndarray): The binary DAPI mask.

    Returns:
        np.ndarray: The labeled mask.
    """
    if len(np.unique(mask)) <= 2:
        return measure.label(mask, connectivity=DAPI_CONNECTIVITY)
    return mask


def get_labels_and_counts(labeled_mask: np.ndarray) -> tuple:
    """
    Retrieves labels and counts of regions in the labeled mask.

    Args:
        labeled_mask (np.ndarray): The labeled mask.

    Returns:
        tuple: A tuple with the labels, counts, and flattened mask.
    """
    flat = labeled_mask.ravel()
    counts = np.bincount(flat)
    labels = np.arange(len(counts))
    valid = (labels != 0) & (counts > 0)
    return labels[valid], counts[valid], flat


def get_centroids_map(labeled_mask: np.ndarray) -> Dict[int, tuple]:
    """
    Retrieves centroids of labeled regions in the mask.

    Args:
        labeled_mask (np.ndarray): The labeled mask.

    Returns:
        Dict[int, tuple]: A dictionary with the label as the key and the centroid as the value.
    """
    return {p.label: p.centroid for p in measure.regionprops(labeled_mask)}


def compute_mean_intensities(mask_flat: np.ndarray, img_channel: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Computes the mean intensities for each region in the image channel.

    Args:
        mask_flat (np.ndarray): The flattened mask.
        img_channel (np.ndarray): The image channel data for intensity computation.
        valid (np.ndarray): Valid labels for regions.
        counts (np.ndarray): Pixel counts for each region.

    Returns:
        np.ndarray: An array of mean intensities for each region.
    """
    sums = np.bincount(mask_flat, weights=img_channel.ravel())
    return sums[valid] / counts


def compute_binary_flags(valid_labels: np.ndarray, mask_flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Computes binary flags for valid labels in the mask.

    Args:
        valid_labels (np.ndarray): The valid labels.
        mask_flat (np.ndarray): The flattened mask.
        mask (np.ndarray): The binary mask to check against.

    Returns:
        np.ndarray: A binary array indicating if a label is present in the mask.
    """
    verify_binary(mask, "mask")
    flat = mask.ravel()
    positive = np.unique(mask_flat[flat > 0])
    return np.isin(valid_labels, positive).astype(int)


def process_roi(
    img_name: str,
    img_data: np.ndarray,
    dapi_masks: Dict[str, np.ndarray],
    ck_masks: Dict[str, np.ndarray],
    ngfr_masks: Dict[str, np.ndarray],
    channels: list,
    marker_dict: Dict[str, str],
    pixel_area_um2: float = PIXEL_AREA
) -> Optional[pd.DataFrame]:
    """
    Processes the ROI of an image to compute intensity and binary flags for different channels.

    Args:
        img_name (str): The image name.
        img_data (np.ndarray): The image data (channels).
        dapi_masks (Dict[str, np.ndarray]): DAPI masks by ROI.
        ck_masks (Dict[str, np.ndarray]): CK masks by ROI.
        ngfr_masks (Dict[str, np.ndarray]): NGFR masks by ROI.
        channels (list): List of channels to analyze.
        marker_dict (Dict[str, str]): A dictionary of channel markers.
        pixel_area_um2 (float, optional): The area in square micrometers per pixel. Default is PIXEL_AREA.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the ROI results, or None if an error occurs.
    """
    roi = extract_roi_key(img_name)
    if roi is None:
        print(f"ROI not found in '{img_name}'")
        return None

    dapi_key = f"{roi}_dapi"
    if dapi_key not in dapi_masks:
        print(f"{dapi_key} not found in dapi_masks")
        return None

    mask_dapi = dapi_masks[dapi_key]
    if mask_dapi.shape != img_data.shape[1:]:
        print(f"Dimensions do not match for {img_name}")
        return None

    lbl = label_dapi(mask_dapi)
    valid_labels, counts, flat = get_labels_and_counts(lbl)
    cent_map = get_centroids_map(lbl)

    # Base DataFrame
    df = pd.DataFrame({
        "ROI": roi,
        "DAPI_ID": valid_labels,
        "Area_pixels": counts,
        "Area_um2": counts * pixel_area_um2,
        "centroid_y": [cent_map.get(l, (np.nan,))[0] for l in valid_labels],
        "centroid_x": [cent_map.get(l, (np.nan,))[1] for l in valid_labels],
    })

    # Intensities and flags for each channel
    for ch in tqdm(channels, desc=f"Channels {roi}", unit="ch"):
        name = marker_dict.get(ch, f"Ch{ch}")
        col_base = name.replace(" ", "_").replace("-", "_").replace(">", "").replace("<", "")
        img_ch = img_data[ch]

        if "ngfr" in name.lower():
            df[f"mean_intensity_{col_base}"] = compute_mean_intensities(flat, img_ch, valid_labels, counts)
            mask_ng = ngfr_masks.get(roi, np.zeros_like(mask_dapi))
            df[f"is_positive_{col_base}"] = compute_binary_flags(valid_labels, flat, mask_ng)

        elif "ck" in name.lower():
            mask_ck = ck_masks.get(roi, np.zeros_like(mask_dapi))
            df[f"is_positive_{col_base}"] = compute_binary_flags(valid_labels, flat, mask_ck)

        else:
            df[f"mean_intensity_{col_base}"] = compute_mean_intensities(flat, img_ch, valid_labels, counts)

    return df


def intensity_to_binary(df: pd.DataFrame, thresholds: Dict[str, float], exclude: Optional[list] = None) -> pd.DataFrame:
    """
    Converts intensities to binary values based on provided thresholds.

    Args:
        df (pd.DataFrame): The DataFrame containing intensity values.
        thresholds (Dict[str, float]): A dictionary of thresholds to convert intensities to binary values.
        exclude (Optional[list], optional): List of columns to exclude from binary conversion. Defaults to columns like ROI, DAPI_ID, etc.

    Returns:
        pd.DataFrame: A DataFrame with binary intensity columns.
    """
    exclude = exclude or ['ROI', 'DAPI_ID', 'Area_pixels', 'Area_um2', 'centroid_x', 'centroid_y']
    markers = [c for c in df if c not in exclude]
    means = df.groupby('ROI')[markers].mean()
    stds = df.groupby('ROI')[markers].std()
    dfb = df.join(means, on='ROI', rsuffix='_mean') \
            .join(stds, on='ROI', rsuffix='_std')

    for m in markers:
        note = thresholds.get(m, 0.0)
        dfb[f"{m}_threshold"] = dfb[f"{m}_mean"] + note * dfb[f"{m}_std"]
        dfb[f"{m}_binary"] = (dfb[m] > dfb[f"{m}_threshold"]).astype(int)

    cols = ['ROI', 'DAPI_ID', 'Area_pixels', 'Area_um2', 'centroid_x', 'centroid_y'] \
         + [f"{m}_binary" for m in markers]
    return dfb[cols]
