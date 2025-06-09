import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from multiplex_pipeline.config import PIXEL_SIZE, PIXEL_AREA

def compute_mask_area_summary(
    ck_masks: dict, 
    ngfr_masks: dict, 
    pixel_area: float = PIXEL_AREA
) -> pd.DataFrame:
    """
    Computes the area summary for each ROI based on CK and NGFR masks.

    Args:
        ck_masks (dict): Dictionary of CK masks for each ROI.
        ngfr_masks (dict): Dictionary of NGFR masks for each ROI.
        pixel_area (float, optional): Area of each pixel in µm². Defaults to PIXEL_AREA.

    Returns:
        pd.DataFrame: DataFrame with columns ['ROI', 'CK_Positive_Area_um2', 
        'CK_NGFR_Positive_Area_um2', 'total_area_roi_um2'] representing area data for each ROI.
    """
    data = []
    for roi in ck_masks:
        ck_mask = ck_masks.get(roi)
        ngfr_mask = ngfr_masks.get(roi)
        
        # Skip if masks are not present or have mismatched shapes
        if ck_mask is None or ngfr_mask is None or ck_mask.shape != ngfr_mask.shape:
            continue
        
        # Calculate areas
        area_ck = np.sum(ck_mask == 1) * pixel_area
        area_ckng = np.sum((ck_mask == 1) & (ngfr_mask == 1)) * pixel_area
        total = ck_mask.size * pixel_area
        
        data.append({
            'ROI': roi,
            'CK_Positive_Area_um2': area_ck,
            'CK_NGFR_Positive_Area_um2': area_ckng,
            'total_area_roi_um2': total
        })
    
    return pd.DataFrame(data)


def compute_subpop_cells_per_area(
    df_binary: pd.DataFrame, 
    subpop_conditions: list, 
    cond_map: dict, 
    mask_summary: pd.DataFrame, 
    rois: list, 
    out_dir: str, 
    roi_col: str = 'ROI'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes subpopulation cell counts and densities for each ROI.

    Args:
        df_binary (pd.DataFrame): Binary DataFrame where rows represent cells and columns represent markers.
        subpop_conditions (list): List of conditions (e.g., markers) to define the subpopulation.
        cond_map (dict): Mapping of condition names to corresponding column names in df_binary.
        mask_summary (pd.DataFrame): DataFrame with area information for each ROI.
        rois (list): List of ROIs to process.
        out_dir (str): Directory to save the output CSV file.
        roi_col (str, optional): Column name in df_binary indicating the ROI. Defaults to 'ROI'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - summary_df: DataFrame with cell count and densities per ROI.
            - summary_fmt: Formatted version of summary_df for saving as CSV.
    """
    parsed = {}
    
    # Parse conditions into binary column values
    for cond in subpop_conditions:
        val = 1 if cond.strip().endswith('+') else 0
        key = cond.strip()[:-1].strip()
        col = cond_map.get(key)
        
        if col:
            parsed[col] = val
        else:
            print(f"Warning: No mapping found for '{cond}' → skipping.")

    # Lookup for areas by ROI
    area_lookup = mask_summary.set_index('ROI')

    # Filter df_binary by ROIs
    df_filt = df_binary[df_binary[roi_col].isin(rois)]
    missing = set(rois) - set(df_filt[roi_col].unique())
    
    if missing:
        print(f"Warning: ROIs not found in df_binary: {missing}")

    results = []
    for roi in rois:
        grp = df_filt[df_filt[roi_col] == roi]
        
        # Skip if group is empty or ROI is not in area lookup
        if grp.empty or roi not in area_lookup.index:
            continue

        # Filter by subpopulation conditions
        mask_sub = pd.Series(True, index=grp.index)
        for col, val in parsed.items():
            mask_sub &= (grp[col] == val)
        
        cnt = mask_sub.sum()

        # Get areas from the area lookup
        area_ck = area_lookup.at[roi, 'CK_Positive_Area_um2']
        area_ckng = area_lookup.at[roi, 'CK_NGFR_Positive_Area_um2']

        # Calculate densities
        dens_ck = cnt / area_ck if area_ck > 0 else 0
        dens_ckng = cnt / area_ckng if area_ckng > 0 else 0

        results.append({
            'ROI': roi,
            'Subpopulation_Cell_Count': cnt,
            'CK_Positive_Area_um2': area_ck,
            'CK_NGFR_Positive_Area_um2': area_ckng,
            'Cells_per_um2_CK_Positive': dens_ck,
            'Cells_per_um2_CK_NGFR_Positive': dens_ckng
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    # Format for saving
    summary_fmt = summary_df.copy()
    summary_fmt['Cells_per_um2_CK_Positive'] = summary_fmt['Cells_per_um2_CK_Positive'].map(lambda x: f"{x:.6f}")
    summary_fmt['Cells_per_um2_CK_NGFR_Positive'] = summary_fmt['Cells_per_um2_CK_NGFR_Positive'].map(lambda x: f"{x:.6f}")
    summary_fmt['CK_Positive_Area_um2'] = summary_fmt['CK_Positive_Area_um2'].map(lambda x: f"{x:.2f}")
    summary_fmt['CK_NGFR_Positive_Area_um2'] = summary_fmt['CK_NGFR_Positive_Area_um2'].map(lambda x: f"{x:.2f}")

    # Rename columns dynamically based on conditions
    label = "_".join([c.replace("+", "Pos").replace(" ", "") for c in subpop_conditions])
    summary_fmt.rename(columns={
        'Subpopulation_Cell_Count': f"Subpopulation Cell Count ({', '.join(subpop_conditions)})",
        'CK_Positive_Area_um2': 'CK⁺ Area (µm²)',
        'CK_NGFR_Positive_Area_um2': 'CK⁺NGFR⁺ Area (µm²)',
        'Cells_per_um2_CK_Positive': 'Cells per µm² (CK⁺)',
        'Cells_per_um2_CK_NGFR_Positive': 'Cells per µm² (CK⁺NGFR⁺)'
    }, inplace=True)

    # Save the formatted DataFrame to CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"cell_density_area_{label}.csv")
    summary_fmt.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return summary_df, summary_fmt


def compute_distances(sub_df: pd.DataFrame, mask: np.ndarray, bin_col: str) -> tuple[list, list]:
    """
    Computes the distances from each cell's centroid to the positive and negative regions of a mask.

    Args:
        sub_df (pd.DataFrame): DataFrame with cell centroids and binary values.
        mask (np.ndarray): Binary mask where 1 represents the positive region and 0 represents the negative region.
        bin_col (str): Column name in sub_df representing the binary condition.

    Returns:
        tuple[list, list]: Two lists containing distances to positive and negative regions in micrometers.
    """
    pos = np.column_stack(np.where(mask == 1))
    neg = np.column_stack(np.where(mask == 0))
    
    tree_p = cKDTree(pos) if len(pos) else None
    tree_n = cKDTree(neg) if len(neg) else None
    
    d_to_pos, d_to_neg = [], []
    for _, r in sub_df.iterrows():
        cent = (r['centroid_row'], r['centroid_col'])
        if r.get(bin_col, 0) == 1:
            d_to_pos.append(0.0)
            d_to_neg.append(tree_n.query(cent)[0] * PIXEL_SIZE if tree_n else 0.0)
        else:
            d_to_pos.append(tree_p.query(cent)[0] * PIXEL_SIZE if tree_p else 0.0)
            d_to_neg.append(0.0)
    
    return d_to_pos, d_to_neg


def get_centroids(dapi_mask: np.ndarray) -> pd.DataFrame:
    """
    Extracts centroids from a labeled DAPI mask.

    Args:
        dapi_mask (np.ndarray): Labeled DAPI mask.

    Returns:
        pd.DataFrame: DataFrame with columns ['DAPI_ID', 'centroid_row', 'centroid_col'].
    """
    data = []
    for p in regionprops(dapi_mask):
        r, c = p.centroid
        data.append({'DAPI_ID': p.label, 'centroid_row': int(r), 'centroid_col': int(c)})
    return pd.DataFrame(data)


def compute_subpop_distances(subpopA_df: pd.DataFrame, subpopB_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pairwise distances (pixels) between centroids of two subpopulations.

    Args:
        subpopA_df (pd.DataFrame): DataFrame with centroids of subpopulation A.
        subpopB_df (pd.DataFrame): DataFrame with centroids of subpopulation B.

    Returns:
        pd.DataFrame: DataFrame with pairwise distances between centroids.
    """
    if subpopA_df.empty or subpopB_df.empty:
        return pd.DataFrame()

    A = subpopA_df.assign(key=1)
    B = subpopB_df.assign(key=1)
    df = pd.merge(B, A, on='key', suffixes=('_b', '_a')).drop('key', axis=1)
    df['distance_px'] = np.sqrt((df['centroid_row_b'] - df['centroid_row_a'])**2 + 
                                (df['centroid_col_b'] - df['centroid_col_a'])**2)
    
    return df.rename(columns={
        'DAPI_ID_b': 'B_cell_id', 'centroid_row_b': 'B_row', 'centroid_col_b': 'B_col',
        'DAPI_ID_a': 'A_cell_id', 'centroid_row_a': 'A_row', 'centroid_col_a': 'A_col'
    })
