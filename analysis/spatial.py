import os
import numpy as np
import pandas as pd
from multiplex_pipeline.config import PIXEL_SIZE, PIXEL_AREA

from scipy.spatial import cKDTree
from skimage.measure import regionprops


def compute_mask_area_summary(ck_masks, ngfr_masks, pixel_area=PIXEL_AREA):
    """
    Computes the area summary for each ROI based on CK and NGFR masks.
    """
    data = []
    
    for roi in ck_masks:
        ck_mask = ck_masks.get(roi, None)
        ngfr_mask = ngfr_masks.get(roi, None)
        
        if ck_mask is None or ngfr_mask is None:
            print(f"CK or NGFR mask for ROI '{roi}' not found. Skipping.")
            continue
        
        if ck_mask.shape != ngfr_mask.shape:
            print(f"Mask shapes do not match for ROI '{roi}'. Skipping.")
            continue
        
        # Calculate areas
        area_ck_pos = np.sum(ck_mask == 1) * pixel_area
        area_ck_ngfr_pos = np.sum((ck_mask == 1) & (ngfr_mask == 1)) * pixel_area
        total_area = ck_mask.size * pixel_area  # Total area in µm²
        
        data.append({
            'ROI': roi,
            'CK_Positive_Area_um2': area_ck_pos,
            'CK_NGFR_Positive_Area_um2': area_ck_ngfr_pos,
            'total_area_roi_um2': total_area
        })
    
    mask_area_summary = pd.DataFrame(data)
    return mask_area_summary


def compute_subpop_cells_per_area(
    df_binary,
    subpop_conditions,
    condition_column_map,
    mask_area_summary,
    selected_rois,
    path_save,
    roi_col='ROI'
):
    """
    For each specified ROI:
      1) Filters the cells that meet ALL the 'subpop_conditions'.
      2) Counts the total number of cells in this subpopulation.
      3) Includes the CK⁺ and CK⁺NGFR⁺ areas from mask_area_summary.
      4) Calculates the concentration of cells per square micrometer (cells/µm²) 
         for CK⁺ and CK⁺NGFR⁺ areas.
      5) Saves the formatted summary as a CSV file in the specified path.
    
    Additionally:
      - Renames the 'Subpopulation Cell Count' column to reflect the actual subpopulation conditions.
    
    Parameters:
    - df_binary (pd.DataFrame): DataFrame containing cell information.
    - subpop_conditions (list of str): List of conditions, e.g., ["CD3_intensity+", "CD4_intensity+", "FOXP3_intensity+"].
    - condition_column_map (dict): Mapping of condition names to column names in df_binary.
    - mask_area_summary (pd.DataFrame): DataFrame containing area information per ROI.
    - selected_rois (list of str): List of ROIs to process, e.g., ["roi8","roi13","roi11","roi5"].
    - path_save (str): Directory path where the CSV will be saved, e.g., "..\\results_spatial_analysis\\user\\cell_density_area".
    - roi_col (str): Name of the column in df_binary that indicates the ROI.
    
    Returns:
    - summary_df (pd.DataFrame): Summary DataFrame with metrics per ROI.
    - summary_formatted (pd.DataFrame): Formatted DataFrame for better readability.
    """
    # 1. Parse the subpopulation conditions into a dictionary {column: value}
    parsed_conditions = {}
    for cond in subpop_conditions:
        cond = cond.strip()
        val = 1 if cond.endswith('+') else 0
        base = cond[:-1].strip()
        if base in condition_column_map:
            col_name = condition_column_map[base]
            parsed_conditions[col_name] = val
        else:
            print(f"Warning: No mapping for '{cond}' → ignoring.")
    
    # 2. Create a lookup dictionary for areas per ROI
    area_lookup = mask_area_summary.set_index('ROI')
    
    # 3. Filter df_binary to include only selected_rois
    df_filtered = df_binary[df_binary[roi_col].isin(selected_rois)]
    
    # Check for ROIs in selected_rois not present in df_binary
    missing_in_binary = set(selected_rois) - set(df_filtered[roi_col].unique())
    if missing_in_binary:
        print(f"Warning: The following ROIs are specified in selected_rois but not found in df_binary: {missing_in_binary}")
    
    # 4. Group the filtered df_binary by ROI
    grouped = df_filtered.groupby(roi_col, dropna=False)
    
    results = []
    
    for roi, group_df in grouped:
        # a. Check if the ROI exists in mask_area_summary
        if roi not in area_lookup.index:
            print(f"Warning: ROI '{roi}' not found in mask_area_summary. Skipping.")
            continue
        
        # b. Filter the subpopulation that meets ALL conditions
        mask_subpop = pd.Series([True] * len(group_df), index=group_df.index)
        for bin_col, needed_val in parsed_conditions.items():
            if bin_col in group_df.columns:
                mask_subpop &= (group_df[bin_col] == needed_val)
            else:
                print(f"Warning: Column '{bin_col}' not found in df_binary for ROI '{roi}'.")
                mask_subpop &= False  # If the column is not present, no match
        
        subpop_df = group_df[mask_subpop]
        subpop_count = len(subpop_df)
        
        # c. Retrieve areas from mask_area_summary
        area_ck = area_lookup.loc[roi, 'CK_Positive_Area_um2'] if 'CK_Positive_Area_um2' in area_lookup.columns else 0
        area_ck_ngfr = area_lookup.loc[roi, 'CK_NGFR_Positive_Area_um2'] if 'CK_NGFR_Positive_Area_um2' in area_lookup.columns else 0
        
        # d. Calculate cell concentration per square micrometer for each specific area
        cells_per_um2_ck = subpop_count / area_ck if area_ck > 0 else 0
        cells_per_um2_ck_ngfr = subpop_count / area_ck_ngfr if area_ck_ngfr > 0 else 0
        
        # e. Add the results
        results.append({
            'ROI': roi,
            'Subpopulation_Cell_Count': subpop_count,
            'CK_Positive_Area_um2': area_ck,
            'CK_NGFR_Positive_Area_um2': area_ck_ngfr,
            'Cells_per_um2_CK_Positive': cells_per_um2_ck,        # Cells per µm² of CK⁺
            'Cells_per_um2_CK_NGFR_Positive': cells_per_um2_ck_ngfr  # Cells per µm² of CK⁺NGFR⁺
        })
        
        # f. Debugging information
        print(f"ROI: {roi}")
        print(f"Subpopulation Count: {subpop_count}")
        print(f"CK⁺ Area (µm²): {area_ck}")
        print(f"CK⁺NGFR⁺ Area (µm²): {area_ck_ngfr}")
        print(f"Cells per µm² (CK⁺): {cells_per_um2_ck:.6f}")
        print(f"Cells per µm² (CK⁺NGFR⁺): {cells_per_um2_ck_ngfr:.6f}")
        print("-" * 50)
    
    # 5. Check for ROIs specified in selected_rois but missing in mask_area_summary
    missing_in_mask = set(selected_rois) - set(area_lookup.index)
    if missing_in_mask:
        print(f"Warning: The following ROIs are specified in selected_rois but not found in mask_area_summary: {missing_in_mask}")
    
    # 6. Create the summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # 7. Format the DataFrame for better readability
    summary_formatted = summary_df.copy()
    
    # Maintain precision in 'Cells_per_um2' without rounding
    summary_formatted['Cells_per_um2_CK_Positive'] = summary_formatted['Cells_per_um2_CK_Positive'].apply(lambda x: f"{x:.6f}")
    summary_formatted['Cells_per_um2_CK_NGFR_Positive'] = summary_formatted['Cells_per_um2_CK_NGFR_Positive'].apply(lambda x: f"{x:.6f}")
    
    # Format areas to two decimal places
    summary_formatted['CK_Positive_Area_um2'] = summary_formatted['CK_Positive_Area_um2'].apply(lambda x: f"{x:.2f}")
    summary_formatted['CK_NGFR_Positive_Area_um2'] = summary_formatted['CK_NGFR_Positive_Area_um2'].apply(lambda x: f"{x:.2f}")
    
    # 8. Generate dynamic label for Subpopulation
    # Join the subpop_conditions with underscores and replace '+' with 'Pos' to make it filename-friendly
    safe_subpop_label = "_".join([cond.replace("+", "Pos").replace(" ", "") for cond in subpop_conditions])
    
    # 9. Rename columns for clarity, including dynamic subpopulation label
    summary_formatted.rename(columns={
        'Subpopulation_Cell_Count': f'Subpopulation Cell Count ({", ".join(subpop_conditions)})',
        'CK_Positive_Area_um2': 'CK⁺ Area (µm²)',
        'CK_NGFR_Positive_Area_um2': 'CK⁺NGFR⁺ Area (µm²)',
        'Cells_per_um2_CK_Positive': 'Cells per µm² (CK⁺)',
        'Cells_per_um2_CK_NGFR_Positive': 'Cells per µm² (CK⁺NGFR⁺)'
    }, inplace=True)
    
    # 10. Ensure the save directory exists
    os.makedirs(path_save, exist_ok=True)
    
    # 11. Construct an appropriate CSV filename without datetime
    # For example: 'cell_density_area_CD3_intensityPos_CD4_intensityPos_FOXP3_intensityPos.csv'
    csv_filename = f"cell_density_area_{safe_subpop_label}.csv"
    csv_path = os.path.join(path_save, csv_filename)
    
    # 12. Save the formatted summary as CSV
    summary_formatted.to_csv(csv_path, index=False)
    print(f"\nFormatted summary saved as CSV at: {csv_path}")
    
    # 13. Display the formatted DataFrame
    print("\nFormatted Summary DataFrame:")
    print(summary_formatted)
    
    return summary_df, summary_formatted  # Return both for flexibility
def compute_distances(sub_df, mask, bin_col):
    """
    Returns two lists:
        dist_to_pos, dist_to_neg   (µm)
    """
    pos_pts = np.column_stack(np.where(mask == 1))
    neg_pts = np.column_stack(np.where(mask == 0))
    tree_pos = cKDTree(pos_pts) if len(pos_pts) else None
    tree_neg = cKDTree(neg_pts) if len(neg_pts) else None

    d_pos, d_neg = [], []
    for _, r in sub_df.iterrows():
        centroid = (r['centroid_row'], r['centroid_col'])
        if r[bin_col] == 1:  # marker+
            dist = tree_neg.query(centroid)[0] if tree_neg else 0.0
            d_pos.append(0.0)
            d_neg.append(dist * PIXEL_SIZE)
        else:                # marker‑
            dist = tree_pos.query(centroid)[0] if tree_pos else 0.0
            d_pos.append(dist * PIXEL_SIZE)
            d_neg.append(0.0)
    return d_pos, d_neg
def get_centroids(dapi_mask):
    """
    Extracts centroids (row, col) from the labeled DAPI mask.
    Returns a DataFrame with DAPI_ID, centroid_row, centroid_col.
    """
    props = regionprops(dapi_mask)
    data = []
    for prop in props:
        data.append({
            'DAPI_ID': prop.label,
            'centroid_row': int(prop.centroid[0]),
            'centroid_col': int(prop.centroid[1])
        })
    return pd.DataFrame(data)




def compute_subpop_distances(subpopA_df, subpopB_df):
    """
    For each cell in A and each cell in B, creates all A–B pairs and computes
    the distance in pixels. Returns a DataFrame with all pairwise distances (no NaN).
    """
    if len(subpopA_df) == 0 or len(subpopB_df) == 0:
        return pd.DataFrame()  # Nothing to compute

    # Cartesian product: B x A
    cartesian_df = subpopB_df.assign(key=1).merge(
        subpopA_df.assign(key=1),
        on='key',
        suffixes=('_b', '_a')
    ).drop('key', axis=1)

    # Compute distance for each pair
    cartesian_df['distance_px'] = np.sqrt(
        (cartesian_df['centroid_row_b'] - cartesian_df['centroid_row_a'])**2 +
        (cartesian_df['centroid_col_b'] - cartesian_df['centroid_col_a'])**2
    )

    # Rename columns for consistency
    result = cartesian_df[[
        'DAPI_ID_b', 'centroid_row_b', 'centroid_col_b',
        'DAPI_ID_a', 'centroid_row_a', 'centroid_col_a', 'distance_px'
    ]]
    result.columns = [
        'B_cell_id', 'B_row', 'B_col',
        'A_cell_id', 'A_row', 'A_col', 'distance_px'
    ]

    return result
