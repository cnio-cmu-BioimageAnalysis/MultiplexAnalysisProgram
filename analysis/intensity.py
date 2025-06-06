import re
import numpy as np
import pandas as pd
from skimage import measure
from multiplex_pipeline.utils.validation import verify_binary

def process_roi(img_name, 
               img_data, 
               dapi_masks_dict, 
               ck_masks_dict, 
               ngfr_masks_dict, 
               channels, 
               marker_dict,
               pixel_area_um2):
    """
    Processes each ROI to generate a DataFrame with mean intensities,
    binary values, area of each cell in micrometers squared,
    and centroid coordinates in the image.
    """
    roi_match = re.search(r"(ROI\d+)", img_name, re.IGNORECASE)
    if not roi_match:
        print(f"ROI not found in image name: {img_name}")
        return None
    roi_key = roi_match.group(1).lower()  # Por ejemplo: "roi1"

    dapi_key = roi_key + "_dapi"
    ck_key   = roi_key
    ngfr_key = roi_key
    
    if dapi_key not in dapi_masks_dict:
        print(f"{dapi_key} not found in dapi_masks_dict.")
        return None

    mask_dapi = dapi_masks_dict[dapi_key]
    if mask_dapi.shape != img_data.shape[1:]:
        print(f"Dimensions do not match for {img_name}: {mask_dapi.shape} vs {img_data.shape[1:]}")
        return None

    # Si la máscara DAPI está binaria, la etiquetamos
    if len(np.unique(mask_dapi)) <= 2:
        print(f"DAPI mask for {roi_key} is binary. Attempting to label.")
        labeled_mask = measure.label(mask_dapi, connectivity=1)
    else:
        labeled_mask = mask_dapi

    # Contamos cuántos píxeles hay de cada etiqueta
    mask_flat = labeled_mask.ravel()
    counts = np.bincount(mask_flat)
    labels = np.arange(len(counts))
    valid = (labels != 0) & (counts > 0)
    valid_labels = labels[valid]

    # === NUEVO: calculamos centroides de cada label con regionprops ===
    props = measure.regionprops(labeled_mask)
    label_centroids = {p.label: p.centroid for p in props}

    # Columnas base
    df = pd.DataFrame({
        "ROI": [roi_key] * len(valid_labels),
        "DAPI_ID": valid_labels,
        "Area_pixels": counts[valid],
        "Area_um2": counts[valid] * pixel_area_um2
    })

    # Agregar columnas de centroides (y=row, x=col)
    centroids_y = []
    centroids_x = []
    for lbl in valid_labels:
        if lbl in label_centroids:
            cy, cx = label_centroids[lbl]
        else:
            cy, cx = (np.nan, np.nan)
        centroids_y.append(cy)
        centroids_x.append(cx)

    df["centroid_y"] = centroids_y
    df["centroid_x"] = centroids_x

    # --- Iterar a través de los canales de interés ---
    for ch in channels:
        marker = marker_dict.get(ch, f"Channel_{ch}")
        marker_lower = marker.lower()

        # Limpiamos el nombre del marcador para usarlo como columna
        formatted_name = marker.replace(' ', '_').replace('-', '_').replace('>', '').replace('<', '')

        # CASO A) NGFR: intensidad media + binario
        if "ngfr" in marker_lower:
            intensity_flat = img_data[ch].ravel()
            sum_intensities = np.bincount(mask_flat, weights=intensity_flat)
            means = sum_intensities[valid] / counts[valid]
            df[f"mean_intensity_{formatted_name}"] = means

            if ngfr_key in ngfr_masks_dict:
                mask_ngfr = ngfr_masks_dict[ngfr_key]
                verify_binary(mask_ngfr, f"NGFR mask for {roi_key}")

                if mask_ngfr.shape != mask_dapi.shape:
                    print(f"Dimensions of NGFR mask {ngfr_key} do not match DAPI mask: {mask_ngfr.shape} vs {mask_dapi.shape}")
                    positive_flags = np.zeros(len(valid_labels), dtype=int)
                else:
                    mask_ngfr_flat = mask_ngfr.ravel()
                    etiquetas_ngfr = mask_flat[mask_ngfr_flat > 0]
                    positive_labels = np.unique(etiquetas_ngfr)
                    positive_flags = np.isin(valid_labels, positive_labels).astype(int)
            else:
                print(f"NGFR mask not found for {roi_key}")
                positive_flags = np.zeros(len(valid_labels), dtype=int)
            
            df[f"is_positive_{formatted_name}"] = positive_flags

        # CASO B) CK: solo máscara binaria
        elif "ck" in marker_lower:
            if ck_key in ck_masks_dict:
                mask_ck = ck_masks_dict[ck_key]
                verify_binary(mask_ck, f"CK mask for {roi_key}")

                if mask_ck.shape != mask_dapi.shape:
                    print(f"Dimensions of CK mask {ck_key} do not match DAPI mask: {mask_ck.shape} vs {mask_dapi.shape}")
                    positive_flags = np.zeros(len(valid_labels), dtype=int)
                else:
                    mask_ck_flat = mask_ck.ravel()
                    etiquetas_ck = mask_flat[mask_ck_flat > 0]
                    positive_labels = np.unique(etiquetas_ck)
                    positive_flags = np.isin(valid_labels, positive_labels).astype(int)
            else:
                print(f"CK mask not found for {roi_key}")
                positive_flags = np.zeros(len(valid_labels), dtype=int)
            
            df[f"is_positive_{formatted_name}"] = positive_flags

        # CASO C) Otros canales: solo intensidad media
        else:
            intensity_flat = img_data[ch].ravel()
            sum_intensities = np.bincount(mask_flat, weights=intensity_flat)
            means = sum_intensities[valid] / counts[valid]
            df[f"mean_intensity_{formatted_name}"] = means

    return df


# multiplex_pipeline/analysis/intensity.py

import pandas as pd
from typing import Dict, List

def intensity_to_binary(
    df_results: pd.DataFrame,
    marker_notes: Dict[str, float],
    exclude_cols: List[str] = None
) -> pd.DataFrame:
    """
    Convierte el DataFrame de intensidades en un DataFrame binario:
    - Para cada marcador calcula umbral = mean_ROI + note * std_ROI
    - Crea columna `{marker}_binary` con 1 si valor > umbral, 0 en caso contrario
    - Devuelve solo [ROI, DAPI_ID, Area_pixels, Area_um2, centroid_x, centroid_y] + {marker}_binary
    """
    df = df_results.copy()
    if exclude_cols is None:
        exclude_cols = ['ROI','DAPI_ID','Area_pixels','Area_um2','centroid_x','centroid_y']
    
    # 1) Identificar columnas de marcador
    marker_cols = [c for c in df.columns if c not in exclude_cols]
    # 2) Media y std por ROI
    means = df.groupby('ROI')[marker_cols].mean().reset_index()
    stds  = df.groupby('ROI')[marker_cols].std().reset_index()
    # 3) Merge
    df_merged = df.merge(means, on='ROI', suffixes=('','_roi_mean'))
    df_merged = df_merged.merge(stds,  on='ROI', suffixes=('','_roi_std'))
    
    # 4) Umbrales y binarización
    for m in marker_cols:
        note = marker_notes.get(m, 0.0)
        df_merged[f'{m}_threshold'] = (
            df_merged[f'{m}_roi_mean'] + note * df_merged[f'{m}_roi_std']
        )
        df_merged[f'{m}_binary'] = (
            (df_merged[m] > df_merged[f'{m}_threshold'])
            .astype(int)
        )
    
    # 5) Seleccionar columnas finales
    binary_cols = [f'{m}_binary' for m in marker_cols]
    return df_merged[
        ['ROI','DAPI_ID','Area_pixels','Area_um2','centroid_x','centroid_y']
        + binary_cols
    ]
