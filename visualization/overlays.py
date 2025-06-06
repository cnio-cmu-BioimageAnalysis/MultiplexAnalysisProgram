
import random

from scipy.spatial import cKDTree, KDTree, Voronoi
from skimage.measure import regionprops, find_contours
from skimage.color import label2rgb

from shapely.geometry import Polygon as ShapelyPolygon

from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon, Patch
import matplotlib.patches as patches

from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines

# Constantes que usas pero no defines aquí:
from multiplex_pipeline.config import PIXEL_SIZE, MASK_ALPHA

# Funciones auxiliares que también debes importar si no están en este mismo módulo:
from multiplex_pipeline.analysis.spatial import (
    compute_distances,
    get_centroids,
    compute_subpop_distances
)

from multiplex_pipeline.config import CNIO_USER
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def parse_conditions(cond_list, col_map):
    """
    Converts strings like 'CK_mask+' into
    {'is_positive_Pan_Cytokeratin_CK_binary': 1}
    """
    parsed = {}
    for c in (cond.strip() for cond in cond_list):
        if c.endswith('+'):
            base, val = c[:-1].strip(), 1
        elif c.endswith('-'):
            base, val = c[:-1].strip(), 0
        else:
            print(f"Ignoring invalid condition '{c}'.")
            continue
        if base in col_map:
            parsed[col_map[base]] = val
        else:
            print(f"No mapping for '{c}'.")
    return parsed

def select_subpopulation(df, parsed):
    """AND‑filter dataframe with parsed condition dict."""
    if not parsed:
        return df.iloc[0:0]
    mask = pd.Series(True, index=df.index)
    for col, val in parsed.items():
        mask &= (df[col] == val) if col in df else False
    return df[mask]

def create_color_composite(img_data, mask_dicts, markers_to_plot, roi_lower, brightness_factor=1.0):
    """
    Crea una composición aditiva multicanal, asignando colores distintos a cada canal y máscara.
    
    Parámetros:
    - img_data: array de numpy con forma (num_channels, height, width)
    - mask_dicts: diccionario que contiene las máscaras disponibles (ej. {'CK_mask': ck_masks_dict, ...})
    - markers_to_plot: lista de tuplas (nombre_marcador, índice_canal o None)
    - roi_lower: string, nombre de la ROI en minúsculas
    - brightness_factor: factor de ajuste de brillo
    
    Retorna:
    - composite: array de numpy con forma (height, width, 3) representando la mezcla aditiva
    """
    height, width = img_data.shape[1], img_data.shape[2]
    composite = np.zeros((height, width, 3), dtype=np.float32)

    # Lista de colores base (Rojo, Verde, Azul, Magenta, Cian, Amarillo, etc.)
    base_colors = np.array([
        [1, 0, 0],      # Rojo
        [0, 1, 0],      # Verde
        [0, 0, 1],      # Azul
        [1, 0, 1],      # Magenta
        [0, 1, 1],      # Cian
        [1, 1, 0],      # Amarillo
        [0.5, 0.5, 0],  # Oliva
        [0.5, 0, 0.5],  # Púrpura
        [0, 0.5, 0.5],  # Teal
    ], dtype=np.float32)

    for i, (marker_name, ch) in enumerate(markers_to_plot):
        color = base_colors[i % len(base_colors)]  # Asigna color único

        if ch is not None:
            # Es un canal de intensidad
            channel_data = img_data[ch, :, :] * brightness_factor
            cmin, cmax = channel_data.min(), channel_data.max()
            if cmax == cmin:
                channel_norm = np.zeros_like(channel_data, dtype=np.float32)
            else:
                channel_norm = (channel_data - cmin) / (cmax - cmin)
            gamma = 0.5
            channel_gamma_corrected = channel_norm ** gamma

            # Suma aditiva al composite en cada canal RGB
            composite[:, :, 0] += channel_gamma_corrected * color[0]
            composite[:, :, 1] += channel_gamma_corrected * color[1]
            composite[:, :, 2] += channel_gamma_corrected * color[2]
        else:
            # Es una máscara
            mask_data = None
            if marker_name in mask_dicts and roi_lower in mask_dicts[marker_name]:
                mask_data = mask_dicts[marker_name][roi_lower]
            if mask_data is not None:
                # Asegurarse de que la máscara es binaria
                mask_binary = (mask_data > 0).astype(float)
                composite[:, :, 0] += mask_binary * color[0]
                composite[:, :, 1] += mask_binary * color[1]
                composite[:, :, 2] += mask_binary * color[2]
            else:
               
                pass

    # Limitar valores para evitar que excedan 1
    composite = np.clip(composite, 0, 1)
    return composite

def plot_conditional_cells_channels(
    rois, 
    conditions, 
    dapi_masks_dict, 
    images_dict, 
    df_binary, 
    marker_dict, 
    ck_masks_dict, 
    ngfr_masks_dict, 
    condition_column_map, 
    brightness_factor=1.0
    ):
    def is_mask_marker(name):
        return name.lower().endswith('_mask')

    def is_intensity_marker(name):
        return name.lower().endswith('_intensity')

    def get_channel_for_marker(marker_short):
        """Devuelve el índice de canal si es *_intensity; None si es *_mask."""
        if is_mask_marker(marker_short):
            return None
        base_name = marker_short.replace('_intensity', '').strip()
        for ch, full_name in marker_dict.items():
            if base_name.lower() == full_name.lower():
                print(f"Marker '{base_name}' matched with channel {ch}")
                return ch
        print(f"Channel not found for marker '{marker_short}'")
        return None

    def parse_conditions(conditions, condition_column_map):
        """Returns un dict con { binary_column: value (1 o 0), ... }"""
        parsed = {}
        for cond in conditions:
            if cond.endswith('+'):
                shorthand = cond[:-1]
                val = 1
            elif cond.endswith('-'):
                shorthand = cond[:-1]
                val = 0
            else:
                print(f"Formato de condición desconocido: {cond}. Debe terminar en '+' o '-'.")
                continue

            if shorthand in condition_column_map:
                marker_col = condition_column_map[shorthand]
                parsed[marker_col] = val
            else:
                print(f"No se encontró mapeo para la condición {shorthand}")
        return parsed

    parsed_conditions = parse_conditions(conditions, condition_column_map)

    # Preparamos la lista de marcadores (tuplas) a mostrar
    markers_to_plot = []
    mask_dicts_combined = {
        'CK_mask': ck_masks_dict,
        'NGFR_mask': ngfr_masks_dict,
    }
    for cond in conditions:
        # Por ej. cond = 'CK_mask+', 'CD3_intensity+', etc.
        shorthand = cond[:-1]  # quita el + o -
        ch = get_channel_for_marker(shorthand)
        markers_to_plot.append((shorthand, ch))

    for roi in rois:
        print(f"Processing {roi}...")
        roi_lower = roi.lower()
        roi_dapi_key = f"{roi_lower}_dapi"

        if roi_dapi_key not in dapi_masks_dict:
            print(f"Warning: {roi_dapi_key} no se encuentra en dapi_masks_dict. Saltando...")
            continue

        # Buscar la clave en images_dict que contenga el ROI
        roi_image_key = None
        for key in images_dict.keys():
            if roi_lower in key.lower():
                roi_image_key = key
                break
        if roi_image_key is None:
            print(f"No se encontró imagen en images_dict para {roi}")
            continue

        img_data = images_dict[roi_image_key]  # shape (channels, height, width)
        cell_mask = dapi_masks_dict[roi_dapi_key]
        df_roi = df_binary[df_binary['ROI'] == roi]

        # Determinar celdas que cumplen TODAS las condiciones
        condition_series = pd.Series([True]*len(df_roi), index=df_roi.index)
        for marker_col, val in parsed_conditions.items():
            condition_series &= (df_roi[marker_col] == val)

        selected_cells = df_roi[condition_series]['DAPI_ID'].tolist()
        print(f"ROI {roi}: {len(selected_cells)} celdas cumplen {conditions}")

        mask_selected = np.isin(cell_mask, selected_cells).astype(int) if selected_cells else np.zeros_like(cell_mask, dtype=int)

        n_markers = len(markers_to_plot)
        n_cols = n_markers + 2
        
        fig = plt.figure(figsize=(6*n_cols, 12))
        
        # Generar un colormap aleatorio para las células (igual que antes)
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]
        num_labels = len(cell_labels) + 1
        np.random.seed(42)
        rand_colors = np.random.rand(num_labels, 3)
        rand_colors[0] = [0, 0, 0] 
        cmap_cells = ListedColormap(rand_colors)
        
        # --------------------------------------
        # Listas donde guardaremos cada imagen de la fila
        top_images = [None] * n_cols
        bottom_images = [None] * n_cols
        # --------------------------------------
        
        # ==================== FILA SUPERIOR ====================
        # (1) Columna 1 (arriba): composite multicolor
        ax_top_left = plt.subplot(2, n_cols, 1)
        composite = create_color_composite(
            img_data, 
            mask_dicts_combined, 
            markers_to_plot, 
            roi_lower, 
            brightness_factor=brightness_factor
        )
        ax_top_left.imshow(composite, interpolation='nearest')
        ax_top_left.set_title('Multicolor Composite\n(Intensity + Masks)')
        ax_top_left.axis('off')
        top_images[0] = composite  # almacenamos la composite en la posición 0
        
        # (2) Columnas 2..(n_markers+1), fila superior: canales individuales
        for i, (marker_name, ch) in enumerate(markers_to_plot, start=2):
            ax_top = plt.subplot(2, n_cols, i)
            if is_mask_marker(marker_name):
                # Plot de la máscara
                if marker_name in mask_dicts_combined and roi_lower in mask_dicts_combined[marker_name]:
                    mask_img = mask_dicts_combined[marker_name][roi_lower]
                    ax_top.imshow(mask_img, cmap='gray')
                    top_images[i-1] = mask_img.astype(float)
                else:
                    zeros_img = np.zeros_like(cell_mask, dtype=float)
                    ax_top.imshow(zeros_img, cmap='gray')
                    top_images[i-1] = zeros_img
                ax_top.set_title(f'Mask: {marker_name}')
        
            elif is_intensity_marker(marker_name):
                # Plot del canal intensidad
                if ch is not None:
                    channel_data = img_data[ch, :, :] * brightness_factor
                    cmin, cmax = channel_data.min(), channel_data.max()
                    if cmax == cmin:
                        channel_norm = np.zeros_like(channel_data)
                    else:
                        channel_norm = (channel_data - cmin) / (cmax - cmin)
                    gamma = 0.5
                    channel_gamma_corrected = channel_norm ** gamma
                    ax_top.imshow(channel_gamma_corrected, cmap='gray')
                    top_images[i-1] = channel_gamma_corrected
                else:
                    zeros_img = np.zeros_like(cell_mask, dtype=float)
                    ax_top.imshow(zeros_img, cmap='gray')
                    top_images[i-1] = zeros_img
                ax_top.set_title(f'Intensity: {marker_name}')
            else:
                # Caso no manejado
                zeros_img = np.zeros_like(cell_mask, dtype=float)
                ax_top.imshow(zeros_img, cmap='gray')
                top_images[i-1] = zeros_img
                ax_top.set_title(f'?? {marker_name}')
        
            ax_top.axis('off')
        
        # (3) Columna n_markers+2, fila superior: AQUI HACEMOS LA MEDIA
        ax_top_right = plt.subplot(2, n_cols, n_markers + 2)
        # Calculamos la media de las imágenes de las columnas 2..(n_markers+1),
        # o sea, indices 1..n_markers en top_images (excluimos la 0, que es el composite).
        imgs_for_average_top = [top_images[idx] for idx in range(1, n_markers+1)]
        # Verificamos que sean 2D todas (en caso de que alguna fuera RGB, puede requerir adaptación).
        avg_top = np.mean(imgs_for_average_top, axis=0)
        
        ax_top_right.imshow(avg_top, cmap='gray')
        ax_top_right.set_title('Media (Fila Superior)')
        ax_top_right.axis('off')
        top_images[n_markers+1] = avg_top  # Guardamos la imagen final por si se usa después
        
        # ==================== FILA INFERIOR ====================
        # (4) Columna 1 (abajo): composite + TODAS las células
        ax_bottom_left = plt.subplot(2, n_cols, n_cols + 1)
        composite_2x = np.clip(top_images[0] * 2, 0, 1)  # duplicamos brillo de la composite
        ax_bottom_left.imshow(composite_2x, interpolation='nearest') 
        ax_bottom_left.imshow(cell_mask, cmap=cmap_cells, alpha=0.5, interpolation='nearest')
        ax_bottom_left.set_title('All Cells (sobre Composite)')
        ax_bottom_left.axis('off')
        bottom_images[0] = composite_2x
        
        # (5) Columnas 2..(n_markers+1) (abajo): el mismo canal/máscara, pero con brillo x2
        for j, (marker_name, _) in enumerate(markers_to_plot, start=2):
            ax_bottom = plt.subplot(2, n_cols, n_cols + j)
            fondo = top_images[j-1]  # la imagen que habíamos guardado arriba
            fondo_2x = np.clip(fondo * 2, 0, 1)  # duplicamos brillo
            bottom_images[j-1] = fondo_2x  # guardamos esta versión en la lista bottom_images
        
            ax_bottom.imshow(fondo_2x, cmap='gray')
            # Overlay: celdas positivas para el marcador
            col_bin = condition_column_map.get(marker_name)
            if col_bin is not None:
                # Aquí consultamos si hay un valor (0 o 1) definido para este marcador en parsed_conditions
                val = parsed_conditions.get(col_bin, None)
                if val is not None:
                    # Muestra celdas == val (si val = 1 → positivas, si val = 0 → negativas)
                    selected_cells_for_marker = df_roi[df_roi[col_bin] == val]['DAPI_ID'].tolist()
                else:
                    # Si el marcador no está en las condiciones, decide por defecto (por ej. siempre 1)
                    selected_cells_for_marker = df_roi[df_roi[col_bin] == 1]['DAPI_ID'].tolist()
            else:
                selected_cells_for_marker = []
            
            mask_positive = np.isin(cell_mask, selected_cells_for_marker).astype(int)
            ax_bottom.imshow(mask_positive * cell_mask, cmap=cmap_cells, interpolation='nearest', alpha=0.5)

        
            ax_bottom.imshow(mask_positive * cell_mask, cmap=cmap_cells, interpolation='nearest', alpha=0.5)
            ax_bottom.set_title(f'Pos Cells\n{marker_name}')
            ax_bottom.axis('off')
        
        # (6) Columna n_markers+2, fila inferior: AQUI TAMBIÉN HACEMOS LA MEDIA
        ax_bottom_right = plt.subplot(2, n_cols, 2*n_cols)
        # Calculamos la media de las columnas 2..(n_markers+1) en la fila inferior,
        # que son indices 1..n_markers en bottom_images (excluyendo el 0 que es composite_2x).
        imgs_for_average_bottom = [bottom_images[idx] for idx in range(1, n_markers+1)]
        avg_bottom = np.mean(imgs_for_average_bottom, axis=0)
        # Mostramos esa media (si quieres aplicarle x2 adicional, podrías hacerlo, pero aquí ya está “iluminada”).
        ax_bottom_right.imshow(avg_bottom, cmap='gray')  
        
        if selected_cells:
            # Celdas finales
            cmap_selected = ListedColormap(['black'] + [np.random.rand(3,) for _ in range(len(selected_cells))])
            ax_bottom_right.imshow(mask_selected * cell_mask, cmap=cmap_selected, interpolation='nearest', alpha=0.7)
            ax_bottom_right.set_title(f'Final Filter:\n{conditions}')
        else:
            ax_bottom_right.imshow(mask_selected, cmap='gray', alpha=0.7, interpolation='nearest')
            ax_bottom_right.set_title(f'Final Filter (0)\n{conditions}')
        
        ax_bottom_right.axis('off')
        bottom_images[n_markers+1] = avg_bottom  # por si lo necesitas más adelante
        
        plt.tight_layout()
        plt.show()
        print(f"Plot generado para {roi}.")


def create_marker_plot(
    roi, marker_mask, marker_name, marker_color,
    dapi_mask, merged_df,
    col_map, subpop_name, subpop_conditions,
    max_cells=None
    ):
    parsed = parse_conditions(subpop_conditions, col_map)
    sub_df = select_subpopulation(merged_df, parsed)
    if sub_df.empty:
        print(f"{roi} / {subpop_name}: 0 cells – skipping.")
        return None

    # KD‑trees
    pos_pts = np.column_stack(np.where(marker_mask == 1))
    neg_pts = np.column_stack(np.where(marker_mask == 0))
    tree_pos = cKDTree(pos_pts) if len(pos_pts) else None
    tree_neg = cKDTree(neg_pts) if len(neg_pts) else None

    if marker_name == 'CK':
        bin_col = 'is_positive_Pan_Cytokeratin_CK_binary'
        dist_col, row_col, col_col = (
            'distance_ck_mask', 'nearest_ck_row', 'nearest_ck_col')
    else:
        bin_col = 'is_positive_NGFR_binary'
        dist_col, row_col, col_col = (
            'distance_ngfr_mask', 'nearest_ngfr_row', 'nearest_ngfr_col')

    dists, near_pts = [], []
    for _, r in sub_df.iterrows():
        centroid = (r['centroid_row'], r['centroid_col'])
        if r[bin_col] == 0:
            dist, idx = tree_pos.query(centroid) if tree_pos else (0.0, 0)
            near  = tuple(map(int, tree_pos.data[idx])) if tree_pos else centroid
        else:
            dist, idx = tree_neg.query(centroid) if tree_neg else (0.0, 0)
            near  = tuple(map(int, tree_neg.data[idx])) if tree_neg else centroid
        dists.append(dist * PIXEL_SIZE)
        near_pts.append(near)

    sub_df = sub_df.copy()
    sub_df[dist_col] = dists
    sub_df[row_col]  = [p[0] for p in near_pts]
    sub_df[col_col]  = [p[1] for p in near_pts]

    # keep only cells with a non‑zero distance (=> a dashed line will be drawn)
    sub_df = sub_df[sub_df[dist_col] > 0]
    if sub_df.empty:
        print(f"{roi}/{subpop_name}: all distances are 0 – nothing to show.")
        return None

    # optional sampling
    if max_cells and max_cells < len(sub_df):
        plot_df = sub_df.sample(max_cells, random_state=42)
    else:
        plot_df = sub_df

    # ------------ FIGURE ------------
    fig, ax = plt.subplots(figsize=(12, 12))
    dapi_rgb = label2rgb(dapi_mask, bg_label=0, alpha=MASK_ALPHA,
                         colors=['lightgrey'])
    ax.imshow(dapi_rgb)

    cmap = 'Reds' if marker_name == 'CK' else 'Blues'
    ax.imshow(marker_mask, cmap=cmap, alpha=MASK_ALPHA)

    for contour in find_contours(marker_mask, .5):
        ax.plot(contour[:, 1], contour[:, 0], color=marker_color, lw=1)

    ax.scatter(plot_df['centroid_col'], plot_df['centroid_row'],
               c='yellow', s=30, edgecolors='black', label=subpop_name)

    for _, r in plot_df.iterrows():
        cent = (r['centroid_col'], r['centroid_row'])
        near = (r[col_col],       r[row_col])
        line_color = marker_color if r[bin_col] == 0 else 'green'
        ax.plot([cent[0], near[0]], [cent[1], near[1]],
                color=line_color, ls='--', lw=1.5)

    # ----------- LEGEND -------------
    legend_handles = [
        patches.Patch(color=marker_color, alpha=MASK_ALPHA,
                      label=f'{marker_name} Mask'),
        mlines.Line2D([], [], color=marker_color, lw=2,
                      label=f'{marker_name} Boundary'),
        mlines.Line2D([], [], color='yellow', marker='o',
                      markeredgecolor='black', markersize=6,
                      lw=0, label=subpop_name),
        mlines.Line2D([], [], color=marker_color, ls='--', lw=2,
                      label='Marker‑ (red) → Mask'),
        mlines.Line2D([], [], color='green', ls='--', lw=2,
                      label='Marker+ (green) → Stroma')
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.set_title(f'ROI {roi} • {subpop_name} vs {marker_name}')
    ax.axis('off')
    fig.tight_layout()
    return fig


def plot_roi_split_markers(
    roi, dapi_masks, ck_masks, ngfr_masks,
    df_bin, col_map,
    subpop_name, subpop_conditions,
    max_cells=None
    ):
    dapi = dapi_masks.get(f"{roi}_dapi")
    if dapi is None:
        print(f"No DAPI for {roi}");  return None, None

    ck   = ck_masks.get(roi,   np.zeros_like(dapi, dtype=np.uint8))
    ngfr = ngfr_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))

    props = regionprops(dapi)
    if not props:
        print(f"No cells in {roi}");  return None, None

    centroids = [{'id_celula': p.label,
                  'centroid_row': int(p.centroid[0]),
                  'centroid_col': int(p.centroid[1])} for p in props]
    cent_df = pd.DataFrame(centroids)

    merged = pd.merge(cent_df, df_bin,
                      left_on='id_celula', right_on='DAPI_ID', how='left')
    merged['is_positive_Pan_Cytokeratin_CK_binary'] = \
        merged['is_positive_Pan_Cytokeratin_CK_binary'].fillna(0).astype(int)
    merged['is_positive_NGFR_binary'] = \
        merged['is_positive_NGFR_binary'].fillna(0).astype(int)

    fig_ck   = create_marker_plot(
        roi, ck, 'CK', 'red', dapi, merged,
        col_map, subpop_name, subpop_conditions, max_cells)
    fig_ngfr = create_marker_plot(
        roi, ngfr, 'NGFR', 'blue', dapi, merged,
        col_map, subpop_name, subpop_conditions, max_cells)
    return fig_ck, fig_ngfr


def compute_and_save(
    roi, subpop_name, subpop_conditions,
    path_save,
    dapi_masks, ck_masks, ngfr_masks,
    df_bin, col_map,
    max_cells=None
    ):
    out_dir = os.path.join(path_save, subpop_name)
    os.makedirs(out_dir, exist_ok=True)

    fig_ck, fig_ngfr = plot_roi_split_markers(
        roi, dapi_masks, ck_masks, ngfr_masks,
        df_bin, col_map, subpop_name, subpop_conditions,
        max_cells)

    if fig_ck is None and fig_ngfr is None:
        print(f"{roi}: {subpop_name} – no figures (distance 0 everywhere)")
        return

    if fig_ck:
        fig_ck.savefig(os.path.join(out_dir, f"roi_{roi}_ck.svg"))
        display(fig_ck)
    if fig_ngfr:
        fig_ngfr.savefig(os.path.join(out_dir, f"roi_{roi}_ngfr.svg"))
        display(fig_ngfr)

    # ---------- DISTANCE TABLE ----------
    dapi = dapi_masks.get(f"{roi}_dapi")
    ck   = ck_masks.get(roi,   np.zeros_like(dapi, dtype=np.uint8))
    ngfr = ngfr_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))

    props = regionprops(dapi)
    centroids = [{'cell_id': p.label,
                  'centroid_row': int(p.centroid[0]),
                  'centroid_col': int(p.centroid[1])} for p in props]
    cent_df = pd.DataFrame(centroids)

    merged = pd.merge(cent_df, df_bin,
                      left_on='cell_id', right_on='DAPI_ID', how='left')
    merged['is_positive_Pan_Cytokeratin_CK_binary'] = \
        merged['is_positive_Pan_Cytokeratin_CK_binary'].fillna(0).astype(int)
    merged['is_positive_NGFR_binary'] = \
        merged['is_positive_NGFR_binary'].fillna(0).astype(int)

    parsed = parse_conditions(subpop_conditions, col_map)
    sub_df = select_subpopulation(merged, parsed)

    ck_pos, ck_neg = compute_distances(
        sub_df, ck, 'is_positive_Pan_Cytokeratin_CK_binary')
    ng_pos, ng_neg = compute_distances(
        sub_df, ngfr, 'is_positive_NGFR_binary')

    sub_df = sub_df.copy()
    sub_df['distance_ck_positive']   = ck_pos
    sub_df['distance_ck_negative']   = ck_neg
    sub_df['distance_ngfr_positive'] = ng_pos
    sub_df['distance_ngfr_negative'] = ng_neg

    dist_tbl = sub_df[[
        'cell_id', 'centroid_row', 'centroid_col',
        'distance_ck_positive', 'distance_ck_negative',
        'distance_ngfr_positive', 'distance_ngfr_negative'
    ]].copy()
    dist_tbl.insert(0, 'ROI', roi)

    keep = (
        (dist_tbl['distance_ck_positive']   > 0) |
        (dist_tbl['distance_ck_negative']   > 0) |
        (dist_tbl['distance_ngfr_positive'] > 0) |
        (dist_tbl['distance_ngfr_negative'] > 0)
    )
    dist_tbl = dist_tbl[keep]

    if dist_tbl.empty:
        print(f"{roi}: {subpop_name} – CSV empty (distance 0 everywhere)")
        return

    csv_path = os.path.join(out_dir, f"roi_{roi}_distance_table.csv")
    dist_tbl.to_csv(csv_path, index=False)
    print(f"✔  CSV saved → {csv_path}")


def _reorder_masks(masks_to_shade):
    """
    Ensures CK_mask is drawn first, and everything else later.
    """
    def priority(mask_name):
        if mask_name == "CK_mask":
            return 0
        else:
            return 1
    return sorted(masks_to_shade, key=priority)

def shade_selected_masks(ax, roi, masks_to_shade, shading_dict, alpha=0.25):
    """
    Overlays multiple masks as translucent colored areas on the current axes.
    Returns a list of Patch handles for the legend.
    """
    legend_handles = []
    ordered_masks = _reorder_masks(masks_to_shade)

    for mask_name in ordered_masks:
        if mask_name not in shading_dict:
            continue
        mask_dict, color = shading_dict[mask_name]
        if roi in mask_dict:
            mask_data = mask_dict[roi]
            # Create a custom colormap with two entries: transparent and 'color'
            custom_cmap = ListedColormap([(0, 0, 0, 0), color + (1.0,)])
            ax.imshow(mask_data, cmap=custom_cmap, alpha=alpha)

            # Dummy patch for the legend
            patch = Patch(facecolor=color, alpha=alpha, label=mask_name)
            legend_handles.append(patch)

    return legend_handles

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct finite Voronoi polygons in 2D.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Map ridge vertices to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Already handled finite ridge
                continue

            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # Tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Sort region counterclockwise
        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()

        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)

def plot_subpopulations_and_distances(
    roi,
    dapi_mask,
    subpopA_df,
    subpopB_df,
    dist_df,
    subpopA_name,
    subpopB_name,
    masks_to_shade=None,
    shading_dict=None,
    pixel_size=0.17,
    max_pairs=None,
    plot_type="line",
    save_plot=False,         # New parameter to control plot saving
    plot_filename=None        # New parameter for the plot filename
):
    """
    Plots subpopulations and their distances within a ROI using either lines or Voronoi diagrams.

    Parameters
    ----------
    roi : str
        Region of Interest identifier.
    dapi_mask : ndarray
        Labeled DAPI mask for the ROI.
    subpopA_df : DataFrame
        DataFrame containing cells of subpopulation A with centroids.
    subpopB_df : DataFrame
        DataFrame containing cells of subpopulation B with centroids.
    dist_df : DataFrame
        DataFrame containing pairwise distances between subpop A and B cells.
    subpopA_name : str
        Name of subpopulation A.
    subpopB_name : str
        Name of subpopulation B.
    masks_to_shade : list of str, optional
        List of mask names to overlay on the plot.
    shading_dict : dict, optional
        Dictionary mapping mask names to their data and colors.
    pixel_size : float, optional
        Size of a pixel in micrometers.
    max_pairs : int, optional
        Maximum number of pairs to plot. If None, all pairs are plotted.
    plot_type : str, optional
        Type of visualization: "line" or "voronoi".
    save_plot : bool, optional
        Whether to save the plot as an SVG file.
    plot_filename : str, optional
        The filename (including path) to save the plot.
    """
    unique_labels = np.unique(dapi_mask)
    unique_labels = unique_labels[unique_labels != 0]
    random.seed(42)
    colors = np.random.rand(len(unique_labels)+1, 3)
    colors[0] = [0, 0, 0]  # background = black
    cmap_cells = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(dapi_mask, cmap=cmap_cells, alpha=0.4, zorder=0)

    # Optional: shading
    shading_handles = []
    if masks_to_shade and shading_dict:
        shading_handles = shade_selected_masks(ax, roi, masks_to_shade, shading_dict, alpha=0.25)

    if plot_type == "line":
        # Calculate minimum distances
        min_distA2B = dist_df.loc[dist_df.groupby('A_cell_id')['distance_px'].idxmin()]
        min_distB2A = dist_df.loc[dist_df.groupby('B_cell_id')['distance_px'].idxmin()]

        # Configure colormap for A->B lines
        if not min_distA2B.empty:
            distances = min_distA2B['distance_px']
            norm = Normalize(vmin=distances.min(), vmax=distances.max())
            cmap = cm.get_cmap('hot')  # Changed to 'hot' for yellow-red color scheme
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # Necessary for colorbar

        # 1) Subpopulation A (blue points) - zorder=1
        ax.scatter(
            subpopA_df['centroid_col'],
            subpopA_df['centroid_row'],
            s=20, c='darkblue',
            label=subpopA_name,
            zorder=1,
            alpha=0.5
        )

        # 2) Lines A->B (color based on distance) - zorder=2
        if max_pairs is not None:
            min_distA2B = min_distA2B.sample(min(len(min_distA2B), max_pairs), random_state=42)
        for _, row in min_distA2B.iterrows():
            a_col, a_row = row['A_col'], row['A_row']
            b_col, b_row = row['B_col'], row['B_row']
            distance = row['distance_px']
            color = cmap(norm(distance))  # Get color based on distance
            ax.plot([a_col, b_col], [a_row, b_row],
                    color=color, linewidth=1, alpha=0.8, zorder=2)

        # 3) Subpopulation B (green points) - zorder=3
        ax.scatter(
            subpopB_df['centroid_col'],
            subpopB_df['centroid_row'],
            s=20, c='darkgreen',
            label=subpopB_name,
            zorder=3
        )

        # 4) Lines B->A (light green) - zorder=4
        if max_pairs is not None:
            min_distB2A = min_distB2A.sample(min(len(min_distB2A), max_pairs), random_state=42)
        for _, row in min_distB2A.iterrows():
            b_col, b_row = row['B_col'], row['B_row']
            a_col, a_row = row['A_col'], row['A_row']
            ax.plot([b_col, a_col], [b_row, a_row],
                    c='lightgreen', linewidth=2, alpha=0.8, zorder=4)

        ax.set_title(f"ROI: {roi} - Distances from '{subpopB_name}' to '{subpopA_name}'")

        # Finalize legend
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        final_handles = existing_handles + shading_handles
        final_labels = existing_labels + [h.get_label() for h in shading_handles]
        ax.legend(final_handles, final_labels, loc='best')

        # Add colorbar for distances
        if not min_distA2B.empty:
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Distance (px)')

    elif plot_type == "voronoi":
        # Determine which subpopulation is smaller
        if len(subpopA_df) <= len(subpopB_df):
            seeds_df = subpopA_df.copy()
            points_df = subpopB_df.copy()
            seeds_name = subpopA_name
            points_name = subpopB_name
            seed_id = 'A_cell_id'
            point_id = 'B_cell_id'
        else:
            seeds_df = subpopB_df.copy()
            points_df = subpopA_df.copy()
            seeds_name = subpopB_name
            points_name = subpopA_name
            seed_id = 'B_cell_id'
            point_id = 'A_cell_id'

        # Extract seed points
        seed_points = seeds_df[['centroid_col', 'centroid_row']].values

        # Handle case with fewer than 1 seed point
        if len(seed_points) < 1:
            print(f"Not enough seed points for Voronoi diagram in ROI {roi}. Skipping Voronoi visualization.")
            return

        # Create KDTree for efficient nearest neighbor search
        tree = KDTree(seed_points)

        # Extract coordinates of subpopulation B cells
        point_coords = points_df[['centroid_col', 'centroid_row']].values

        # Query nearest seed for each point
        distances, indices = tree.query(point_coords, k=1)

        # Assign nearest seed index to each point
        points_df['nearest_seed_idx'] = indices.flatten()
        points_df['distance_px'] = distances.flatten()

        # Get coordinates of nearest seeds for each point
        nearest_seeds = seed_points[points_df['nearest_seed_idx'].values]
        points_df['seed_col'] = nearest_seeds[:, 0]
        points_df['seed_row'] = nearest_seeds[:, 1]

        # Normalize distances for colormap
        norm = Normalize(vmin=points_df['distance_px'].min(), vmax=points_df['distance_px'].max())
        cmap = cm.get_cmap('hot')  # From red to yellow
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Necessary for colorbar

        # Plot Voronoi polygons (optional)
        try:
            vor = Voronoi(seed_points)
            regions, vertices = voronoi_finite_polygons_2d(vor)

            # Define plot boundaries
            min_x = 0
            max_x = dapi_mask.shape[1]
            min_y = 0
            max_y = dapi_mask.shape[0]
            bounding_polygon = ShapelyPolygon([
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y)
            ])

            # Assign unique color to each Voronoi region
            num_regions = len(regions)
            vor_colors = cm.tab20(np.linspace(0, 1, num_regions))

            patches = []
            colors_voronoi = []
            for region_idx, region in enumerate(regions):
                polygon = vertices[region]
                shapely_poly = ShapelyPolygon(polygon)
                # Clip polygon to plot boundaries
                shapely_poly = shapely_poly.intersection(bounding_polygon)
                if not shapely_poly.is_empty:
                    if isinstance(shapely_poly, ShapelyPolygon):
                        patches.append(Polygon(np.array(shapely_poly.exterior.coords)))
                        colors_voronoi.append(vor_colors[region_idx % len(vor_colors)])
                    elif isinstance(shapely_poly, (list, np.ndarray)):
                        for poly in shapely_poly:
                            patches.append(Polygon(np.array(poly.exterior.coords)))
                            colors_voronoi.append(vor_colors[region_idx % len(vor_colors)])
            # Create a collection of patches
            p = PatchCollection(patches, facecolor=colors_voronoi, edgecolor='k', alpha=0.3, zorder=1)
            ax.add_collection(p)
        except Exception as e:
            print(f"Error generating Voronoi diagram: {e}")

        # Plot subpopulation B points colored by distance
        sc = ax.scatter(
            points_df['centroid_col'],
            points_df['centroid_row'],
            c=points_df['distance_px'],
            cmap=cmap,
            s=20,
            label=f"{points_name} (colored by distance)",
            zorder=3,
            alpha=0.4
        )

        # Add colorbar for heatmap
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance to Centroid (px)')

        # Plot seed points
        ax.scatter(
            seeds_df['centroid_col'],
            seeds_df['centroid_row'],
            s=40,
            c='green',
            marker='o',
            label=f"{seeds_name}",
            zorder=4
        )

        ax.set_title(f"ROI: {roi} - Voronoi Diagram between '{seeds_name}' and '{points_name}'")

        # Finalize legend
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        final_handles = existing_handles + shading_handles
        final_labels = existing_labels + [h.get_label() for h in shading_handles]
        ax.legend(final_handles, final_labels, loc='best')

    else:
        print(f"Unknown plot_type '{plot_type}'. Supported types are 'line' and 'voronoi'.")

    ax.set_axis_off()

    # Save the plot as SVG if required
    if save_plot and plot_filename:
        # Ensure the directory exists
        plot_dir = os.path.dirname(plot_filename)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # Save the figure
        plt.savefig(plot_filename, format='svg', bbox_inches='tight')
        print(f"Plot saved as SVG: {plot_filename}")

    plt.show()

def compute_and_plot_subpop_distances_for_all_rois(
    rois,
    subpop_conditions_A,
    subpop_conditions_B,
    df_binary,
    dapi_masks_dict,
    condition_column_map,
    pixel_size=0.17,
    max_pairs=None,
    masks_to_shade=None,
    shading_dict=None,
    save_matrix_as_csv=False,
    path_save="..\\results_spatial_analysis\\"+CNIO_USER+"\\distances_between_populations",
    print_pivot_head=False,
    plot_type="line",
    subpopB_label=None  # Nuevo parámetro opcional
):
    """
    For each ROI in 'rois':
      1) Parse subpop A conditions and subpop B conditions.
      2) Find subpop A / subpop B within that ROI.
      3) Compute all distances between every cell in A and every cell in B.
      4) Create a pivot table (rows = A_roi_cell, columns = B_roi_cell).
      5) Optionally print the first 10 rows of the pivot table if print_pivot_head=True.
      6) Optionally save the pivot table as CSV if save_matrix_as_csv=True.
      7) Plot each ROI, shading selected masks, and show subpopulations plus lines
         (A->B and B->A) or Voronoi diagrams based on the specified plot_type.

    Parameters
    ----------
    rois : list of str
        List of Regions of Interest identifiers.
    subpop_conditions_A : list of str
        List of conditions for subpopulation A.
    subpop_conditions_B : list of str
        List of conditions for subpopulation B.
    df_binary : DataFrame
        DataFrame containing binary data for cells.
    dapi_masks_dict : dict
        Dictionary of DAPI masks by ROI.
    condition_column_map : dict
        Mapping from condition names to DataFrame column names.
    pixel_size : float, optional
        Size of a pixel in micrometers.
    max_pairs : int, optional
        Maximum number of pairs to plot. If None, all pairs are plotted.
    masks_to_shade : list of str, optional
        List of mask names to overlay on the plot.
    shading_dict : dict, optional
        Dictionary mapping mask names to their data and colors.
    save_matrix_as_csv : bool, optional
        Whether to save the distance matrix as a CSV file.
    print_pivot_head : bool, optional
        Whether to print the first 10 rows of the pivot table.
    plot_type : str, optional
        Type of visualization: "line" or "voronoi".

    Returns
    -------
    all_distances : dict
        Dictionary of DataFrames, one per ROI, containing all pairwise distances.
    """
    subpopA_name = " & ".join(subpop_conditions_A)
    subpopB_name = subpopB_label if subpopB_label is not None else " & ".join(subpop_conditions_B)
    parsedA = parse_conditions(subpop_conditions_A, condition_column_map)
    parsedB = parse_conditions(subpop_conditions_B, condition_column_map)

    all_distances = {}

    for roi in rois:
        print(f"\n=== ROI {roi} ===")
        dapi_key = roi.lower() + "_dapi"
        if dapi_key not in dapi_masks_dict:
            print(f"Missing DAPI mask for {dapi_key}. Skipping.")
            continue

        # Extract the labeled DAPI mask
        dapi_mask = dapi_masks_dict[dapi_key]

        # Filter df_binary for this ROI
        df_roi = df_binary[df_binary['ROI'] == roi].copy()

        # Merge with centroids
        centroids_df = get_centroids(dapi_mask)
        df_roi = pd.merge(df_roi, centroids_df, on='DAPI_ID', how='left')

        # Select subpopulations
        subpopA = select_subpopulation(df_roi, parsedA)
        subpopB = select_subpopulation(df_roi, parsedB)

        # Compute all distances
        dist_df = compute_subpop_distances(subpopA, subpopB)
        if dist_df.empty:
            print(f"No distances to compute for {roi} (subpopulation A or B is empty).")
            all_distances[roi] = dist_df
            continue  # Skip to next ROI      
        dist_df['ROI'] = roi
        dist_df['distance_um'] = dist_df['distance_px'] * pixel_size

        all_distances[roi] = dist_df

        # Create pivot table
        dist_df['A_roi_cell'] = dist_df['ROI'].astype(str) + "_" + dist_df['A_cell_id'].astype(str)
        dist_df['B_roi_cell'] = dist_df['ROI'].astype(str) + "_" + dist_df['B_cell_id'].astype(str)

        distance_matrix = dist_df.pivot(
            index='A_roi_cell',
            columns='B_roi_cell',
            values='distance_um'
        )

        print(f"rows: {subpopA_name}")
        print(f"columns: {subpopB_name}")
        if print_pivot_head:
            print(distance_matrix.head(10))

        if save_matrix_as_csv:
            # Create the filename
            csv_filename = f"distance_matrix_{roi}_{subpopA_name}_vs_{subpopB_name}.csv"
            
            # Ensure the directory exists
            if path_save and not os.path.exists(path_save):
                os.makedirs(path_save)
            
            # Construct the full file path
            csv_filepath = os.path.join(path_save, csv_filename)
            
            # Save the DataFrame as CSV
            distance_matrix.to_csv(csv_filepath, index=True)
            print(f"Matrix saved as CSV: {csv_filepath}")

        # Plot only if both subpopulations have cells
        if len(subpopA) > 0 and len(subpopB) > 0:
            # Define the plot filename
            plot_filename = os.path.join(
                path_save,
                f"plot_{roi}_{subpopA_name}_vs_{subpopB_name}.svg"
            )
            
            plot_subpopulations_and_distances(
                roi=roi,
                dapi_mask=dapi_mask,
                subpopA_df=subpopA,
                subpopB_df=subpopB,
                dist_df=dist_df,
                subpopA_name=subpopA_name,
                subpopB_name=subpopB_name,
                masks_to_shade=masks_to_shade,
                shading_dict=shading_dict,
                pixel_size=pixel_size,
                max_pairs=max_pairs,
                plot_type=plot_type,  # Pass the plot_type parameter
                save_plot=True,       # Enable plot saving
                plot_filename=plot_filename  # Provide the plot filename
            )
        else:
            print(f"One or both subpopulations are empty in {roi}. No plot.")

    return all_distances

