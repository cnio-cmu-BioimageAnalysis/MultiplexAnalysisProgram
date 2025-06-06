import re
import numpy as np
from matplotlib.patches import  Patch

def plot_masks(dapi_masks_dict):
    import numpy as np 
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt 

    for extracted_name, mask in dapi_masks_dict.items():
        # Create a random color map for the labels
        unique_labels = np.unique(mask)
        num_labels = len(unique_labels)
        colors = np.random.rand(num_labels, 3)  # Generate random colors
        colors[0] = [0, 0, 0]  # Background (label 0) in black
        cmap = ListedColormap(colors)

        # Display the mask with the assigned colors
        plt.figure(figsize=(12, 12))
        plt.imshow(mask, cmap=cmap, interpolation='nearest')
        plt.title(f"Segmented Mask - {extracted_name}")
        plt.colorbar(boundaries=np.arange(-0.5, num_labels, 1), ticks=np.linspace(0, num_labels - 1, min(20, num_labels), dtype=int))
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.show()

        # Print statistics
        print(f"Number of unique labels in {extracted_name} (excluding background): {num_labels - 1}")


def generate_boxplots_by_subpop(subpop_name, roi_data, positive_col, negative_col, marker_label):
    """
    Genera un boxplot combinado de distancias positivas y negativas para cada ROI de una subpoblación.
    La figura tendrá de título la subpoblación y el eje x mostrará los nombres de ROI (spliteados).
    """
    import numpy as np 
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import matplotlib.patches as mpatches

    try:
        print(f"\nGenerando boxplot para subpoblación: {subpop_name}")
        
        # Ordena ROIs y prepara datos (filtrando ceros)
        roi_names = sorted(roi_data.keys())
        pos_data, neg_data = [], []
        for roi in roi_names:
            try:
                print(f"Procesando ROI: {roi}")
                df = roi_data[roi]
                pos_vals = df[positive_col][df[positive_col] != 0].tolist()
                neg_vals = df[negative_col][df[negative_col] != 0].tolist()
                
                if len(pos_vals) == 0:
                    print(f"Advertencia: Lista positiva vacía en {roi}; se inserta NaN")
                    pos_vals = [float('nan')]
                elif len(pos_vals) == 1:
                    print(f"Advertencia: Un solo valor positivo en {roi}; duplicándolo")
                    pos_vals *= 2
                
                if len(neg_vals) == 0:
                    print(f"Advertencia: Lista negativa vacía en {roi}; se inserta NaN")
                    neg_vals = [float('nan')]
                elif len(neg_vals) == 1:
                    print(f"Advertencia: Un solo valor negativo en {roi}; duplicándolo")
                    neg_vals *= 2

                pos_data.append(pos_vals)
                neg_data.append(neg_vals)
            except Exception as e:
                print(f"Error procesando ROI {roi}: {e}")
        
        # Crear figura y eje
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            print("Figura y eje creados")
        except Exception as e:
            print(f"Error al crear figura: {e}")
            return None
        
        positions = list(range(1, len(roi_names) + 1))
        
        # Configuración según marker_label
        if marker_label.lower() == "ck":
            top_title = "Cell to Tumor Distances"
            bottom_title = "Cell to Stroma Distances"
            pos_color = "lightcoral"
            neg_color = "lightblue"
        else:
            top_title = f"Cell to {marker_label} Positive Distances"
            bottom_title = f"Cell to {marker_label} Negative Distances"
            pos_color = "violet"
            neg_color = "skyblue"
        
        offset = 0.2
        pos_positions = [p - offset for p in positions]
        neg_positions = [p + offset for p in positions]
        
        # Trazar datos positivos
        try:
            vp_pos = ax.violinplot(
                pos_data, positions=pos_positions, widths=0.4,
                showmeans=False, showmedians=False, showextrema=False
            )
            for body in vp_pos['bodies']:
                body.set_facecolor(pos_color)
                body.set_edgecolor('black')
                body.set_alpha(0.5)
            bp_pos = ax.boxplot(
                pos_data, positions=pos_positions, widths=0.15,
                patch_artist=True, showfliers=False
            )
            for box in bp_pos['boxes']:
                box.set_facecolor('white')
                box.set_edgecolor('black')
            for median in bp_pos['medians']:
                median.set_color('black')
            print("Datos positivos trazados")
        except Exception as e:
            print(f"Error trazando datos positivos: {e}")
        
        # Trazar datos negativos
        try:
            vp_neg = ax.violinplot(
                neg_data, positions=neg_positions, widths=0.4,
                showmeans=False, showmedians=False, showextrema=False
            )
            for body in vp_neg['bodies']:
                body.set_facecolor(neg_color)
                body.set_edgecolor('black')
                body.set_alpha(0.5)
            bp_neg = ax.boxplot(
                neg_data, positions=neg_positions, widths=0.15,
                patch_artist=True, showfliers=False
            )
            for box in bp_neg['boxes']:
                box.set_facecolor('white')
                box.set_edgecolor('black')
            for median in bp_neg['medians']:
                median.set_color('black')
            print("Datos negativos trazados")
        except Exception as e:
            print(f"Error trazando datos negativos: {e}")
        
        # Etiquetas, título y leyenda
        try:
            ax.set_ylabel("Distance (µm)")
            ax.set_title(f"{top_title} vs {bottom_title}", pad=15)
            # Se splittea cada ROI para mejorar la lectura en el eje x
            xlabels = [" ".join(roi.split('_')) for roi in roi_names]
            ax.set_xticks(positions)
            ax.set_xticklabels(xlabels, rotation=45)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))
            
            if marker_label.lower() == "ck":
                pos_label = "Cell to Tumor"
                neg_label = "Cell to Stroma"
            else:
                pos_label = f"Cell to {marker_label} Positive"
                neg_label = f"Cell to {marker_label} Negative"
            pos_patch = mpatches.Patch(color=pos_color, label=pos_label)
            neg_patch = mpatches.Patch(color=neg_color, label=neg_label)
            ax.legend(handles=[pos_patch, neg_patch], loc='best')
            print("Etiquetas y leyenda configuradas")
        except Exception as e:
            print(f"Error configurando etiquetas: {e}")
        
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.88])
            print("Layout ajustado")
        except Exception as e:
            print(f"Error ajustando layout: {e}")
        
        print(f"Boxplot generado para subpoblación: {subpop_name}")
        return fig
    except Exception as e:
        print(f"Error general en generate_boxplots_by_subpop para {subpop_name}: {e}")
        return None


def generate_boxplots_by_roi(roi_name, subpop_data_for_roi, positive_col, negative_col, marker_label):
    """
    Genera un boxplot combinado de distancias positivas y negativas para cada subpoblación.
    Se agregan múltiples bloques try/except y mensajes de depuración para evitar que se detenga la ejecución.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter

    try:
        print(f"\nGenerando boxplot para ROI: {roi_name}")
        import numpy as np
        import matplotlib.patches as mpatches

        # Ordena subpoblaciones y prepara listas de datos (filtrando ceros)
        subpops = sorted(subpop_data_for_roi.keys())
        pos_data = []
        neg_data = []
        for subpop in subpops:
            try:
                print(f"Procesando subpoblación: {subpop}")
                df = subpop_data_for_roi[subpop]
                pos_values = df[positive_col][df[positive_col] != 0].tolist()
                neg_values = df[negative_col][df[negative_col] != 0].tolist()
                
                if len(pos_values) == 0:
                    print(f"Advertencia: Lista positiva vacía en {subpop}; se inserta NaN")
                    pos_values = [float('nan')]
                elif len(pos_values) == 1:
                    print(f"Advertencia: Un solo valor positivo en {subpop}; duplicándolo")
                    pos_values = pos_values * 2
                
                if len(neg_values) == 0:
                    print(f"Advertencia: Lista negativa vacía en {subpop}; se inserta NaN")
                    neg_values = [float('nan')]
                elif len(neg_values) == 1:
                    print(f"Advertencia: Un solo valor negativo en {subpop}; duplicándolo")
                    neg_values = neg_values * 2

                pos_data.append(pos_values)
                neg_data.append(neg_values)
            except Exception as e:
                print(f"Error procesando subpoblación {subpop}: {e}")
        
        # Crear figura y eje
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            print("Figura y eje creados correctamente")
        except Exception as e:
            print(f"Error al crear la figura: {e}")
            return None
        
        positions = list(range(1, len(subpops) + 1))
        
        # Configuración según marker_label
        if marker_label.lower() == "ck":
            top_title = "Cell to Tumor Distances"
            bottom_title = "Cell to Stroma Distances"
            pos_color = "lightcoral"
            neg_color = "lightblue"
        else:
            top_title = f"Cell to {marker_label} Positive Distances"
            bottom_title = f"Cell to {marker_label} Negative Distances"
            pos_color = "violet"
            neg_color = "skyblue"
        
        offset = 0.2
        pos_positions = [p - offset for p in positions]
        neg_positions = [p + offset for p in positions]
        
        # Trazado de datos positivos
        try:
            vp_pos = ax.violinplot(
                pos_data, positions=pos_positions, widths=0.4,
                showmeans=False, showmedians=False, showextrema=False
            )
            for body in vp_pos['bodies']:
                body.set_facecolor(pos_color)
                body.set_edgecolor('black')
                body.set_alpha(0.5)
            bp_pos = ax.boxplot(
                pos_data, positions=pos_positions, widths=0.15,
                patch_artist=True, showfliers=False
            )
            for box in bp_pos['boxes']:
                box.set_facecolor('white')
                box.set_edgecolor('black')
            for median in bp_pos['medians']:
                median.set_color('black')
            print("Datos positivos trazados correctamente")
        except Exception as e:
            print(f"Error al trazar datos positivos: {e}")
        
        # Trazado de datos negativos
        try:
            vp_neg = ax.violinplot(
                neg_data, positions=neg_positions, widths=0.4,
                showmeans=False, showmedians=False, showextrema=False
            )
            for body in vp_neg['bodies']:
                body.set_facecolor(neg_color)
                body.set_edgecolor('black')
                body.set_alpha(0.5)
            bp_neg = ax.boxplot(
                neg_data, positions=neg_positions, widths=0.15,
                patch_artist=True, showfliers=False
            )
            for box in bp_neg['boxes']:
                box.set_facecolor('white')
                box.set_edgecolor('black')
            for median in bp_neg['medians']:
                median.set_color('black')
            print("Datos negativos trazados correctamente")
        except Exception as e:
            print(f"Error al trazar datos negativos: {e}")
        
        # Configuración de etiquetas, títulos y leyenda
        try:
            ax.set_ylabel("Distance (µm)")
            ax.set_title(f"{top_title} vs {bottom_title}", pad=15)
            ax.set_xticks(positions)
            ax.set_xticklabels(subpops, rotation=45)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))

            if marker_label.lower() == "ck":
                pos_label = "Cell to Tumor"
                neg_label = "Cell to Stroma"
            else:
                pos_label = f"Cell to {marker_label} Positive"
                neg_label = f"Cell to {marker_label} Negative"
            pos_patch = mpatches.Patch(color=pos_color, label=pos_label)
            neg_patch = mpatches.Patch(color=neg_color, label=neg_label)
            ax.legend(handles=[pos_patch, neg_patch], loc='best')
            print("Etiquetas y leyenda configuradas")
        except Exception as e:
            print(f"Error al configurar etiquetas y leyenda: {e}")
        
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.88])
            print("Layout ajustado")
        except Exception as e:
            print(f"Error al ajustar el layout: {e}")
        
        print(f"Boxplot generado para ROI: {roi_name}")
        return fig
    except Exception as e:
        print(f"Error general en generate_boxplots_by_roi para ROI {roi_name}: {e}")
        return None
        



# multiplex_pipeline/visualization/qc.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple

def plot_combination_counts(
    df: pd.DataFrame,
    rois: List[str],
    combinations: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    output_dir: str,
    base_filename: str,
    plot_title: str,
    figsize: Tuple[int,int] = (20, 10),
    rotation: int = 45,
    font_scale: float = 1.0
) -> pd.DataFrame:
    """
    Filtra `df` por ROIs, cuenta para cada ROI el número de True en cada condición de `combinations`,
    guarda la tabla de counts como CSV en output_dir/{base_filename}.csv,
    genera un barplot apilado con escala log y guarda como SVG en output_dir/{base_filename}.svg,
    y retorna el DataFrame de counts.
    """
    # 1. Filtrado
    filtered = df[df['ROI'].isin(rois)]

    # 2–5. Cálculo de counts_df
    counts_df = pd.DataFrame()
    grouped = filtered.groupby('ROI')
    for name, cond in combinations.items():
        counts_df[name] = grouped.apply(lambda x: cond(x).sum())
    counts_df = counts_df.reset_index()
    counts_df['Total Cells'] = grouped.size().values

    # 6–7. Melt para plot
    counts_melted = counts_df.melt(
        id_vars=['ROI','Total Cells'],
        var_name='Combination',
        value_name='Count'
    )

    # 8–9. Directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # 10–13. Guardar CSV
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    counts_df.to_csv(csv_path, index=False)
    print(f"Table saved as CSV at: {csv_path}")

    # 14. Plot
    plt.figure(figsize=figsize)
    sns.set_context("talk", font_scale=font_scale)
    ax = sns.barplot(
        x='ROI', y='Count', hue='Combination',
        data=counts_melted
    )
    # anotar encima de cada barra
    for p in ax.patches:
        h = p.get_height()
        if pd.notnull(h) and h > 0:
            ax.text(
                p.get_x() + p.get_width()/2,
                h + 0.1,
                f"{int(h)}",
                ha='center', va='bottom', fontsize=10, rotation=90
            )
    # total en etiquetas x
    new_lbls = [
        f"{roi}\n(Total: {tot})"
        for roi, tot in zip(counts_df['ROI'], counts_df['Total Cells'])
    ]
    ax.set_xticklabels(new_lbls, rotation=rotation)
    ax.set_yscale('log')
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('ROI', fontsize=14)
    ax.set_ylabel('Number of Positive Cells', fontsize=14)
    ax.legend(title='Combination', fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # 15. Guardar SVG
    svg_path = os.path.join(output_dir, f"{base_filename}.svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Plot saved as SVG at: {svg_path}")

    # 16. Mostrar
    plt.show()

    return counts_df

def generate_plots_by_population_heatmap(dic_distancias, save_path, cnio_user):
    out_dir = os.path.join(save_path, cnio_user, "by_population_heatmap")
    os.makedirs(out_dir, exist_ok=True)

    pop_dict = {}
    for _, files in dic_distancias.items():
        for fname, df in files.items():
            m = re.match(r"(distance_matrix_(roi\d+)_.*NGFR_intensity)([\+\-])(_vs.*)\.csv", fname)
            if not m:
                continue
            roi, sign = m.group(2), m.group(3)
            pop = (m.group(1).replace(f"distance_matrix_{roi}_", "") + m.group(4))
            arr = pd.to_numeric(df.values.flatten(), errors='coerce')
            arr = arr[~np.isnan(arr)]
            pop_dict.setdefault(pop, {}).setdefault(roi, {})['plus' if sign=='+' else 'minus'] = arr

    c_plus, c_minus = 'lightgreen', 'lightcoral'
    for pop, rois in pop_dict.items():
        sorted_rois = sorted(rois, key=lambda x: int(re.findall(r'\d+', x)[0]))
        fig, ax = plt.subplots(figsize=(8, 5))
        pos_plus, data_plus = [], []
        pos_minus, data_minus = [], []
        for i, roi in enumerate(sorted_rois):
            d = rois[roi]
            if 'plus' in d and 'minus' in d:
                pos_plus.append(3*i+1); data_plus.append(d['plus'])
                pos_minus.append(3*i+2); data_minus.append(d['minus'])
        bp1 = ax.boxplot(data_plus, positions=pos_plus, patch_artist=True, widths=0.6, showfliers=False)
        bp2 = ax.boxplot(data_minus, positions=pos_minus, patch_artist=True, widths=0.6, showfliers=False)
        for b in bp1['boxes']: b.set(facecolor=c_plus, alpha=0.5)
        for b in bp2['boxes']: b.set(facecolor=c_minus, alpha=0.5)
        vp1 = ax.violinplot(data_plus, positions=pos_plus, widths=0.6, showextrema=False)
        vp2 = ax.violinplot(data_minus, positions=pos_minus, widths=0.6, showextrema=False)
        for v in vp1['bodies']: v.set_facecolor(c_plus); v.set_edgecolor('black'); v.set_alpha(0.5)
        for v in vp2['bodies']: v.set_facecolor(c_minus); v.set_edgecolor('black'); v.set_alpha(0.5)
        centers, labels = [], []
        for i, roi in enumerate(sorted_rois):
            if 'plus' in rois[roi] and 'minus' in rois[roi]:
                centers.append((3*i+1 + 3*i+2)/2)
                labels.append(roi)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Distance [µm]")
        pop_name = re.search(r'vs_(.*)', pop).group(1).replace("_", " ") if re.search(r'vs_(.*)', pop) else pop
        ax.set_title(f"Distances tumor vs {pop_name}")
        ax.legend([Patch(facecolor=c_plus), Patch(facecolor=c_minus)], ['NGFR+','NGFR−'], loc='upper right')
        plt.tight_layout()
        safe = re.sub(r'[^\w\-_\. ]','_', pop_name)
        plt.savefig(os.path.join(out_dir, f"{safe}.svg"), format='svg')
        plt.show()
        plt.close(fig)
    return pop_dict

def generate_plots_by_roi_heatmap(population_dict, save_path, cnio_user):
    out_dir = os.path.join(save_path, cnio_user, "by_roi_heatmap")
    os.makedirs(out_dir, exist_ok=True)

    c_plus, c_minus = 'lightgreen', 'lightcoral'
    roi_dict = {}
    for pop, rois in population_dict.items():
        for roi, d in rois.items():
            roi_dict.setdefault(roi, {})[pop] = d

    for roi, pops in roi_dict.items():
        sorted_pops = sorted(pops)
        fig, ax = plt.subplots(figsize=(10, 6))
        pos_plus, data_plus = [], []
        pos_minus, data_minus = [], []
        for i, pop in enumerate(sorted_pops):
            d = pops[pop]
            if 'plus' in d and 'minus' in d:
                pos_plus.append(3*i+1); data_plus.append(d['plus'])
                pos_minus.append(3*i+2); data_minus.append(d['minus'])
        bp1 = ax.boxplot(data_plus, positions=pos_plus, patch_artist=True, widths=0.6, showfliers=False)
        bp2 = ax.boxplot(data_minus, positions=pos_minus, patch_artist=True, widths=0.6, showfliers=False)
        for b in bp1['boxes']: b.set(facecolor=c_plus, alpha=0.5)
        for b in bp2['boxes']: b.set(facecolor=c_minus, alpha=0.5)
        vp1 = ax.violinplot(data_plus, positions=pos_plus, widths=0.6, showextrema=False)
        vp2 = ax.violinplot(data_minus, positions=pos_minus, widths=0.6, showextrema=False)
        for v in vp1['bodies']: v.set_facecolor(c_plus); v.set_edgecolor('black'); v.set_alpha(0.5)
        for v in vp2['bodies']: v.set_facecolor(c_minus); v.set_edgecolor('black'); v.set_alpha(0.5)
        centers = [(3*i+1 + 3*i+2)/2 for i in range(len(sorted_pops))]
        ax.set_xticks(centers)
        ax.set_xticklabels([pop.replace("_"," ") for pop in sorted_pops], rotation=90, ha='center')
        ax.set_ylabel("Distance [µm]")
        ax.set_title(f"Distances for ROI {roi}")
        ax.legend([Patch(facecolor=c_plus), Patch(facecolor=c_minus)], ['NGFR+','NGFR−'], loc='upper right')
        plt.tight_layout()
        safe = re.sub(r'[^\w\-_\. ]','_', roi)
        plt.savefig(os.path.join(out_dir, f"{safe}.svg"), format='svg')
        plt.show()
        plt.close(fig)
