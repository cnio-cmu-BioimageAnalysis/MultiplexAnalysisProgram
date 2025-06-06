import os
import pandas as pd



def load_csv_data(base_dir):
    """
    Lee todos los archivos CSV (terminan en _distance_table.csv) de cada subcarpeta en base_dir.
    Retorna un diccionario: {subpop: {roi: dataframe, ...}, ...}
    """
    data_dict = {}
    for subpop in os.listdir(base_dir):
        subpop_path = os.path.join(base_dir, subpop)
        if os.path.isdir(subpop_path):
            data_dict[subpop] = {}
            for file in os.listdir(subpop_path):
                if file.endswith('_distance_table.csv'):
                    roi = file.replace("roi_", "").replace("_distance_table.csv", "")
                    file_path = os.path.join(subpop_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        data_dict[subpop][roi] = df
                        print(f"CSV leído: {file_path}")
                    except Exception as e:
                        print(f"Error leyendo {file_path}: {e}")
    return data_dict

def load_csv_data2(base_dir):
    """
    Lee todos los archivos CSV (que terminan en _distance_table.csv) de cada subcarpeta en base_dir.
    Retorna un diccionario anidado: {subpop: {roi: dataframe, ...}, ...}
    """
    data_dict = {}
    for subpop_folder in os.listdir(base_dir):
        subpop_path = os.path.join(base_dir, subpop_folder)
        if os.path.isdir(subpop_path):
            data_dict[subpop_folder] = {}
            for file in os.listdir(subpop_path):
                if file.endswith('_distance_table.csv'):
                    roi = file.replace("roi_", "").replace("_distance_table.csv", "")
                    file_path = os.path.join(subpop_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        data_dict[subpop_folder][roi] = df
                        print(f"CSV leído correctamente: {file_path}")
                    except Exception as e:
                        print(f"Error al leer {file_path}: {e}")
    return data_dict




import os
import tifffile
from typing import Dict

def load_ome_tif_images(folder: str) -> Dict[str, any]:
    """
    Carga todas las imágenes OME-TIFF de la carpeta indicada.
    Devuelve un dict {nombre_archivo: ndarray de la imagen}.
    Imprime un resumen de cuántas imágenes y sus formas.
    """
    images = {}
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.ome.tiff', '.ome.tif', '.tiff', '.tif')):
            full_path = os.path.join(folder, file_name)
            images[file_name] = tifffile.imread(full_path)
            break


    return images



# multiplex_pipeline/io/loaders.py

import os
import re
import imageio
from typing import Dict, Any

def load_dapi_masks(
    folder: str,
    roi_name: str = "roi1_dapi",
    pattern: str = r"(roi\d+_dapi)"
) -> Dict[str, Any]:
    """
    Carga las máscaras DAPI (.tif) de `folder` y retorna un dict
    {roiX_dapi: ndarray} filtrando únicamente por `roi_name`.
    Imprime progreso y dimensiones de cada máscara cargada.
    """
    masks: Dict[str, Any] = {}
    # Listado de .tif (insensible a mayúsculas)
    tif_files = [f for f in os.listdir(folder) if f.lower().endswith('.tif')]

    for file_name in tif_files:
        m = re.search(pattern, file_name.lower())
        if not m:
            print(f"El fichero '{file_name}' no encaja con el patrón.")
            continue
        key = m.group(1)
        if key != roi_name:
            continue

        path = os.path.join(folder, file_name)
        try:
            mask = imageio.imread(path)
            masks[key] = mask
            print(f"Cargando: {key} — dimensiones {mask.shape}")
        except Exception as e:
            print(f"Error cargando {file_name}: {e}")

    return masks



def load_distance_matrices(base_path):
    """
    Recorre cada subcarpeta de base_path, lee todos los archivos .csv
    y construye un diccionario {subcarpeta: {archivo.csv: DataFrame, ...}, ...}.
    
    Parámetros
    ----------
    base_path : str
        Ruta a la carpeta que contiene las subcarpetas con los CSV.

    Devuelve
    -------
    dict
        Diccionario anidado con los DataFrames cargados.
    """
    dic_distancias = {}
    for subcarpeta in os.listdir(base_path):
        ruta_sub = os.path.join(base_path, subcarpeta)
        if not os.path.isdir(ruta_sub):
            continue

        dicc_archivos = {}
        for archivo in os.listdir(ruta_sub):
            if archivo.lower().endswith('.csv'):
                ruta_archivo = os.path.join(ruta_sub, archivo)
                try:
                    df = pd.read_csv(ruta_archivo)
                    dicc_archivos[archivo] = df
                except Exception as e:
                    print(f"Error al leer {ruta_archivo}: {e}")
        dic_distancias[subcarpeta] = dicc_archivos

    return dic_distancias