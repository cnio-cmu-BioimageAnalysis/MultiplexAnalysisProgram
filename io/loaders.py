from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from tifffile import imread
import imageio.v2 as imageio
from tqdm import tqdm

from multiplex_pipeline.config import (
    DATA_FOLDER,
    EXPORT_DAPI_FOLDER,
    IMAGE_EXTENSIONS,
    DAPI_PATTERN,
    CSV_EXTENSION,
)


def load_ome_tif_images(
    data_folder: Path = DATA_FOLDER,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Loads all OME-TIFF images from the specified folder.

    Args:
        data_folder (Path): Path to search for images.
        show_progress (bool): If True, displays a progress bar during loading.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping filenames to image arrays (C, H, W).
    """
    images: Dict[str, np.ndarray] = {}
    base = Path(data_folder)

    # Normalize extensions to lowercase
    exts = {e.lower() for e in IMAGE_EXTENSIONS}

    # Collect all files that match any of the image extensions
    candidates = [
        p for p in base.iterdir()
        if p.is_file() and ''.join(p.suffixes).lower() in exts
    ]

    # Use tqdm for progress bar if requested
    iterator = tqdm(candidates, desc="Loading OME-TIFF images", unit="img") if show_progress else candidates

    for path in iterator:
        try:
            img = imread(path)
            images[path.name] = img
            
        except Exception as e:
            print(f"Error loading {path.name}: {e}")
    
    return images


def load_dapi_masks(
    export_folder: Path = EXPORT_DAPI_FOLDER,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Loads DAPI mask files matching DAPI_PATTERN from the export folder.

    Args:
        export_folder (Path): Path where DAPI masks are stored.
        show_progress (bool): If True, displays a progress bar during loading.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping 'roiX' to 2D mask arrays.
    """
    masks: Dict[str, np.ndarray] = {}
    base = Path(export_folder)
    tif_paths = [p for p in base.iterdir() if p.suffix.lower() == '.tif']

    # Use tqdm for progress bar if requested
    iterator = tqdm(tif_paths, desc="Loading DAPI masks", unit="mask") if show_progress else tif_paths

    for path in iterator:
        m = DAPI_PATTERN.search(path.stem)
        if not m:
            print(f"Skipping file '{path.name}', does not match DAPI pattern.")
            continue
        key = f"roi{m.group(1)}_dapi"
        
        
        try:
            masks[key] = imageio.imread(path)
        except Exception as e:
            print(f"Error loading DAPI mask {path.name}: {e}")
    
    return masks


def load_csv_data(
    base_path: Path,
    show_progress: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Recursively loads all CSV files under the given base path into a nested dictionary.
    The dictionary is structured as {subfolder: {filename: DataFrame}}.

    Args:
        base_path (Path): The root directory to search for CSV files.
        show_progress (bool): If True, displays a progress bar during loading.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: A nested dictionary of CSV files with DataFrames.
    """
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    base = Path(base_path)
    subdirs = [d for d in base.iterdir() if d.is_dir()]

    # Use tqdm for progress bar if requested
    dir_iter = tqdm(subdirs, desc="Loading CSV subfolders", unit="dir") if show_progress else subdirs

    for subdir in dir_iter:
        files_dict: Dict[str, pd.DataFrame] = {}
        csv_files = list(subdir.glob(f"*{CSV_EXTENSION}"))
        
        # Iterate over CSV files in each subdirectory
        file_iter = tqdm(csv_files, desc=f"Reading CSVs in {subdir.name}", unit="file") if show_progress else csv_files

        for file in file_iter:
            try:
                df = pd.read_csv(file)
                files_dict[file.name] = df
            except Exception as e:
                print(f"Error reading CSV {file}: {e}")
        
        result[subdir.name] = files_dict
    
    return result


def load_distance_matrices(base_path: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads distance matrices from subfolders and sub-subfolders under the given base path.

    Args:
        base_path (Path): The root directory to search for distance matrices.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: A dictionary mapping group -> subpopulation -> DataFrame.
    """
    dic_distancias = {}
    
    for group in os.listdir(base_path):  # "A_neg", "A_pos"
        path_group = os.path.join(base_path, group)
        
        # Skip non-directories
        if not os.path.isdir(path_group):
            continue
        
        files_dict = {}
        
        # Iterate over sub-populations within each group
        for subpop in os.listdir(path_group):
            path_subpop = os.path.join(path_group, subpop)
            
            # Skip non-directories
            if not os.path.isdir(path_subpop):
                continue
            
            for fname in os.listdir(path_subpop):
                if fname.lower().endswith('.csv'):
                    full = os.path.join(path_subpop, fname)
                    files_dict[f"{subpop}/{fname}"] = pd.read_csv(full)
        
        dic_distancias[group] = files_dict
    
    return dic_distancias
