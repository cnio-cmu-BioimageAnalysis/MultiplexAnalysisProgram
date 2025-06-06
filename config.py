import os
from multiplex_pipeline.preprocessing.segmentation import post_process_mask, post_process_mask_closing

# User information
CNIO_USER = "jgagullo"

# Server base paths
BASE_SERVER_PATH = r"\\imgserver.cnio.es\IMAGES\CONFOCAL\IA\Projects\2024\2024_10_21_jgagullo\spatial analisis\qpath"
DATA_FOLDER = os.path.join(BASE_SERVER_PATH, "data", CNIO_USER)
EXPORT_DAPI_FOLDER = os.path.join(BASE_SERVER_PATH, "qpathprojects", CNIO_USER, "export_DAPI")
EXPORTED_MASKS_FOLDER = os.path.join(BASE_SERVER_PATH, "qpathprojects", CNIO_USER, "exportedmasks")

# Results directories
RESULTS_BASE_DIR = os.path.join("..", "results_spatial_analysis", CNIO_USER)
CELL_COUNT_OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, "cell_counting")
CELL_DENSITY_OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, "cell_density_area")
BOXPLOTS_DISTANCES_DIR = os.path.join(RESULTS_BASE_DIR, "boxplots_distances_to_mask")
DISTANCES_SUBPOP_DIR = os.path.join(RESULTS_BASE_DIR, "distances_between_mask_and_subpopulation")
DISTANCES_POPULATIONS_DIR = os.path.join(RESULTS_BASE_DIR, "distances_between_populations")

# Imaging constants
PIXEL_SIZE = 0.17  # micrometers
PIXEL_AREA = PIXEL_SIZE ** 2  # µm²
MASK_ALPHA = 0.5
DEFAULT_BRIGHTNESS = 10

# Channels and markers
CHANNELS_OF_INTEREST = [10, 13, 7, 17, 1, 0, 15, 5, 20, 21, 9, 19, 3]
MARKER_LABELS = {
    0: "FOXP3",
    1: "IFN-gamma",
    2: "CD20 - cyto",
    3: "HLA-DR",
    4: "CD279",
    5: "CD4",
    6: "A_Podoplan",
    7: "Ki67",
    8: "CD163",
    9: "CD11b",
    10: "Pan-Cytokeratin CK",
    11: "A_Actin",
    12: "CD31",
    13: "NGFR",
    14: "DAPI",
    15: "CD3",
    16: "CD56",
    17: "CD274",
    18: "CD45",
    19: "CD11c",
    20: "CD8a",
    21: "CD68",
}

# Mask creation settings
CK_SETTINGS = {
    "channel_index": 10,
    "user_scores": -1,
    "scaling_divisor": 10,
    "base_folder_path": EXPORTED_MASKS_FOLDER,
    "min_size": 2000,
    "max_hole_size": 10000,
    "mask_label": "CK",
    "mask_filename": "CK_mask_binary.tif",
    "post_process_funcs": [post_process_mask],
    "brightness_factor": None,
    "require_dapi": False,
}
NGFR_SETTINGS = {
    "channel_index": 13,
    "user_scores": {
        'ROI1.ome.tiff': 2.7, 'ROI2.ome.tiff': 2.7, 'ROI3.ome.tiff': 2.7,
        'ROI4.ome.tiff': 1.8, 'ROI5.ome.tiff': 1.8, 'ROI6.ome.tiff': 1.8,
        'ROI7.ome.tiff': 1.5, 'ROI8.ome.tiff': 1.2, 'ROI10.ome.tiff': 2.7,
        'ROI11.ome.tiff': 2.4, 'ROI12.ome.tiff': 2.7, 'ROI13.ome.tiff': 2.1,
    },
    "scaling_divisor": 3,
    "base_folder_path": EXPORTED_MASKS_FOLDER,
    "min_size": 200,
    "max_hole_size": 75000,
    "mask_label": "NGFR",
    "mask_filename": "NGFR_mask_binary.tif",
    "post_process_funcs": [post_process_mask_closing, post_process_mask],
    "brightness_factor": 25,
    "require_dapi": True,
}

# Intensity thresholds
INTENSITY_THRESHOLDS = {
    'is_positive_Pan_Cytokeratin_CK': 0,
    'mean_intensity_NGFR': 0.25,
    'is_positive_NGFR': 0,
    'mean_intensity_Ki67': 0,
    'mean_intensity_CD274': 2.5,
    'mean_intensity_IFN_gamma': 1.5,
    'mean_intensity_FOXP3': 2.25,
    'mean_intensity_CD3': 1.25,
    'mean_intensity_CD4': 1,
    'mean_intensity_CD8a': 2.5,
    'mean_intensity_CD68': 2,
    'mean_intensity_CD11b': 2,
    'mean_intensity_CD11c': 2,
    'mean_intensity_HLA_DR': 1.5,
}



CARACTERIZATION_COMBINATIONS = {
    'Tumor': lambda df: df['is_positive_Pan_Cytokeratin_CK_binary'] == 1,
    'Tumor NGFR+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 1),
    'Tumor NGFR-': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0),
    'Tumor NGFR+ Ki67+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 1) & (df['mean_intensity_Ki67_binary'] == 1),
    'Tumor NGFR- Ki67+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0) & (df['mean_intensity_Ki67_binary'] == 1),
    'Tumor NGFR+ CD274+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 1) & (df['mean_intensity_CD274_binary'] == 1),
    'Tumor NGFR- CD274+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0) & (df['mean_intensity_CD274_binary'] == 1),
    'Tumor NGFR+ IFN+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 1) & (df['mean_intensity_IFN_gamma_binary'] == 1),
    'Tumor NGFR- IFN+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0) & (df['mean_intensity_IFN_gamma_binary'] == 1),
    'Tumor NGFR+ HLA_DR+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0) & (df['mean_intensity_HLA_DR_binary'] == 1),
    'Tumor NGFR- HLA_DR+': lambda df: (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) & (df['mean_intensity_NGFR_binary'] == 0) & (df['mean_intensity_HLA_DR_binary'] == 1),
    
  
}


# Combinations for tumor infiltration
INFILTRATION_COMBINATIONS = {
    'Tumor CD3+ CD4+ FOXP3+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD4_binary'] == 1) &
        (df['mean_intensity_FOXP3_binary'] == 1)
    ),
    'Stroma CD3+ CD4+ FOXP3+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD4_binary'] == 1) &
        (df['mean_intensity_FOXP3_binary'] == 1)
    ),
    'Tumor CD3+ CD8+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1)
    ),
    'Stroma CD3+ CD8+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1)
    ),
    'Tumor CD68+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_CD68_binary'] == 1)
    ),
    'Stroma CD68+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_CD68_binary'] == 1)
    ),
    'Tumor CD11b+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_CD11b_binary'] == 1)
    ),
    'Stroma CD11b+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_CD11b_binary'] == 1)
    ),
    'Tumor CD11c+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_CD11c_binary'] == 1)
    ),
    'Stroma CD11c+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_CD11c_binary'] == 1)
    ),
    'Tumor MHCII+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['mean_intensity_HLA_DR_binary'] == 1)
    ),
    'Stroma MHCII+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 0) &
        (df['mean_intensity_HLA_DR_binary'] == 1)
    ),
}

# Combinations for NGFR-related infiltration
NGFR_INFILTRATION_COMBINATIONS = {
    'Tumor NGFR+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1)
    ),
    'Tumor NGFR-': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0)
    ),
    'Tumor NGFR+ CD3+ CD4+ FOXP3+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD4_binary'] == 1) &
        (df['mean_intensity_FOXP3_binary'] == 1)
    ),
    'Tumor NGFR- CD3+ CD4+ FOXP3+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD4_binary'] == 1) &
        (df['mean_intensity_FOXP3_binary'] == 1)
    ),
    'Tumor NGFR+ CD3+ CD8+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1)
    ),
    'Tumor NGFR- CD3+ CD8+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1)
    ),
    'Tumor NGFR+ CD3+ CD8+ IFN+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1) &
        (df['mean_intensity_IFN_gamma_binary'] == 1)
    ),
    'Tumor NGFR- CD3+ CD8+ IFN+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD3_binary'] == 1) &
        (df['mean_intensity_CD8a_binary'] == 1) &
        (df['mean_intensity_IFN_gamma_binary'] == 1)
    ),
    'Tumor NGFR+ CD68+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD68_binary'] == 1)
    ),
    'Tumor NGFR- CD68+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD68_binary'] == 1)
    ),
    'Tumor NGFR+ CD11b+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD11b_binary'] == 1)
    ),
    'Tumor NGFR- CD11b+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD11b_binary'] == 1)
    ),
    'Tumor NGFR+ CD11c+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_CD11c_binary'] == 1)
    ),
    'Tumor NGFR- CD11c+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_CD11c_binary'] == 1)
    ),
    'Tumor NGFR+ MHCII+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 1) &
        (df['mean_intensity_HLA_DR_binary'] == 1)
    ),
    'Tumor NGFR- MHCII+': lambda df: (
        (df['is_positive_Pan_Cytokeratin_CK_binary'] == 1) &
        (df['is_positive_NGFR_binary'] == 0) &
        (df['mean_intensity_HLA_DR_binary'] == 1)
    ),
}

# Condition column mapping
CONDITION_COLUMN_MAP = {
    'FOXP3_intensity': 'mean_intensity_FOXP3_binary',
    'IFN-gamma_intensity': 'mean_intensity_IFN_gamma_binary',
    'HLA-DR_intensity': 'mean_intensity_HLA_DR_binary',
    'CD4_intensity': 'mean_intensity_CD4_binary',
    'Ki67_intensity': 'mean_intensity_Ki67_binary',
    'CD163_intensity': 'mean_intensity_CD163_binary',
    'CD11b_intensity': 'mean_intensity_CD11b_binary',
    'CD3_intensity': 'mean_intensity_CD3_binary',
    'CD274_intensity': 'mean_intensity_CD274_binary',
    'CD11c_intensity': 'mean_intensity_CD11c_binary',
    'CD8a_intensity': 'mean_intensity_CD8a_binary',
    'CD68_intensity': 'mean_intensity_CD68_binary',
    'CK_mask': 'is_positive_Pan_Cytokeratin_CK_binary',
    'NGFR_mask': 'is_positive_NGFR_binary',
    'NGFR_intensity': 'mean_intensity_NGFR_binary',
}

# Subpopulation definitions
SUBPOPULATIONS = {
    "Tregs": ["CD3_intensity+", "CD4_intensity+", "FOXP3_intensity+"],
    "T CD8+": ["CD3_intensity+", "CD8a_intensity+"],
    "T CD8+ Activator": ["CD3_intensity+", "CD8a_intensity+", "IFN-gamma_intensity+"],
    "Macrophages CD68+": ["CD68_intensity+"],
    "DC CD11b+": ["CD11b_intensity+"],
    "DC CD11c+": ["CD11c_intensity+"],
    "DC HLA-DR+": ["HLA-DR_intensity+"],
}

# Conditions for two-population distances (A group)
SUBPOPULATION_A_POSITIVE = ["CK_mask+", "NGFR_intensity+"]
SUBPOPULATION_A_NEGATIVE = ["CK_mask+", "NGFR_intensity-"]

# CK-only subpopulations for two-population distances
CK_SUBPOPULATIONS = {
    "CK+ Tregs": ["CK_mask+", "CD3_intensity+", "CD4_intensity+", "FOXP3_intensity+"],
    "CK+ T CD8+": ["CK_mask+", "CD3_intensity+", "CD8a_intensity+"],
    "T CD8+ Activator": ["CD3_intensity+", "CD8a_intensity+", "IFN-gamma_intensity+"],
    "CK+ Macrophages": ["CK_mask+", "CD68_intensity+"],
    "CK+ DC CD11b+": ["CK_mask+", "CD11b_intensity+"],
    "CK+ DC CD11c+": ["CK_mask+", "CD11c_intensity+"],
    "CK+ DC HLA-DR+": ["CK_mask+", "HLA-DR_intensity+"],
}

# Shading colors for heatmaps
SHADING_COLORS = {
    "CK_mask": (1.0, 0.0, 0.0),  # red
    "NGFR_mask": (1.0, 0.0, 1.0),  # magenta
}

# ROIs to analyze
ROIS_TO_ANALYZE = ["roi1"]