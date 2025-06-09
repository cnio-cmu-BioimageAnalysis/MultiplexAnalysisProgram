import re
from typing import Dict, Optional


def extract_roi_number(filename: str) -> Optional[str]:
    """
    Extracts the ROI number from the image filename.

    Example:
        'ROI1.ome.tiff' -> '1'

    Args:
        filename (str): The filename from which the ROI number is to be extracted.

    Returns:
        Optional[str]: The extracted ROI number as a string, or None if not found.
    """
    match = re.search(r"ROI(\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def invert_dict(subpop_data: Dict) -> Dict:
    """
    Inverts a nested dictionary such that the new keys are the ROIs.

    The function re-arranges the given nested dictionary, where each subpopulation's 
    data (DataFrame) will be grouped by ROI. For example, it changes the structure from 
    {subpopulation: {roi: dataframe}} to {roi: {subpopulation: dataframe}}.

    Args:
        subpop_data (Dict): A nested dictionary containing subpopulation data.

    Returns:
        Dict: A dictionary where each key is an ROI, and each value is another dictionary 
              with subpopulation names as keys and their corresponding DataFrames as values.
    """
    roi_dict = {}
    for subpop, roi_data in subpop_data.items():
        for roi, df in roi_data.items():
            if roi not in roi_dict:
                roi_dict[roi] = {}
            roi_dict[roi][subpop] = df
            print(f"Inverting: subpop {subpop} for ROI {roi}")
    return roi_dict
