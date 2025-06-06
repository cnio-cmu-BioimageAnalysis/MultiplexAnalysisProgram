def extract_roi_number(filename):
    """
    Extracts the ROI number from the image name.
    Example: 'ROI1.ome.tiff' -> '1'
    """
    import re
    match = re.search(r"ROI(\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def invert_dict(subpop_data):
    """
    Invierte el diccionario anidado para que la nueva llave sea el ROI.
    Retorna: {roi: {subpop: dataframe, ...}, ...}
    """
    roi_dict = {}
    for subpop, roi_data in subpop_data.items():
        for roi, df in roi_data.items():
            if roi not in roi_dict:
                roi_dict[roi] = {}
            roi_dict[roi][subpop] = df
            print(f"Invirtiendo: subpop {subpop} para ROI {roi}")
    return roi_dict
