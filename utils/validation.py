import numpy as np
from typing import Any


def verify_binary(mask: np.ndarray, mask_name: str) -> None:
    """
    Verifies whether a given mask is binary (contains only 0 and 1).

    Args:
        mask (np.ndarray): The mask to be verified.
        mask_name (str): The name of the mask, used for identification in the output.

    Returns:
        None: Prints a message indicating whether the mask is binary or not.
    """
    unique_values = np.unique(mask)
    
    # Check if the unique values in the mask are either 0 or 1
    if set(unique_values).issubset({0, 1}):
        print(f"{mask_name} is binary.")
    else:
        print(f"{mask_name} is NOT binary. Unique values found: {unique_values}")
