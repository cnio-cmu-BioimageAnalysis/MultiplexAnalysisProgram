
def verify_binary(mask, mask_name):
    """
    Verifies if a mask is binary (contains only 0 and 1).
    
    Parameters:
    - mask (numpy array): Mask to verify.
    - mask_name (str): Name of the mask for identification.
    """
    # Get the unique values in the mask
    import numpy as np
    unique_values = np.unique(mask)
    
    # Check if all unique values are either 0 or 1
    if set(unique_values).issubset({0, 1}):
        print(f"{mask_name} is binary.")
    else:
        print(f"{mask_name} is NOT binary. Unique values found: {unique_values}")