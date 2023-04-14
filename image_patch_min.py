import numpy as np

def find_minsum_patch(image, patch_size):
    # Calculate number of patches in each dimension
    n_patches_h = image.shape[0] // patch_size
    n_patches_w = image.shape[1] // patch_size

    # Initialize variables for minimum value and its index
    min_val = np.inf
    min_idx = None

    # Loop through all patches
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Get patch indices
            h_start = i * patch_size
            h_end = h_start + patch_size
            w_start = j * patch_size
            w_end = w_start + patch_size
            
            # Extract patch and calculate minimum value
            patch = image[h_start:h_end, w_start:w_end]
            patch_min = np.sum(patch)
            
            # Update minimum value and its index if necessary
            if patch_min < min_val:
                min_val = patch_min
                min_idx = (i, j)
                
    # Print minimum value and its index
    # print("Minimum value:", min_val)
    # print("Minimum patch index:", min_idx[0]*patch_size+patch_size//2, min_idx[1]*patch_size+patch_size//2)
    center_x = min_idx[0]*patch_size+patch_size//2
    center_y = min_idx[1]*patch_size+patch_size//2

    return center_x, center_y