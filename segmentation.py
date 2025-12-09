import skimage
from scipy import ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt

def segment_cells(image, thresh=0.99):
    mask = image < (thresh * np.max(image))  # Note: Most images have a saturated white background

    plt.imshow(mask)
    
    mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(30))
    mask = skimage.morphology.remove_small_holes(mask, 200)
    
    
    distance = ndi.distance_transform_edt(mask)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask, threshold_abs=(0.5 * np.max(distance)), min_distance=75)
    mask_marker = np.zeros(distance.shape, dtype=bool)
    mask_marker[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_marker)
    labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
    
    labels = skimage.segmentation.clear_border(labels)        
    labels = skimage.morphology.remove_small_objects(labels, 200)

    # Do a global threshold to determine dark/light threshold
    thresh_cell = skimage.filters.threshold_otsu(image[labels > 0])
   
    inner_cell_mask = image >= thresh_cell
    
    inner_cell_labels = labels.copy()
    inner_cell_labels[~inner_cell_mask] = 0

    return (labels, inner_cell_labels)