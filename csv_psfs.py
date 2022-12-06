import glob
import numpy as np

# for sorting
import os

def load_from_dir(psfs_path, sort_psfs=False):
    """
    load a stack of PSFs as all the CSV files from the specified directory
    
    psfs_path: the path to the directory with all the CSV files
    
    Returns: np.ndarray of shape (width, height, number_of_psfs)
    """
    psf_paths = glob.glob(psfs_path.removesuffix('/') + '/*')
    # iterate through that list,
    # open and append each to the psfs array,
    
    # TODO: sort psf_paths by name of PSF
    if sort_psfs:
        psf_paths.sort(key = lambda path: int(os.path.basename(path).removeprefix("F").removesuffix(".csv")))
    psfs = []
    for path in psf_paths:
        # need the next line when dealing with Excel CSVs, which have a signature byte at beginning of contents
        #psfs.append(np.loadtxt(path, delimiter=',', encoding='utf-8-sig'))
        # with the ZOS-API script I wrote, no signature at the beginning
        psfs.append(np.loadtxt(path, delimiter=','))
    # convert the psfs array to an np.ndarray
    psfs = np.transpose(np.asarray(psfs), (1,2,0))
    return psfs

def load_from_dir_index(psfs_path):
    """
    do something similar to load_from_dir, but instead of sorting names initially, return a list of field numbers
    that corresponds to layers in the returned stack of psfs
    
    Returns: ndarray, list
    The former is the set of PSFs of shape (width, height, number_of_psfs)
    The latter is of length number_of_psfs
    """
    psf_paths = glob.glob(psfs_path.removesuffix('/') + '/*')
    psfs = []
    indices = []
    for path in psf_paths:
        # need the next line when dealing with Excel CSVs, which have a signature byte at beginning of contents
        #psfs.append(np.loadtxt(path, delimiter=',', encoding='utf-8-sig'))
        # with the ZOS-API script I wrote, no signature at the beginning
        psfs.append(np.loadtxt(path, delimiter=','))
        indices.append(int(os.path.basename(path).removeprefix("F").removesuffix(".csv")))
    # convert the psfs array to an np.ndarray
    psfs = np.transpose(np.asarray(psfs), (1,2,0))
    return psfs, indices

def pad_as_center(initial_psfs, height, width):
    '''
    zero-pad a stack of PSFs to the desired width and height, so they stay in the center
    '''
    hdiff = height - initial_psfs.shape[0]
    wdiff = width - initial_psfs.shape[1]
    
    if hdiff < 0 or wdiff < 0:
        raise ValueError("initial PSF larger than expected size")
    # Pad PSF with zeros to the specified height and width so that it ends up in the middle
    elif hdiff > 0 or wdiff > 0:
        initial_psfs = np.pad(initial_psfs,
                              ((np.math.ceil(hdiff/2), np.math.floor(hdiff/2)),
                              (np.math.ceil(wdiff/2), np.math.floor(wdiff/2)),
                              (0,0)))
    return initial_psfs

# def pad_to_position(initial_psfs, height, width, center_x=0, center_y=0):
#     '''
#     zero-pad a stack of PSFs to the desired width and height, so the center of the PSF
#     ends up at a particular point. To be used with the center coordinates given by Zemax
#     (when I get that fixed).
#     Default should be the same behavior as pad_as_center
#     '''
#     hdiff = height - initial_psfs.shape[0]
#     wdiff = width - initial_psfs.shape[1]
    
#     if hdiff < 0 or wdiff < 0:
#         raise ValueError("initial PSF larger than expected size")
#     # Pad PSF with zeros to the specified height and width so that it ends up in the middle
#     elif hdiff > 0 or wdiff > 0:
#         initial_psfs = np.pad(initial_psfs,
#                               ((np.math.ceil(hdiff/2) + int(center_y), np.math.floor(hdiff/2) - int(center_y)),
#                               (np.math.ceil(wdiff/2) + int(center_x), np.math.floor(wdiff/2) - int(center_x)),
#                               (0,0)))
#     return initial_psfs