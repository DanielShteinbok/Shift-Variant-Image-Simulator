import glob
import os
import numpy as np
import csv

class MetaMan:
    def __init__(self, meta_path):
        self.data = []
        with open(meta_path, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append(row)
#         self.shifts = {}
#         with open(meta_path, newline='', encoding='utf-8-sig') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 num = int(row["Field Number"])
#                 self.shifts[num] = [int(float(row["X image (px)"])), int(float(row["Y image (px)"]))]
    @property
    def shifts(self):
        """
        Get the [x,y] position of the center of the PSF in the image space, in pixels.
        Returns a dictionary, so shifts[fieldnum] = [x, y]
        the pixel values returned are integers
        """
        shifts = {}
        for row in self.data:
            num = int(row["Field Number"])
            # undid artificial inversion
            # besides, this inversion was wrong: it moved the PSF but did not invert it
            # resulting in "pincoushiony" PSF variation over image instead of "barrelly"
            shifts[num] = [int(float(row["X image (px)"])), int(float(row["Y image (px)"]))]
            #shifts[num] = [-int(float(row["X image (px)"])), -int(float(row["Y image (px)"]))]
        return shifts
    
    @property
    def field_origins(self):
        """
        Get the [x,y] position of the origin of the field in microns.
        x, y are floats. See shifts.help for usage
        """
        origins = {}
        for row in self.data:
            num = int(row["Field Number"])
            origins[num] = [float(row["X (mm)"])*1000, float(row["Y (mm)"])*1000]
            # invert the origins... this is a LIE! but a necessary one for now...
#             origins[num] = [float(row["X (mm)"])*-1000, float(row["Y (mm)"])*-1000]
        return origins
    

def pad_to_position(psf, centerpoint, img_dims):
    """
    zero-pad a PSF so its centerpoint ends up at the centerpoint coordinate
    relative to the center of the image
    
    Parameters:
        psf: np.ndarray
        centerpoint: tuple (int x, int y)
        img_dims: tuple (int height, int width)
    Returns:
        ndarray of shape img_dims
    """
    hdiff = img_dims[0] - psf.shape[0]
    wdiff = img_dims[1] - psf.shape[1]
    
    # In the future, these could be used in another way
    # Now, I'm just catching the error and discarding the image
    # A much better way would be to just put all of these in a single if statement with 'or' between them
    if np.math.floor(hdiff/2) - centerpoint[1] < 16:
        raise ValueError("PSF falls off the right side of image field")
    if np.math.ceil(hdiff/2) + centerpoint[1] < 16:
        raise ValueError("PSF falls off the left side of image field")
    if np.math.floor(wdiff/2) - centerpoint[0] < 16:
        raise ValueError("PSF falls off the bottom side of image field")
    if np.math.ceil(wdiff/2) + centerpoint[0] < 16:
        raise ValueError("PSF falls off the top side of image field")
    
    return np.pad(psf,
                              ((np.math.ceil(hdiff/2) + centerpoint[1], np.math.floor(hdiff/2) - centerpoint[1]),
                              (np.math.ceil(wdiff/2) + centerpoint[0], np.math.floor(wdiff/2) - centerpoint[0]))
                 )

def load_PSFs_csv(psfs_path, meta_path, img_dims):
    """
    load PSFs from a bunch of CSV files stored in a directory.
    The assumption is that all the PSFs will have a name like F19.csv
    where 19 is the field number. These should correspond to the field numbers in
    the meta file.
    
    The meta file is an excel csv file with the first row:
    Field Number,X (mm),Y (mm),X image (px),Y image (px) 
    
    Parameters:
        psfs_path: str, path to directory containing all the PSFs
        meta_path: str, path to the meta csv file
        img_dims: a tuple with two elements, signifying the dimensions of the expected image or PSF
        
    Returns:
        psfs: an ndarray of the PSFs.
    """
    # open up the meta csv file, append each row to a list:
#     meta_list = []
#     with open(meta_path, newline='', encoding='utf-8-sig') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             meta_list.append(row)
    metaman = MetaMan(meta_path)        
    
    # expect a directory with a bunch of PSFs therein as separate csvs;
    # list out all csv file names in the directory
    psf_paths = glob.glob(psfs_path.removesuffix('/') + '/F*')
    # iterate through that list,
    # open and append each to the psfs dictionary, key is the Field number
    psfs = [[]]
    for path in psf_paths:
        #psfs[0].append(np.loadtxt(path, delimiter=',', encoding='utf-8-sig'))
        fieldnum = int(os.path.basename(path).removeprefix("F").removesuffix(".csv"))
        unpadded_psf = np.loadtxt(path, delimiter=',')
        #centerpoint = (int(float(meta_list[fieldnum-1]["X image (px)"])), int(float(meta_list[fieldnum-1]["Y image (px)"])))
        centerpoint = metaman.shifts[fieldnum]
        try:
            padded_psf = pad_to_position(unpadded_psf, centerpoint, img_dims)
            psfs[0].append(padded_psf)
        except ValueError:
            print("caught error")
    # convert the psfs array to an np.ndarray
    return np.transpose(np.asarray(psfs), (2,3,1,0))
