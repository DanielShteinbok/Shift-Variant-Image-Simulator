import csv_psfs
import svd_model as svm
import numpy as np
import scipy
from numba import jit

import load_PSFs

def load_metaman(psf_meta_path):
    return load_PSFs.MetaMan(psf_meta_path)

def find_pixel_on_obj(x, y, img_dims, obj_dims):
    """
    It generally makes sense that the object space is separate from the image space,
    so it makes sense that they can be in different units in general and can be represented
    by matrices with different dimensions. This is how everything is presented in other functions.

    However, when we want to deal with images, we want to deal with a finite number of pixels.
    It is convenient for this to be the same number of pixels in both the object and the image.

    If that assumption is true (both images have the same pixel dimensions), then it is possible
    to relate a point in the object space to a pixel in the object space by this function.

    Parameters:
        x: the x-coordinate in object space
        y: the y-coordinate in object space
        obj_dims: the object field-of-view, as (y, x)
        img_dims: the pixel dimensions of the image, as (y, x)
    """
    return (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))


def generate_model(psf_directory, psf_meta_path, img_dims, obj_dims):
    """
    Generate the eigen-PSFs and weights for the entire field of view.

    Parameters:
        psf_directory: str, path to directory with the PSF CSV files therein
        psf_meta_path: str, path to meta csv file which describes the locations and shifts of the PSFs
        img_dims: (int height, int width), the dimensions of the sensor, in pixels
        obj_dims: (int height, int width), the dimensions of the field of view, in microns
    Returns:
        h: the eigen-PSFs
        weights: the weights at each pixel of the image
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # load the metaman:
    metaman = load_PSFs.MetaMan(psf_meta_path)

    # get the dictionary of field origins once to save computation
    field_origins = metaman.field_origins
    shifts = metaman.shifts

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    #find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    # we care about the difference between the placement of the origin and the shift position
    #origins_pixel = {k: find_pixel_on_obj(*v) for k, v in metaman.field_origins}
    origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}
    shift_from_origin = {}
    for index in indices:
        shift_from_origin[index] = (shifts[index][0] - origins_pixel[index][0], shifts[index][1] - origins_pixel[index][1])
        # since everything given by metaman is [x,y] we need to reverse to match numpy and linear algebra
        #shift_from_origin[index] = (shifts[index][1] - origins_pixel[index][1], shifts[index][0] - origins_pixel[index][0])

    # dictionary comprehension for what's commented out above
    #shift_from_origin = {index: (shifts[index][0] - origins_pixel[index][0], shifts[index][1] - origins_pixel[index][1]) for index in indices}

    # pad the PSFs based on the difference between the the calculated origin pixel of the field
    # and the shift of the PSF, all from the metaman
    padded_psfs = []
    for psf_index in range(unpadded_psfs.shape[-1]):
        padded_psfs.append(load_PSFs.pad_to_position(unpadded_psfs[:,:,psf_index], shift_from_origin[indices[psf_index]], img_dims))

    psfs_ndarray = np.transpose(np.asarray(padded_psfs), (1,2,0))

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    rank = 28
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    comps, weights_interp=svm.calc_svd_indexed(psfs_ndarray, origins_pixel, indices, rank)
    weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))

    return h, weights

def generate_and_save(psf_directory, psf_meta_path, img_dims, obj_dims, h_save_path, weights_save_path):
    h, weights = generate_model(psf_directory, psf_meta_path, img_dims, obj_dims)
    save_generated(h, weights, h_save_path, weights_save_path)

def save_generated(h, weights, h_save_path, weights_save_path):
    h_dict = {"array_out": h}
    scipy.io.savemat(h_save_path, h_dict)
    weights_dict = {"array_out": weights}
    scipy.io.savemat(weights_save_path, weights_dict)

# TODO: for warp-convolve architecture, whereby we first multiply pixels by weights, then warp the resulting image in spatial domain,
# then convolve the warped image with the PSFs in the Fourier domain, we need some stuff:
# we need to produce, save and later load an entire shifts array, which is 3-dimensional: e.g. shape=(800, 1280, 2) where the last dimension
# separates the positions to which to shift this point in the X-direction and Y-direction.
# Also, need a version of the above-given one-shot SVD methods for generation and saving with only center-padding.
# Also, need an altered version of the simulator which implements this warp-convolve algorithm.


# TODO: we actually DON'T WANT to resort to warp-convolve quite yet, want to take mastermat approach first with a sparse matrix.
# This involves:
# * Perform SVD etc with NO padding
# ** This includes the interpolation to fill the whole sensor area with weights as before
# * Ravel each of the kernels
# * ~Slice out the rows of the kernels that only contain values close to zero~
# * Make the matrix of kernels sparse
# * Matrix-multiply the set of weights for a pixel by this matrix of kernels, add the information about the non-zero values and their indices to special lists.
# ** Use these special lists to make the big sparse matrix, which we'll use later for accurate image simulation


#def generate_unpadded(psf_directory, psf_meta_path, img_dims, obj_dims):
def generate_unpadded(psf_directory, metaman, img_dims, obj_dims, method="nearest"):
    """
    Generate unpadded eigen-PSFs, and weights for them covering the entire field
    Parameters:
        psf_directory: str, path to directory with the PSF CSV files therein
        psf_meta_path: str, path to meta csv file which describes the locations and shifts of the PSFs
        img_dims: (int height, int width), the dimensions of the sensor, in pixels
        obj_dims: (int height, int width), the dimensions of the field of view, in microns
    Returns:
        h: the eigen-PSFs, shape = e.g. (32,32,28)
        weights: the weights at each pixel of the image, shape = e.g. (800, 1280, 28)
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # added this line to deal with nan-filled PSFs produced by Zemax
    unpadded_psfs[np.isnan(unpadded_psfs)] = 0

    # load the metaman:
    # for this function, we're just going to be passing the metaman directly in.
    # We want to have a high-level function that does things like instantiate it,
    # then call this function and others to orchestrate the "Mastermat" approach.
    #metaman = load_PSFs.MetaMan(psf_meta_path)

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    #find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    #rank = 28
    rank = unpadded_psfs.shape[2] - 1
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    comps, weights_interp=svm.calc_svd_indexed_sized(unpadded_psfs, origins_pixel, indices, rank, img_dims, method=method)
    weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))

    return h, weights

@jit(nopython=True)
def interp_grid(points, values, grid_dims=(32,32)):
    """
    linearly interpolate the values at points xi,
    kind of similar to griddata.
    However, points must be a 1D complex-valued array!
    The real component represents x-value, the imaginary component represents y-value.
    NOTE: the grid must be reularly-spaced and square.
    returns the interpolated grid of values
    """
    # triangulate each of the points we are interested in
    # (find the nearest three points on the grid)
    # and then do a linear barycentric interpolation

    # First, the triangulation is done as follows:
    # Round each of the real and imaginary components
    # of points to the nearest whole, which will give us
    # the coordinates of the nearest grid point.
    # Then, take one step in each of x and y in the opposite direction of rounding
    # to get the other two points.

    # IMPLEMENTING:
    # find nearest point:
    # still just complex values
    nearest_point = np.rint(points)

    # find the direction from the nearest point to the point itself
    # non-normalized direction:
    non_normed = points - nearest_point
    # should return ndarray where each element is one of (1 + 1j, 1 -1j, -1 + 1j, -1 -1j)
    # we need to handle cases where the points neatly lie on the gridline
    # for those cases, it doesn't matter where we step; we just need to choose a direction
    # arbitrarily choose positive
    step_dir = non_normed*1
    step_dir[np.real(step_dir) == 0] += 1
    step_dir[np.imag(step_dir) == 0] += 1j
    #step_dir = np.real(non_normed)/np.absolute(np.real(non_normed)) +\
            #1j*np.imag(non_normed)/np.absolute(np.imag(non_normed))
    step_dir = np.real(step_dir)/np.absolute(np.real(step_dir)) +\
            1j*np.imag(step_dir)/np.absolute(np.imag(step_dir))


    # real(step_dir) = x_1 - x_3 = x_1 - x_2
    # imag(step_dir) = y_2 - y_3 = y_2 - y_1
    # performing the interpolation
    grid = np.zeros(grid_dims)
    w_xstep = (np.imag(step_dir)*np.real(non_normed))/(np.imag(step_dir)*np.real(step_dir))
    w_ystep = np.real(step_dir)*np.imag(non_normed)/(np.imag(step_dir)*np.real(step_dir))
    w_nearest = 1 - w_xstep - w_ystep

    #grid[(np.imag(nearest_point)).astype(np.int64), (np.real(nearest_point + step_dir)).astype(np.int64)] \
            #+= w_xstep*values
    #grid[np.imag(nearest_point + step_dir).astype(np.int64), np.real(nearest_point).astype(np.int64)] \
            #+= w_ystep*values
    #grid[np.imag(nearest_point).astype(np.int64), np.real(nearest_point).astype(np.int64)] \
            #+= w_nearest*values

    # above code works fine, but numba can't handle it

    for i in range(values.shape[0]):
        if np.real(non_normed[i]) != 0:
            grid[(np.imag(nearest_point)).astype(np.int64)[i], (np.real(nearest_point + step_dir)).astype(np.int64)[i]] \
                += w_xstep[i]*values[i]

        if np.imag(non_normed[i]) != 0:
            grid[np.imag(nearest_point + step_dir).astype(np.int64)[i], np.real(nearest_point).astype(np.int64)[i]] \
                += w_ystep[i]*values[i]

        grid[np.imag(nearest_point).astype(np.int64)[i], np.real(nearest_point).astype(np.int64)[i]] \
                += w_nearest[i]*values[i]
    return grid

@jit(nopython=True)
def rotate_unpadded_psfs(unpadded_psfs, origins_pixel, reverse=False):
    """
    Rotate the unpadded PSFs appropriately based on their location in the FOV,
    resample to get the resulting 3D array with the first two axes in cartesian coordinates
    but in rotated axes. Returned is sampled with same number of points as input.
    Parameters:
        unpadded_psfs: 3D ndarray where the first two axes are the cartesian axes, and the last
            separates PSFs from one another
        origins_pixel: ndarray of shape (m,2), where each row represents
            the [x,y] coordinates of the center of the PSF in the FOV.
            Must be already sorted to match the unpadded PSFs

    Returns:
        rotated_psfs: the PSFs, rotated and resampled and in polar coordinates

    NOTE: resampling is done through cubic spline interpolation
    """
    # 1. create two meshgrids respectively for x- and y- indices of the PSF
    #   Alternatively this can be just two 1D ndarrays that we use for the next step
    # 2. create a complex-valued 2D ndarray by combining these
    # 3. Use the complex-valued ndarray to make one 2D array for r and another for theta
    # 4. Shift the angle of everything by just adding to the angular array elementwise
    # 5. Now, instead of having indices on the range of (-pi, pi], you'll have (-pi+x, pi+x]
    #       so if you want to convert to the former range, you'll have to
    #       add 2*pi to everything less than -pi, and subtract 2*pi from everything exceeding pi
    # You will want to perform step 5 because later you'll take for granted that the coordinates
    # are the same across PSFs when you do the SVD

    #FIXME: shift each of the x- and y- indices to make the origin the center of the image!!!


    # Possibly an alternative to converting to angle and then shifting would be to
    # pointwise-multiply the entire complex 2D array by an appropriate complex exponential
    x_indices_int = np.arange(unpadded_psfs.shape[1])
    # y_indices will be a column vector because that is necessary to make the a-la complex meshgrid below
    y_indices_int = np.arange(unpadded_psfs.shape[0]).reshape((unpadded_psfs.shape[0], 1))

    # shift the indices (so the origin is in the center of the image)
    # the indices will end up floats, because the original dimensions are typically even (32 by 32).
    # hence I am creating a new variable. This should be investigated by means of pdb
    x_indices = x_indices_int - x_indices_int[-1]/2
    y_indices = y_indices_int - y_indices_int[-1,0]/2

    # complex representation of the x- and y-indices
    #inds_complex = np.ones((unpadded_psfs.shape[0], unpadded_psfs.shape[1]))*x_indices + 1j*y_indices.transpose()
    inds_complex = np.ones((unpadded_psfs.shape[0], unpadded_psfs.shape[1]))*x_indices + 1j*y_indices

    # figure out the angles of the lines that pass through the optical axis and the center of each PSF in radians
    # first, get the complex representation of the rearranged origins_pixel,
    # then by Euler's formula we know:
    # x + j*y = r*cos(theta) + j*r*sin(theta) = r*exp(j*theta)
    # so we can just divide by the magnitude and then use this to pointwise-multiply the indices to shift them
    #origins_complex = np.empty(len(indices))
    #for i in range(len(indices)):
        #origins_complex[i] = origins_pixel[indices[i]][0] + 1j*origins_pixel[indices[i]][1]

    #origins_complex = np.empty(len(origins_pixel))
    #for i in range(len(origins_pixel))
        #origins_complex[i] = origins_pixel[i][0] + 1j*origins_pixel[i][1]
    origins_complex = np.empty(origins_pixel.shape[0])
    #origins_complex[:] = origins_pixel[:,0] + 1j*origins_pixel[:,1]
    origins_complex = origins_pixel[:,0] + 1j*origins_pixel[:,1]
    # shift the angles by multiplying these compex indices by a complex exponential
    # see the whole thing above
    # this is r*cos(theta) + j*r*sin(theta) = r*exp(j*theta)
    # so exp(j*theta) = (r*cos(theta) + j*r*sin(theta))/r = (x + jy)/r
    shifted_inds_complex = np.empty((inds_complex.shape[0],inds_complex.shape[1], origins_complex.shape[0]), dtype=np.complex128)

    # create a rotator, which is a complex exponential by which we multiply our complex indices to rotate the image
    # by Euler's formula we can consider the complex origin to be a phasor, so dividing by its magnitude
    # gives a unit phasor. Multiplying by this unit phasor adds a phase.
    rotator = np.empty_like(origins_complex)
    rotator[np.absolute(origins_complex) == 0] = 1
    rotator[np.absolute(origins_complex)>0] = origins_complex[np.absolute(origins_complex)>0]/np.absolute(origins_complex[np.absolute(origins_complex)>0])

    # bandaid solution to the problem of nan rotator at the center of the image
    # there, we want no rotation: set rotator to 1
    #rotator[np.isnan(rotator)] = 1

    if reverse:
        # if we're going in the reverse direction, we need to subtract the angle
        # this means we negate our exponent, which is equivalent to taking a reciprocal.
        # this is an elementwise operation.
        rotator = 1/rotator

    for r in range(origins_complex.shape[0]):
        shifted_inds_complex[:,:,r] = inds_complex*rotator[r]

    # now, the indices we want to interpolate at are just the inds_complex that we started with
    # griddata can only work with 2D arrays, so we'll need to use a dreaded for-loop through the third axis
    rotated_unpadded_psfs = np.empty_like(unpadded_psfs)

    # undo shifting to origin:
    shifted_inds_complex = shifted_inds_complex + x_indices_int[-1]/2 + 1j*y_indices_int[-1,0]/2
    inds_complex = inds_complex + x_indices_int[-1]/2 + 1j*y_indices_int[-1,0]/2

    # create the mask of points that are clipped (which fall beyond the square we are trying to sample in)
    clipped_points = (np.real(shifted_inds_complex) < 0) | (np.real(shifted_inds_complex) > unpadded_psfs.shape[0]) | \
        (np.imag(shifted_inds_complex) < 0) | (np.imag(shifted_inds_complex) > unpadded_psfs.shape[1])

    # we have N PSFs, so for n in N
    for n in range(unpadded_psfs.shape[2]):
        # need ravelled clipped points mask for nth PSF
        ravelled_clipped_points = np.ravel(clipped_points[:,:,n])
        # anchor points and indices must be ravelled
        # format these things correctly
        values = np.ravel(unpadded_psfs[:,:,n])[~ravelled_clipped_points]
        points_complex = np.ravel(shifted_inds_complex[:,:,n])[~ravelled_clipped_points]
        #points_2d = np.empty((points_complex.shape[0], 2))
        #points_2d[:,1] = np.real(points_complex)
        #points_2d[:,0] = np.imag(points_complex)

        # similarly, we want to format the locations at which we want to interpolate
        #interp_points_complex = np.ravel(inds_complex)
        #xi = np.empty((interp_points_complex.shape[0], 2))
        #xi[:,1] = np.real(interp_points_complex)
        #xi[:,0] = np.imag(interp_points_complex)
        # points produced will be ravelled
        # FIXME before interpolating, undo the shifting-to-the-origin that we did before, so that we are able to snap to coords
        # in my version of the interpolation function.

        #ravelled_points = scipy.interpolate.griddata(points_2d, values, xi)
        # Below, stuff breaks. This is because we have less than 1024 points
        #rotated_unpadded_psfs[:,:,n] = np.reshape(ravelled_points, rotated_unpadded_psfs[:,:,n].shape)
        rotated_unpadded_psfs[:,:,n] = interp_grid(points_complex, values, grid_dims=unpadded_psfs[:,:,n].shape)
    return rotated_unpadded_psfs

def generate_unpadded_rotated(psf_directory, metaman, img_dims, obj_dims, method="nearest"):
    """
    A sort of copy of generate_unpadded, except that it first rotates the PSFs
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # added this line to deal with nan-filled PSFs produced by Zemax
    unpadded_psfs[np.isnan(unpadded_psfs)] = 0

    # load the metaman:
    # for this function, we're just going to be passing the metaman directly in.
    # We want to have a high-level function that does things like instantiate it,
    # then call this function and others to orchestrate the "Mastermat" approach.
    #metaman = load_PSFs.MetaMan(psf_meta_path)

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    #find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}

    origins_sorted = np.empty((len(indices),2))
    for i in range(len(indices)):
        origins_sorted[i, :] = np.asarray(origins_pixel[indices[i]])

    rotated_psfs = rotate_unpadded_psfs(unpadded_psfs, origins_sorted)

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    #rank = 28
    rank = unpadded_psfs.shape[2] - 1
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    comps, weights_interp=svm.calc_svd_indexed_sized(rotated_psfs, origins_pixel, indices, rank, img_dims, method=method)
    weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))

    return h, weights


#def generate_unpadded(psf_directory, psf_meta_path, img_dims, obj_dims):
def generate_unpadded_nosvd_nonorm(psf_directory, metaman, img_dims, obj_dims, method="nearest"):
    """
    Generate unpadded eigen-PSFs, and weights for them covering the entire field
    Parameters:
        psf_directory: str, path to directory with the PSF CSV files therein
        psf_meta_path: str, path to meta csv file which describes the locations and shifts of the PSFs
        img_dims: (int height, int width), the dimensions of the sensor, in pixels
        obj_dims: (int height, int width), the dimensions of the field of view, in microns
    Returns:
        h: the eigen-PSFs, shape = e.g. (32,32,28)
        weights: the weights at each pixel of the image, shape = e.g. (800, 1280, 28)
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # added this line to deal with nan-filled PSFs produced by Zemax
    unpadded_psfs[np.isnan(unpadded_psfs)] = 0

    # load the metaman:
    # for this function, we're just going to be passing the metaman directly in.
    # We want to have a high-level function that does things like instantiate it,
    # then call this function and others to orchestrate the "Mastermat" approach.
    #metaman = load_PSFs.MetaMan(psf_meta_path)

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    #find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    #rank = 28
    rank = unpadded_psfs.shape[2]
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    #comps, weights_interp=svm.calc_svd_indexed_sized(unpadded_psfs, origins_pixel, indices, rank, img_dims, method=method)
    #weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    #weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    #h=comps/np.linalg.norm(np.ravel(comps))
    #h = psfs_ndarray
    weights = np.eye(rank)
    weights_interp = np.zeros((img_dims[0], img_dims[1],rank));

    return h, weights


def interpolate_shifts(metaman, img_dims, obj_dims, method="cubic"):
#def interpolate_shifts(metaman, img_dims, obj_dims):
    """
    Gets shifts from the metaman at the particular points of the field origins,
    and interpolates the shifts at all other points on the image based on that.
    Then, converts the (x,y) coordinate pairs into a single value each,
    encoding all information through modular arithmetic.
    Ravels the result into a vector with a number of entries
    equal to the product of img_dims.

    Parameters:
        img_dims: pixel dimensions of the image, of shape (y,x)
        obj_dims: micron dimensions of the object, of shape (y,x)
        metaman: instance of tensorflow/load_PSFs.MetaMan, which manages the field origin and shift information
    Returns:
        img_dims[0]*img_dims[1]-entry vector, where the k-th entry corresponds to the k-th pixel on the image
        and if the value thereof is v, then:
            v // img_dims[1] = the row on the image on which the PSF produced by this point would be centerd
            v % img_dims[1] = the column on the image on which the PSF produced by this point would be centerd
        UPDATE:
        now returns a 2D vector of whape (img_dims[0]*img_dims[1], 2) which contains the (y, x) shifts for each point
    """

    # SKETCH OF SOLUTION:
    # create two interpolated grids, respectively corresponding to y-shift and x-shift
    # multiply the one representing y-shift by img_dims[1], then add this to the x-shift
    # ravel the resulting matrix

    # interpolation should be sinusoidal (?)
    # to be more precise: the shift is related to the angle over to FOV by:
    # shift = r(\tan\theta - \sin\theta)
    # In the case of the curved object plane, let's say there was no shift,
    # and the distance from origin on the curved plane was proportional to the angle
    # That implies that at a distance from the origin x on the flat object plane,
    # given r being the radius at which the PSF was aligned with the origin
    # (specifically, 3.627 mm)
    # we should get:
    # shift = rx/sqrt(r^2 - x^2) - x
    # ignoring all inverting optics.
    # It could make sense to actually check this, and then just do everything
    # from theory rather than interpolation of empirical data.

    # For now, could do cubic interpolation.
    # TODO: look for bounds on error due to cubic interpolation of shift
    # through taylor expansion based on distance between field origins

    # creating the meshgrid to represent the image axes:
    xq = np.arange(-img_dims[1]/2,img_dims[1]/2);
    yq = np.arange(-img_dims[0]/2,img_dims[0]/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    # create list of x-shifts, y_shifts, x_origin, y_origin
    x_shifts = []
    y_shifts = []
    x_origin = []
    y_origin = []
    shifts = metaman.shifts
    origins = metaman.field_origins
    for key in shifts.keys():
        #x_shifts.append(shifts[key][0])
        # we need x_shifts and y_shifts to all be positive,
        # to represent points on the image plane
        x_shifts.append(shifts[key][0] + img_dims[1]/2)
        y_shifts.append(shifts[key][1] + img_dims[0]/2)
        # origins can be 0 at center of image, because the meshgrids
        # range over (-dim/2, dim/2)
        #x_origin.append(origins[key][0])
        #y_origin.append(origins[key][1])
        this_origin = find_pixel_on_obj(*(origins[key]), img_dims, obj_dims)
        x_origin.append(this_origin[0])
        y_origin.append(this_origin[1])

    # create the matrices for interpolation
    # may not need this, unlike the case for calc_svd
    #x_shifts_int = np.zeros((img_dims[0], img_dims[1]))
    #y_shifts_int = np.zeros((img_dims[0], img_dims[1]))

    # perform the interpolations themselves
    # we want to do a cubic interpolation, but then want to replace
    # the regions outside the convex hull with nearest-neighbor interpolations.

    # FIXME: nearest neighbor interpolation here causes bright spots
    x_int = scipy.interpolate.griddata((y_origin, x_origin), x_shifts, (Yq,Xq),method=method)
    #x_int = scipy.interpolate.griddata((y_origin, x_origin), x_shifts, (Yq,Xq),method='cubic')
    #x_int_nearest = scipy.interpolate.griddata((y_origin, x_origin), x_shifts, (Yq,Xq),method='nearest')
    #x_int[np.isnan(x_int)] = x_int_nearest[np.isnan(x_int)]

    #y_int = scipy.interpolate.griddata((y_origin, x_origin), y_shifts, (Yq,Xq),method='cubic')
    y_int = scipy.interpolate.griddata((y_origin, x_origin), y_shifts, (Yq,Xq),method=method)
    #y_int_nearest = scipy.interpolate.griddata((y_origin, x_origin), y_shifts, (Yq,Xq),method='nearest')
    #y_int[np.isnan(y_int)] = y_int_nearest[np.isnan(y_int)]

    # x_int and y_int are the interpolated shifts for each point in respectively the x-
    # and y- direction.
    # We now need to consolidate these via modular arithmetic.
    #modular_shifts = x_int + img_dims[1]*y_int

    # turn modular shift values of all points that are shifted out of the FOV to our magic number
    #strong_negative = -1025 # < -32*32
    #modular_shifts[(x_int < 0) * (x_int > img_dims[1])] = strong_negative # we know to filter these points out

    #return np.ravel(modular_shifts)
    shifts_out = np.zeros((2, img_dims[0], img_dims[1]))
    shifts_out[0,:,:] = y_int
    shifts_out[1,:,:] = x_int
    return shifts_out.reshape((2,img_dims[0]*img_dims[1])).swapaxes(0,1)

def interpolate_shifts_circular(metaman, img_dims, obj_dims):
    """
    Like interpolate_shifts above, but considers the shifts in terms of circular cooridinates
    and interpolates based on circular coordinates of position.
    Where interpolate_shifts would lead to position-variant scaling,
    this should lead to position-variant warping about the center.
    """
    xq = np.arange(-img_dims[1]/2,img_dims[1]/2);
    yq = np.arange(-img_dims[0]/2,img_dims[0]/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    # create list of x-shifts, y_shifts, x_origin, y_origin
    x_shifts = []
    y_shifts = []
    x_origin = []
    y_origin = []
    shifts = metaman.shifts
    origins = metaman.field_origins
    for key in shifts.keys():
        #x_shifts.append(shifts[key][0])
        # we need x_shifts and y_shifts to all be positive,
        # to represent points on the image plane
        x_shifts.append(shifts[key][0] + img_dims[1]/2)
        y_shifts.append(shifts[key][1] + img_dims[0]/2)
        # origins can be 0 at center of image, because the meshgrids
        # range over (-dim/2, dim/2)
        #x_origin.append(origins[key][0])
        #y_origin.append(origins[key][1])
        this_origin = find_pixel_on_obj(*(origins[key]), img_dims, obj_dims)
        x_origin.append(this_origin[0])
        y_origin.append(this_origin[1])

    # complex number with x being the real component, which is converted to an angle
    origin_complex = np.asarray(x_origin) + 1j*np.asarray(y_origin)
    #r_origin = np.absolute(origin_complex)
    #phi_origin = np.angle(origin_complex)
    circle_origins = np.empty((origin_complex.shape[0], 2))
    #circle_origins[:,0] is r, [:,1] is phi
    circle_origins[:,0] = np.absolute(origin_complex)
    circle_origins[:,1] = np.angle(origin_complex)

    shift_complex = np.asarray(x_shifts) + 1j*np.asarray(y_shifts)
    r_shift = np.absolute(shift_complex)
    phi_shift = np.angle(shift_complex)
    #circle_shifts = np.empty((shift_complex.shape[0], 2))
    #circle_shifts[:,0] is r, [:,1] is phi
    #circle_shifts[:,0] = np.absolute(shift_complex)
    #circle_shifts[:,1] = np.angle(shift_complex)

    # do the same with our meshgrid
    # make a complex meshgrid representing coordinates:
    mg_comp = Xq + 1j*Yq

    # make the two respective meshgrids for r and phi
    Rq = np.absolute(mg_comp)
    Phiq = np.angle(mg_comp)

    # do the shift interpolation:
    r_int = scipy.interpolate.griddata(circle_origins, r_shift, (Rq, Phiq),method='cubic')
    #r_int[np.isnan(r_int)] = 0
    phi_int = scipy.interpolate.griddata(circle_origins, phi_shift, (Rq, Phiq),method='cubic')
    #phi_int[np.isnan(phi_int)] = 0
    # Will handle NaNs later--they should just carry through computation

    # now that the radial and angular shifts have been interpolated,
    # we want to turn these back into coordinate shifts so they're usable with everything else
    #x_shift = np.cos(phi_int)*r_int
    #y_shift = np.sin(phi_int)*r_int
    #x_shift[np.isnan(x_shift)] = 0
    #y_shift[np.isnan(y_shift)] = 0
    # setting the shifts_out to zero at nan positions is useless because that would create a bright spot
    # in the top left corner of the image.

    shifts_out = np.zeros((2, img_dims[0], img_dims[1]))
    shifts_out[0,:,:] = np.sin(phi_int)*r_int
    shifts_out[1,:,:] = np.cos(phi_int)*r_int
    # currently returning nans where they're clipped
    return shifts_out.reshape((2,img_dims[0]*img_dims[1])).swapaxes(0,1)



def mul_dense_sparse(dense_matrix, csc_matrix, shifts, img_dims, ker_dims, out_cols, out_rows, out_vals, quite_small = 0.001):
    """
    Do left-multiplication of a CSC matrix by a dense matrix
    because this isn't implemented in scipy (lol)
    and then shift by the appropriate value in a given ndarray of shifts

    For instance, let dense_matrix be the p-by-k matrix of kernel weights for each pixel
    let csc_matrix be the k-by-b compressed-sparse-column matrix representing the kernels;
    very useful if csc_matrix also has all nonzero values concentrated within a few rows
    (as is the case with registered PSFs)

    Puts the output stuff into the provided memmaps.
    Parameters:
        dense_matrix: of type np.ndarray, with shape (a, b)
        csc_matrix: of type scipy.sparse.csc_matrix with shape (b, c)
        img_dims: (height, width)
        ker_dims: (height, width)
        out_cols: a memmap or ndarray in which to put the column indices for the output
        out_rows: a memmap or ndarray in which to put the row indices for the output
        out_vals: a memmap or ndarray in which to put the values of the output COO matrix
    Returns:
        nothing
    """
    # UPDATES: shifting has to be perpendicular to axis in which we will need to rapidly access.
    # Also, can't do it with another matrix by which we multiply the one returned from here
    # see physical notebook for details.
    # This means that, when we think we can know all the row locations, we can't.
    # To deal with this, we would have to either start by returning a COO matrix,
    # or return a CSC matrix that we subsequently shift the columns of, and then convert to CSR
    # for fast simulation of images.
    # Today, I felt like going the COO route. Maybe tomorrow I'll change my mind.
    # The reasoning for choosing COO would be that it's easier to convert to CSR afterward.
    # The reasoning for CSC would be that it's smaller and the shifting computation can be done better...
    # but this thing about the shifting computation being more doable in CSC than in COO is moot considering GPUs.

    # we need the transpose of our dense matrix because broadcastable indexing does so in the first axis
    # we want this first axis to be the columns of the input dense matrix
    #dense_transpose = dense_matrix.swapaxes(0,1)
    # create a view of dense_transpose to use henceforth which has irelevant columns removed
    # (that is, irrelevant rows of dense_matrix)
    # by means of dense_matrix[shifts > 0 and shifts < img_dims[0]*img_dims[1]]
    dense_transpose = dense_matrix[shifts > 0 and shifts < img_dims[0]*img_dims[1]].swapaxes(0,1)

    # iterate through columns of our csc_matrix
    for j in range(len(csc_matrix.indptr) -1):

        # list of indices of nonzero values in the jth column of the sparse matrix
        jth_col_ind = csc_matrix.indices[csc_matrix.indptr[j]:csc_matrix.indptr[j+1]]

        # Got rid of shifting at this stage entirely. Should have a different function that shifts the COO

        # FIXME: we shift it to the center, but we can't be specific about shifting yet
        #col_ind_shifted = jth_col_ind \
                #+ img_dims[1]*(img_dims[0]//2-1) + (img_dims[0]//2 - 1) \
                #+ ker_dims[1]*(ker_dims[0]//2-1) + (ker_dims[0]//2 - 1) \
                #+ shifts

        # TODO: verify that this particular column is not irrelevant;
        # if it is irrelevant, continue
        # TODO: cut out parts that fall off along x-axis of image
        #x_cut = col_ind_shifted % ker_dims[1]
        #in_fov_selector = col_ind_shifted > 0 and col_ind_shifted < img_dims[0]*img_dims[1] # just cut out stuff falling off y-axis
        #in_fov_selector = col_ind_shifted > 0 and col_ind_shifted < img_dims[0]*img_dims[1] and
        #if not np.any(in_fov_selector):
            #continue

        # extract the elements in col_ind_shifted which are still in view
        #px_ind_in_fov = col_ind_shifted[col_ind_shifted > 0 and col_ind_shifted < img_dims[0]*img_dims[1]]
        #px_ind_in_fov = col_ind_shifted[in_fov_selector]

        # extract relevant columns of dense_matrix
        #relevant_columns = dense_transpose[px_ind_in_fov]
        relevant_columns = dense_transpose[jth_col_ind]

        # want to pointwise multiply the values of the jth column of the sparse matrix
        # by the relevant columns, then sum the result along the second axis
        # pointwise multiplication happens along last axis, so we need to transpose relevant_columns
        # this gives us the list of values in the jth column of the output matrix
        product_full = np.sum(relevant_columns.swapaxes(0,1)*csc_matrix.data[csc_matrix.indptr[j]:csc_matrix.indptr[j+1]], 1)
        out_col_indices = np.arange(product_full.shape[0])[product_full > quite_small]
        out_data = product_full[product_full > quite_small]
        #out_

def mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, ker_dims, rows_disk, cols_disk, vals_disk, quite_small=0.001):
    """
    create the shifted COO mastermat.
    This can later be gone through to make the CSR that we need for the actual simulations.
    Parameters:
        kermat: the b-by-k CSR matrix of vectorized kernels
        weightsmat: the dense k-by-p matrix of pixel weights
        shifts: the p-vector of modulo-encoded shifts
        img_dims: (height, width) dimensions of the image
        ker_dims: (height, width) dimensions of the kernel; e.g. both sqrt of b
        rows_disk: (huge,) containing the row indices
        cols_disk: (huge,) containing the col indices
        vals_disk: (huge,) containing the values for our COO matrix
    """
    # extract only the pixels in the field of view
    # it's expected that everything which is to be shifted off the FOV in the x-dimension
    # has already had its shift set to a large negative value, since otherwise we can't distinguish
    # diagonal shifting down and to the left from x-overflow
    #selector = [shifts >= 0 and shifts < (img_dims[0]+1)*img_dims[1]]
    #pixels_in_fov = weightsmat.swapaxes(0,1)[selector]

    add_index = 0
    for pixel_ind in range(weightsmat.shape[1]):
        # a print statement to let us know where we are
        print("pixel_ind: ", pixel_ind)
        # the specific PFS vector produced by matrix muliplication of weights vector with
        # kernel matrix
        out_col = csr_kermat.dot(weightsmat[:,pixel_ind])

        # the position in which the top-left corner of the kernel must end up after shifting,
        # in vector form (top-left-corner shift):
        #tlcs = int(shifts[pixel_ind]) \
                #- ker_dims[1]*(ker_dims[0]//2-1) - (ker_dims[1]//2 - 1) \

        # get the indices of all nonzero values in the out_col b-vector
        kernel_row_ind = np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)[out_col>quite_small]

        # select only those indices that don't get cut off the FOV
        #kernel_row_ind_keep = kernel_row_ind[\
                #(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] < img_dims[1]) \
                #*(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] > 0)\
                #*(kernel_row_ind + tlcs < img_dims[0]*img_dims[1])\
                #*(kernel_row_ind + tlcs > 0)]

        # select those indices that don't get clipped off, using the new approach:
        selector = ((kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] < img_dims[1]) # within right side
        * (kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] > 0) # within left side
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] < img_dims[0]) # above bottom
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] > 0) # below top
        )

        kernel_row_ind_keep = kernel_row_ind[selector]

        if kernel_row_ind_keep.shape[0] > 0:
            print("keeping something")

        # for the above-selected unclipped indices, select only the corresponding values
        values = out_col[kernel_row_ind_keep]

        # shift to convert kernel space to image space
        #convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*img_dims[1]).swapaxes(0,1))
        # should be adding img_dims[1] - ker_dims[1] for each row, because otherwise convert_shift basically causes
        # a lateral shift with each line
        convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*(img_dims[1]-ker_dims[1])).swapaxes(0,1))
        # convert_shift excluding small values
        wo_zero = convert_shift[out_col>quite_small]
        # wo_zero excluding clipped
        convert_keep = wo_zero[selector[:,0]]
        # shift the the row indices as needed:
        img_row_ind = kernel_row_ind_keep + convert_keep + int(shifts[pixel_ind, 0])*img_dims[1] + shifts[pixel_ind,1]

        # add the row index to the row indices-row of the out_array memmap
        #out_array[0, add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        rows_disk[add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        #out_array[1, add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        cols_disk[add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        #out_array[2, add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        vals_disk[add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        #out_array.flush()
        rows_disk.flush()
        cols_disk.flush()
        vals_disk.flush()
        add_index = add_index + img_row_ind.shape[0]

def make_mastermat(psfs_directory, psf_meta_path, img_dims, obj_dims):
    # SKETCH OF THE SOLUTION:
    # get the PSFs and do the SVD.
    # Reshape the kernels into a k-by-1024 matrix, where 1024=32*32
    # Make this matrix sparse, because it may already make sense to do that

    # Iterate through each pixel, multiply the associated weight vector by the kernel matrix
    # Get array of indices of resulting non-zero values,
    # and another array of the values themselves by something like the following:
    # indices = arange[vec > 0.01]
    # values = vec[vec > 0.01]
    # to apply the appropriate shift:
    # indices.add(shift_vec[pixel_index])

    # then append the elements of indices to a big_indices list
    # append the elements of values to a big_values list,
    # for i in range(len(values)) append pixel_index to cols_vec

    # you end up with:
    # a cols_vec that contains the column in which each non-zero occurred;
    # an indices vector that contains the row in which each non-zero occurred;
    # and a values vector that contains all the non-zero values.
    # All of this could be used to create the sparse matrix with all the info
    # about the optical system that we want.

    # UPDATE: some of the code below was not efficient for making a CSR matrix.
    # We would want to transpose some things to do that.

    metaman = load_metaman(psf_meta_path)
    h, weights = generate_unpadded(psfs_directory, metaman, img_dims, obj_dims)

    # get the shifts to apply to each point
    shifts = interpolate_shifts(metaman, img_dims, obj_dims)

    # the reshaped matrix of PSFs, where each column is a vectorized PSF
    #kermat = psfs.reshape((psfs.shape[0]*psfs.shape[1], psfs.shape[2]))
    # reshape tries to change the last axis first, so we need to transpose the matrix
    # to put the x- and y- axes at the end and
    kermat = h.transpose((2, 0, 1)) \
    .reshape((h.shape[2], h.shape[0]*h.shape[1])) \
    .transpose((1,0)) # don't want to transpose, because we want k by \beta matrix

    # the reshaped matrix of weights
    # the column is the pixel index, the row is the kernel index
    weightsmat = weights.transpose((2,0,1)) \
    .reshape((weights.shape[2], weights.shape[0]*weights.shape[1]))

    # compressed sparse row matrix version of kermat
    # that we will henceforth use for multiplication
    csr_kermat = scipy.sparse.csr_matrix(kermat)
    # want a CSC for the kernel matrix, because now we're right-multiplying the weights vector for each pixel
    #csc_kermat = scipy.sparse.csc_matrix(kermat)

    # create the numpy memmap to which we will save the coordinate matrix
    row_inds = np.memmap('row_inds.dat', mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    col_inds = np.memmap('col_inds.dat', mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    values = np.memmap('values.dat', mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

    # commented out below to prevent overwrites to good data
    mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values)

    #######################################################################################################
    # FOR TESTING, END HERE################################################################################
    #######################################################################################################
    # Mathematically, we could just matrix-multiply the two to get our mastermat:
    # mastermat = kermat*weightsmat
    # (actually, we would need to shift; but whatever, the following point stands)
    # However, matrix multiplication would give us a matrix when we really want
    # to produce a modified sparse matrix.
    # consequently, we will have to effectively iterate through each column of weightsmat
    # and do the matrix multiplication before extracting the info we want
    # and applying the shift.

    # we need three lists from which to generate the output sparse matrix:
    # the list of column indices for the non-zero values;
    # the list of row indices for the non-zero values;
    # and the list of the non-zero values themselves
    #col_ind_final = []
    #row_ind_final = []
    #vals_final = []

    # we need a definition for "close to zero":
    #quite_small = 0.001

    # UPDATE: the code is correct, but I am limited by the technology of my time.
    # The loop below freezes up at about the 788598th pixel because the computer
    # has trouble keeping such huge lists in memory at once.
    # My first approach would be to use lists of ndarrays rather than pulling the elements out
    # one at a time
    #for pixel_ind in range(weightsmat.shape[1]):
        # a print statement to let us know where we are
        #print("pixel_ind: ", pixel_ind)
        # the specific PFS vector produced by matrix muliplication of weights vector with
        # kernel matrix
        #out_col = csr_kermat.dot(weightsmat[:,pixel_ind])

        # the indices of the rows on which the consequent PSF vector is non-zero
        # shift so the center of the kernel is in the center of the image,
        # then shift from there by whatever we want
        #row_ind = (np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)[out_col>quite_small]\
                #+ img_dims[1]*(img_dims[0]//2-1) + (img_dims[0]//2 - 1) \
                #+ h.shape[1]*(h.shape[0]//2-1) + (h.shape[0]//2 - 1) \
                #+ int(shifts[pixel_ind])) \
        #[:,0] # funky notation to turn column array into regular array

        # the values that are nonzero,
        # and also only corresponding to non-negative row indices
        #values = (out_col[out_col>quite_small])[row_ind >= 0]

        # now that we've only chosen values corresponding to non-negative rows,
        # we should select only those rows as well for the row_ind array
        #row_ind = row_ind[row_ind >= 0]

        # add the appropriate values to the appropriate lists initialized before the for-loop:
        #row_ind_final.extend(row_ind)
        # like the line above, but directly appending the ndarray into the list instead
        #row_ind_final.append(row_ind)
        #vals_final.extend(values)
        #vals_final.append(values)
        # add the index of the current pixel (which is the column index
        # of the mastermat) as many times as we have non-zero values in this column
        #col_ind_final.extend(pixel_ind*np.ones(row_ind.shape[0]))
        #col_ind_final.append(pixel_ind*np.ones(row_ind.shape[0]))

    # now we can generate the sparse mastermat, which we now want to return
    #return scipy.sparse.csr_matrix(vals_final, (row_ind_final, col_ind_final))
    # since we appended ndarrays into lists, we should create an ndarray of the appropriate size from this
    # and then ravel this and use it to make our CSR matrix
    #row_ind_ndarray = np.ravel(np.asarray(row_ind_final))
    #col_ind_ndarray = np.ravel(np.asarray(col_ind_final))
    #vals_ndarray = np.ravel(np.asarray(vals_final))
    #return scipy.sparse.csr_matrix(vals_ndarray, (row_ind_ndarray, col_ind_ndarray))

    # now, we want to produce a CSR from the massive returned COO matrix
    # to do this: iterate through each possible image pixel index;
    # create a mask that covers all entries with this value in row_inds;
    # characterize the length of this mask; apply to col_inds, values
    NNZ = values[values!=0].shape[0] # the number of nonzero values

    # changed all below to r+ to prevent overwrites
    row_inds_csr = np.memmap('row_inds_csr.dat', mode='r+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
    col_inds_csr = np.memmap('col_inds_csr.dat', mode='r+', shape=(NNZ), dtype=np.uint64)
    values_csr = np.memmap('values_csr.dat', mode='r+', shape=(NNZ), dtype=np.float64)

    to_csr_ind = 0

    # since checking every possible pixel takes forever,
    # we can just extract the ones that have nonzero values.
    # These should already be sorted
    nnz_pixels = np.unique(row_inds[:NNZ])

    for pixel_ind in range(img_dims[0]*img_dims[1]):
    #for pixel_ind in nnz_pixels:
        print("pixel_ind: ", pixel_ind)
        mask = row_inds[:NNZ] == pixel_ind
        mask_size = np.count_nonzero(mask)
        col_inds_csr[to_csr_ind:mask_size + to_csr_ind] = col_inds[:NNZ][mask]
        values_csr[to_csr_ind:mask_size + to_csr_ind] = values[:NNZ][mask]
        row_inds_csr[pixel_ind] = mask_size + to_csr_ind
        to_csr_ind = to_csr_ind + mask_size

    return row_inds_csr, col_inds_csr, values_csr

def load_memmaps(img_dims, coo_paths=('row_inds.dat', 'col_inds.dat', 'values.dat'), csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat')):
    row_inds = np.memmap(coo_paths[0], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    col_inds = np.memmap(coo_paths[1], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    values = np.memmap(coo_paths[2], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

    #NNZ = values[values!=0].shape[0] # the number of nonzero values

    #row_inds_csr = np.memmap(csr_paths[0], mode='r+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
    #col_inds_csr = np.memmap(csr_paths[1], mode='r+', shape=(NNZ), dtype=np.uint64)
    #values_csr = np.memmap(csr_paths[2], mode='r+', shape=(NNZ), dtype=np.float64)

    row_inds_csr, col_inds_csr, values_csr = load_csr_memmaps(row_inds, col_inds, values, img_dims, csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat'))

    return row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values

def load_csr_memmaps(row_inds, col_inds, values, img_dims, csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat')):
    NNZ = values[values!=0].shape[0] # the number of nonzero values

    row_inds_csr = np.memmap(csr_paths[0], mode='r+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
    col_inds_csr = np.memmap(csr_paths[1], mode='r+', shape=(NNZ), dtype=np.uint64)
    values_csr = np.memmap(csr_paths[2], mode='r+', shape=(NNZ), dtype=np.float64)

    return row_inds_csr, col_inds_csr, values_csr


def compute_csr(row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values):
    # compute CSR form from the COO matrix
    # takes the same approach as in make_mastermat except by means of vectorization
    NNZ = values[values!=0].shape[0] # the number of nonzero values

    # quickly compute the "deltas"; the number of nonzero entries in each row
    ri_delta = np.bincount(row_inds[:NNZ].astype(np.int64))
    row_inds_csr[0] = 0

    # essentially want to get the discrete "integral" of the ri_delta
    # do this in O(n) time because we actually have to iterate
    for pixel_ind in range(len(ri_delta)):
        row_inds_csr[pixel_ind + 1] = row_inds_csr[pixel_ind] + ri_delta[pixel_ind]

    # function that, given a pixel index, reorders the column indices and values
    # as needed
    def deal_with_pixel(pixel_ind):
        print("pixel_ind: ", pixel_ind)
        mask = row_inds[:NNZ] == pixel_ind
        col_inds_csr[row_inds_csr[pixel_ind]:row_inds_csr[pixel_ind + 1]] = col_inds[:NNZ][mask]
        values_csr[row_inds_csr[pixel_ind]:row_inds_csr[pixel_ind + 1]] = values[:NNZ][mask]

    # vectorized process for dealing with a particular pixel
    vf = np.vectorize(deal_with_pixel)

    # to deal with all the pixels, simply need to apply this to an arange which includes all possible pixel values
    vf(np.arange(img_dims[0]*img_dims[1]))

def make_mastermat_coo(psfs_directory, psf_meta_path, img_dims, obj_dims, memmap_paths=('row_inds.dat','col_inds.dat','values.dat')):
    """
    Just a wrapper around the coo_logic function above.
    Makes the underlying memmaps and then calls the former.
    memmap_paths is a tuple of (row_path, col_path, vals_path)
    """
    metaman = load_metaman(psf_meta_path)
    h, weights = generate_unpadded(psfs_directory, metaman, img_dims, obj_dims)

    # get the shifts to apply to each point
    shifts = interpolate_shifts(metaman, img_dims, obj_dims)

    # the reshaped matrix of PSFs, where each column is a vectorized PSF
    #kermat = psfs.reshape((psfs.shape[0]*psfs.shape[1], psfs.shape[2]))
    # reshape tries to change the last axis first, so we need to transpose the matrix
    # to put the x- and y- axes at the end and
    kermat = h.transpose((2, 0, 1)) \
    .reshape((h.shape[2], h.shape[0]*h.shape[1])) \
    .transpose((1,0)) # don't want to transpose, because we want k by \beta matrix

    # the reshaped matrix of weights
    # the column is the pixel index, the row is the kernel index
    weightsmat = weights.transpose((2,0,1)) \
    .reshape((weights.shape[2], weights.shape[0]*weights.shape[1]))

    # compressed sparse row matrix version of kermat
    # that we will henceforth use for multiplication
    csr_kermat = scipy.sparse.csr_matrix(kermat)
    # want a CSC for the kernel matrix, because now we're right-multiplying the weights vector for each pixel
    #csc_kermat = scipy.sparse.csc_matrix(kermat)

    # create the numpy memmap to which we will save the coordinate matrix
    row_inds = np.memmap(memmap_paths[0], mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    col_inds = np.memmap(memmap_paths[1], mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    values = np.memmap(memmap_paths[2], mode='r+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

    # commented out below to prevent overwrites to good data
    mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values)

    return row_inds, col_inds, values



