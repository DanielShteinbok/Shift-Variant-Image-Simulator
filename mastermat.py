import one_shot_svd
import numpy as np
import scipy
import scipy.sparse

import load_PSFs
import tempfile

from numba import jit

#@jit(nopython=True)
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
    # shift to convert kernel space to image space
    #convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*img_dims[1]).swapaxes(0,1))
    # should be adding img_dims[1] - ker_dims[1] for each row, because otherwise convert_shift basically causes
    # a lateral shift with each line
    convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*(img_dims[1]-ker_dims[1])).swapaxes(0,1))

    # simply skipping columns will lead to virtual zero-columns in the mastermat

    # shifts is going to have, e.g. shape=(2,1024000)
    #not_nan_selector = (~np.isnan(shifts[:,0]))*(~np.isnan(shifts[:,1]))
    #relevant_pixels = np.arange(weightsmat.shape[1])[not_nan_selector].tolist()

    add_index = 0
    for pixel_ind in range(weightsmat.shape[1]):
    #for pixel_ind in relevant_pixels:
        if np.isnan(shifts[pixel_ind,0]) or np.isnan(shifts[pixel_ind,1]):
            continue
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
        #kernel_row_ind = np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)[out_col>quite_small]
        # make kernel_row_ind all indices, then select by value later
        kernel_row_ind = np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)
        value_selector = (out_col > quite_small).reshape((out_col.shape[0],1))

        # select only those indices that don't get cut off the FOV
        #kernel_row_ind_keep = kernel_row_ind[\
                #(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] < img_dims[1]) \
                #*(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] > 0)\
                #*(kernel_row_ind + tlcs < img_dims[0]*img_dims[1])\
                #*(kernel_row_ind + tlcs > 0)]

        # select those indices that don't get clipped off, using the new approach:
        shift_selector = ((kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] < img_dims[1]) # within right side
        * (kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] > 0) # within left side
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] < img_dims[0]) # above bottom
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] > 0) # below top
        )

        selector = shift_selector*value_selector

        kernel_row_ind_keep = kernel_row_ind[selector]

        if kernel_row_ind_keep.shape[0] > 0:
            print("keeping something")

        # for the above-selected unclipped indices, select only the corresponding values
        values = out_col[kernel_row_ind_keep]

        # convert_shift excluding small values
        #wo_zero = convert_shift[out_col>quite_small]
        # wo_zero excluding clipped
        #convert_keep = wo_zero[selector[:,0]]
        #convert_keep = convert_shift[selector[:,0]]
        convert_keep = convert_shift[kernel_row_ind_keep]

        # shift the the row indices as needed:
        img_row_ind = kernel_row_ind_keep + convert_keep + int(shifts[pixel_ind, 0])*img_dims[1] + shifts[pixel_ind,1]

        # add the row index to the row indices-row of the out_array memmap
        #out_array[0, add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        rows_disk[add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        #out_array[1, add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        cols_disk[add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        #out_array[2, add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        vals_disk[add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        rows_disk.flush()
        cols_disk.flush()
        vals_disk.flush()
        add_index = add_index + img_row_ind.shape[0]

@jit(nopython=True)
def csr_mul_vec(csr_mat, vec):
    """
    home-rolled CSR matrix-vector multiplication
    csr_mat is a 3-element tuple containing ndarrays in the order of
    (row, col, values)
    where if the csr_mat-represented matrix is m by n, row.shape = (m+1,)

    The reason for this nonsense home-rolling as opposed to the use of scipy CSR
    is that this is compatible with numba, so while each matrix multiplication
    will probably be slower than that done with scipy,
    I can do a whole bunch (e.g. with a for-loop) in mastermat_coo_creation_logic with numba
    much faster than I would if I were limited to a slow python-only for-loop
    which contains fast scipy matrix multiplication
    """
    row_inds = csr_mat[0]
    col_inds = csr_mat[1]
    vals = csr_mat[2]
    out_vec = np.empty(row_inds.shape[0]-1)
    for i in range(row_inds.shape[0]-1):
        out_vec[i] = np.sum(vals[row_inds[i]:row_inds[i+1]]*vec[col_inds[row_inds[i]:row_inds[i+1]]])
    return out_vec

@jit(nopython=True)
def shift_PSF(psf_sparse, shift, img_dims, ker_dims, convert_shift):
    """
    shifts a sparse PSF by the given shift amount.
    Clips off the part that should fall off the image.
    Assumes the shift is floating-point, so ends up between pixels;
    linearly interpolates the value of the shifted PSF at each pixel
    based on that assumption

    Parameters:
        psf_sparse: tuple (indices, values) of the PSF
        shift: tuple (y, x)
        img_dims: tuple (height, width)
        ker_dims: tuple (height, width)
        convert_shift: 1-D ndarray which could be pointwise added to a vectorized kernel
            to convert it from kernel-modular space to image-modular space
    """
    kernel_row_ind = psf_sparse[0]
    out_col = psf_sparse[1]

    # Had a problem with that things kind of overflowed because we are selecting by top left
    # and then potentially shifting out--be stricter by one at right and bottom
    shift_selector = ((kernel_row_ind % ker_dims[1] + shift[1] < img_dims[1] - 1) # within right side
        * (kernel_row_ind % ker_dims[1] + shift[1] > 0) # within left side
        * (kernel_row_ind / ker_dims[1] + shift[0] < img_dims[0] - 1) # above bottom
        * (kernel_row_ind / ker_dims[1] + shift[0] > 0) # below top
        )

    #selector = shift_selector*value_selector

    kernel_row_ind_keep = kernel_row_ind[shift_selector]

    if kernel_row_ind_keep.shape[0] > 0:
        print("keeping something")

    # for the above-selected unclipped indices, select only the corresponding values
    ref_values = out_col[shift_selector]

    # convert_shift excluding small values
    #wo_zero = convert_shift[out_col>quite_small]
    # wo_zero excluding clipped
    #convert_keep = wo_zero[selector[:,0]]
    #convert_keep = convert_shift[selector[:,0]]
    convert_keep = convert_shift[kernel_row_ind_keep]

    # the final indices of the PSF, snapping to top left
    img_row_ind = kernel_row_ind_keep + convert_keep + np.floor(shift[0])*img_dims[1] + np.floor(shift[1])

    # ref_values are the points that are now to be shifted,
    # to somewhere BETWEEN the pixel locations.
    # This means we must interpolate the values at each of the pixels.
    # The returned array should have kernel_row_ind_keep.shape[0] + ker_dims[0] + ker_dims[1] + 1 entries
    #vals_out = np.empty(kernel_row_ind_keep.shape[0] + ker_dims[0] + ker_dims[1] + 1)
    # NOTE: cannot actually know size of vals_out in advance, because values may
    # be zero-spaced in the matrix

    # right_coeff is the coefficient by which we must multiply the value of the PSF
    # at a floating position to get the brightness of the pixel to the right.
    # This is equal to the normalized distance from the left.
    # Ibid for bottom.
    right_coeff = shift[1] % 1
    left_coeff = 1-right_coeff
    bottom_coeff = shift[0] % 1
    top_coeff = 1-bottom_coeff

    coeffs = (top_coeff*left_coeff,
            top_coeff*right_coeff,
            bottom_coeff*left_coeff,
            bottom_coeff*right_coeff)

    # if we can have the values in any order as long as they match indices,
    # we can just slap them together out-of-order.
    # But then we have to be careful how we apply the convert_shift
    # Alternatively, we can use the kernel_row_ind_keep to select from the get-go:

    # in this interpolation, we create a superposition of the PSF shifted to each
    # of the 4 corners, weighted with the above coefficients

    # find the set of all indices that will be covered in the superposition
    # of the 4 shifted reference kernels
    # in the worst case, we have a total of quadruple the unique
    # indices as the number within kernel_row_ind_keep
    #repeated_inds = np.empty((4, kernel_row_ind_keep.shape[0]))
    #repeated_inds[0,:] = kernel_row_ind_keep
    #repeated_inds[1,:] = kernel_row_ind_keep + 1
    #repeated_inds[2,:] = kernel_row_ind_keep + ker_dims[1]
    #repeated_inds[3,:] = kernel_row_ind_keep + ker_dims[1] + 1

    # this is the tuple that tells us how to snap our PSF to the four corners
    #corner_shift = (0, 1, ker_dims[1], ker_dims[1] + 1)
    # since we'll be operating on already-shifted indices, use img_dims
    corner_shift = (0, 1, img_dims[1], img_dims[1] + 1)

    # in reality, some will repeat
    #unique_ind = np.unique(repeated_inds)

    # we can go by the indices, adding up all the values at those indices for the 4
    # superimposed elements. If ever one of the elements doesn't have such an index,
    # add zero.
    # use a dictionary for fast access by key, as we will see
    vals_out = {}
    unique_inds = np.empty(img_row_ind.shape[0]*4)
    num_unique = 0

    for i in range(img_row_ind.shape[0]):
        for j in range(4):
            index = img_row_ind[i] + corner_shift[j]
            if index not in vals_out:
                vals_out[index] = ref_values[i]*coeffs[j]
                unique_inds[num_unique] = index
                num_unique += 1
            else:
                vals_out[index] += ref_values[i]*coeffs[j]

    unique_inds_stripped = unique_inds[:num_unique]
    # iterate through the stripped unique indices, pull those vals out of the dictionary
    # into an ndarray
    vals_out_array = np.empty_like(unique_inds_stripped)
    for i in range(num_unique):
        vals_out_array[i] = vals_out[unique_inds_stripped[i]]

    return unique_inds_stripped, vals_out_array

@jit(nopython=True)
def rotate_PSF(psf_vec, shift, ker_dims):
    """
    Rotate the PSF.
    Should be done before shifting

    Parameters:
        psf_vec is the dense vector representation of the PSF at the point in question
        shift: is in [y, x] and is an ndarray
    """
    unravelled_psf = np.empty((ker_dims[0], ker_dims[1], 1))
    unravelled_psf[:,:,0] = psf_vec.reshape((ker_dims[0], ker_dims[1]))
    shift_processed = np.empty((1,2)) # we need this to be 2D for rotate_unpadded_psfs
    shift_processed[0,:] = np.flip(shift) # we want to reverse x and y, to get [x,y]
    return np.ravel(
                one_shot_svd.rotate_unpadded_psfs(
                    unravelled_psf, shift_processed, reverse=True)
                )

@jit(nopython=True)
def mastermat_coo_creation_logic_homemade(csr_kermat, weightsmat, shifts, img_dims, ker_dims, rows_disk, cols_disk, vals_disk, quite_small=0.001, rotate_psfs=False):
    """
    A numba-friendly alternative to the function above, using my self-rolled CSR multiplication
    csr_kermat is a 3-element tuple, where each element is an ndarray in the order of (row, col, values)
    """
    # Sketch of solution:
    # iterate through pixels in weightsmat
    # multiply appropriate weight vector by csr_kermat
    # shift the result appropriately
    # insert the result into the appropriate memmap appropriately
    convert_shift = np.ravel(np.swapaxes((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*(img_dims[1]-ker_dims[1])), 0,1))

    # simply skipping columns will lead to virtual zero-columns in the mastermat

    # shifts is going to have, e.g. shape=(2,1024000)
    #not_nan_selector = (~np.isnan(shifts[:,0]))*(~np.isnan(shifts[:,1]))
    #relevant_pixels = np.arange(weightsmat.shape[1])[not_nan_selector].tolist()

    # add_index allows me to later insert appropriatel into the memmaps
    add_index = 0

    # iterating through the pixels
    for pixel_ind in range(weightsmat.shape[1]):
        if np.isnan(shifts[pixel_ind,0]) or np.isnan(shifts[pixel_ind,1]):
            continue
        # a print statement to let us know where we are
        print("pixel_ind: ", pixel_ind)
        #print("add_index: ", add_index)

        # multiply the csr by appropriate column in weightsmat
        out_col = csr_mul_vec(csr_kermat, weightsmat[:,pixel_ind])

        # NOTE: rotate the image here, while it is still dense
        # use this only if already rotated in make_mastermat_save_homemade
        # commented out to make the mastermate code usable while that's under construction
        #if rotate_psfs:
            # we are unrotating the PSF
            #out_col = rotate_PSF(out_col, shifts[pixel_ind, :], ker_dims)

        # grab only values of significant magnitude
        nz_vals = out_col[out_col > quite_small]
        nz_inds = np.arange(out_col.shape[0])[out_col > quite_small]

        # shift values (that is: shift and interpolate four surrounding corners)
        shifted_inds, shifted_vals = shift_PSF((nz_inds, nz_vals), shifts[pixel_ind, :], img_dims, ker_dims, convert_shift)

        rows_disk[add_index:shifted_inds.shape[0] + add_index] = shifted_inds[:] # row indices
        #out_array[1, add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        cols_disk[add_index:shifted_inds.shape[0] + add_index] = pixel_ind*np.ones(shifted_inds.shape[0]) # column indices
        #out_array[2, add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        vals_disk[add_index:shifted_inds.shape[0] + add_index] = shifted_vals[:] # values
        # commented out flushing because numba doesn't understand it
        #rows_disk.flush()
        #cols_disk.flush()
        #vals_disk.flush()
        add_index += shifted_inds.shape[0]

def make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        savepath = ("row_inds_csr.npy", "col_inds_csr.npy", "values_csr.npy"), w_interp_method="nearest", s_interp_coords="cartesian", rotate_psfs=False):
    metaman = load_PSFs.MetaMan(psf_meta_path)
    # TODO: replace generate_unpadded with one_shot_svd.generate_unpadded_rotated
    if not rotate_psfs:
        # normally, we just generate a regular unpadded PSF
        h, weights = one_shot_svd.generate_unpadded(psfs_directory, metaman, img_dims, obj_dims, method=w_interp_method)
    else:
        # if rotate_psfs is True, then we want to first rotate them based on their origin positions, then perform the SVD
        #since we rotated here, we need to unrotate in mastermat_coo_logic_homemade later
        h, weights = one_shot_svd.generate_unpadded_rotated(psfs_directory, metaman, img_dims, obj_dims, method=w_interp_method)

    # get the shifts to apply to each point
    if s_interp_coords=="cartesian":
        shifts = one_shot_svd.interpolate_shifts(metaman, img_dims, obj_dims)
    elif s_interp_coords=="circular":
        shifts = one_shot_svd.interpolate_shifts_circular(metaman, img_dims, obj_dims)

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
    kermat_tuple = (csr_kermat.indptr, csr_kermat.indices, csr_kermat.data)

    # these files should be entirely temporary
    # TODO make it so, could use tempfile module, including tempfile.mkdtemp

    with tempfile.TemporaryDirectory() as name:
        # really bad approach to take!
        # Since I am pre-allocating memmaps below, I need to know what size to make them
        # If I were doing this in C++, I'd just create an array of 1024000 pointers,
        # and the array referenced by any of these pointers would be created to have
        # just the right number of entries.
        # Since I want to be able to move this to disk, I have to use memmaps
        # numba doesn't know anything about memmaps--they don't work with numba
        # consequently I can't dump to disk within my jit-ted method
        # lol what are pointers
        # so I'm just assuming that, on average, the number of elements in a column
        # of my master matrix will be less than avg_nnz
        # 100 works well for the nV3 without probes, but with probes our PSF is larger
        # so need 500
        avg_nnz = 500

        prefix = name + "/"
        row_inds = np.memmap(prefix + 'row_inds_temp.dat', mode='w+', shape=(avg_nnz*img_dims[0]*img_dims[1]), dtype=np.uint64)
        col_inds = np.memmap(prefix + 'col_inds_temp.dat', mode='w+', shape=(avg_nnz*img_dims[0]*img_dims[1]), dtype=np.uint64)
        values = np.memmap(prefix + 'values_temp.dat', mode='w+', shape=(avg_nnz*img_dims[0]*img_dims[1]), dtype=np.float64)

        #mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values)
        mastermat_coo_creation_logic_homemade(kermat_tuple, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values, quite_small=0.001, rotate_psfs=rotate_psfs)
        NNZ = values[values!=0].shape[0] # the number of nonzero values

        row_inds_csr = np.memmap(prefix + 'row_inds_temp_csr.dat', mode='w+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
        col_inds_csr = np.memmap(prefix + 'col_inds_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.uint64)
        values_csr = np.memmap(prefix + 'values_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.float64)

        row_inds_csr, col_inds_csr, values_csr = compute_csr(row_inds, col_inds, values)

        np.save(savepath[0], row_inds_csr)
        np.save(savepath[1], col_inds_csr)
        np.save(savepath[2], values_csr)



def make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims,
        savepath = ("row_inds_csr.npy", "col_inds_csr.npy", "values_csr.npy"), w_interp_method="nearest", s_interp_coords="cartesian"):
    """
    Complete process for making the mastermat and converting to CSR form,
    then saving as .npy to the given path.
    """
    metaman = load_PSFs.MetaMan(psf_meta_path)
    h, weights = one_shot_svd.generate_unpadded(psfs_directory, metaman, img_dims, obj_dims, method=w_interp_method)

    # get the shifts to apply to each point
    if s_interp_coords=="cartesian":
        shifts = one_shot_svd.interpolate_shifts(metaman, img_dims, obj_dims)
    elif s_interp_coords=="circular":
        shifts = one_shot_svd.interpolate_shifts_circular(metaman, img_dims, obj_dims)

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

    # these files should be entirely temporary
    # TODO make it so, could use tempfile module, including tempfile.mkdtemp

    with tempfile.TemporaryDirectory() as name:
        prefix = name + "/"
        row_inds = np.memmap(prefix + 'row_inds_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
        col_inds = np.memmap(prefix + 'col_inds_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
        values = np.memmap(prefix + 'values_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

        mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values)

        NNZ = values[values!=0].shape[0] # the number of nonzero values

        row_inds_csr = np.memmap(prefix + 'row_inds_temp_csr.dat', mode='w+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
        col_inds_csr = np.memmap(prefix + 'col_inds_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.uint64)
        values_csr = np.memmap(prefix + 'values_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.float64)

        row_inds_csr, col_inds_csr, values_csr = compute_csr(row_inds, col_inds, values)

        np.save(savepath[0], row_inds_csr)
        np.save(savepath[1], col_inds_csr)
        np.save(savepath[2], values_csr)

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

def compute_csr(row_inds, col_inds, values):
    """
    just a wrapper around scipy's sparse.csr_matrix
    which returns the three ndarrays of underlying data behind the computed CSR
    so that they can be saved to disk
    """
    out_csr = scipy.sparse.csr_matrix((values, (row_inds, col_inds)))
    return out_csr.indptr, out_csr.indices, out_csr.data

def simulate_image(img, csr_mastermat):
    """
    ravels the image, matrix-multiplies with the mastermat, unravels the result
    """
    img_vec = img.reshape((img.shape[0]*img.shape[1]))
    return csr_mastermat.dot(img_vec).reshape(img.shape)

def load_csr_files(csr_paths=('row_inds_csr.npy', 'col_inds_csr.npy', 'values_csr.npy')):
    row_inds_csr = np.load(csr_paths[0])
    col_inds_csr = np.load(csr_paths[1])
    values_csr = np.load(csr_paths[2])

    return row_inds_csr, col_inds_csr, values_csr


