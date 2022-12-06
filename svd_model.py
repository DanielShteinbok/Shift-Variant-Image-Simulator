import numpy as np
from scipy.fftpack import dct, idct

import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def  register_psfs(stack,ref_im,dct_on=True):
    # stack is the (h, w, c)-dimensioned set of the PSFs, all at one layer (as passed in by process_psf_for_svd.ipynb)
    # ref_im is e.g. stack[:,:,4] (it is that in process_psf_for_svd.ipynb)

    [Ny, Nx] = stack[:,:,0].shape;
    vec = lambda x: x.ravel()
    pad2d = lambda x: np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)),'constant', constant_values=(0))
    fftcorr = lambda x,y:np.fft.ifft2(np.fft.fft2(pad2d(x))*np.conj(np.fft.fft2(np.fft.ifftshift(pad2d(y)))));
    M = stack.shape[2]
    Si = lambda x,si:np.roll(np.roll(x,si[0],axis=0),si[1],axis=1);

    pr = Ny + 1;
    pc = Nx + 1; # Relative centers of all correlations

    yi_reg = 0*stack;   #Registered stack
    pad = lambda x:x;
    crop = lambda x:x;
    pad2d = lambda x:np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)),'constant', constant_values=(0))
    crop2d = lambda x: x[Ny//2:3*Ny//2,Nx//2:3*Nx//2];


    #     % Normalize the stack first
    # M-entry vector of zeros
    stack_norm = np.zeros((1,M));
    
    # by-value copy of stack
    stack_dct = stack*1;
    
    # calculate the Frobenius norm of ref_im
    ref_norm = np.linalg.norm(ref_im,'fro');
    
    # iterate through each of the PSFs
    for m in range (M):
        # stack_norm is just a 1D vector--the 0 index is for 'semantics'
        # set the mth entry in stack_norm to the Frobenius norm of the mth PSF
        stack_norm[0,m] = np.linalg.norm(stack_dct[:,:,m],'fro');
        
        # normalize stack_dct by the appropriate Frobenius norm
        stack_dct[:,:,m] = stack_dct[:,:,m]/stack_norm[0,m];
        
        # normalize stack by the appropriate Frobenius norm
        stack[:,:,m] = stack[:,:,m]/ref_norm;

    # normalize reference image according to its norm, calc'd above
    ref_im = ref_im/ref_norm;


    #     #########
    si={}

    # #     % Do fft registration


    if dct_on:
        print('Removing background\n')
        
        # iterate through PSFs; consider nth PSF
        for n in range (stack_dct.shape[2]):
            # im is the nth PSF
            im = stack_dct[:,:,n];
            # scipy.fftpack.dct: Discrete Cosine Transform
            bg_dct = dct(im);
            bg_dct[0:19,0:19] = 0;

            stack_dct[:,:,n] = idct(np.reshape(bg_dct,im.shape));


        print('done\n')
    roi=np.zeros((Ny,Nx))
    print('registering\n')
    good_count = 0;

    for m in range (M):

        corr_im = np.real(fftcorr(stack_dct[:,:,m],ref_im));

        if np.max(corr_im) < .01:
            print('image %i has poor match. Skipping\n',m);
        else:

            [r,c] =np.unravel_index(np.argmax(corr_im),(2*Ny,2*Nx))

            si[good_count] = [-(r-pr),-(c-pc)];


            W = crop2d(Si(np.logical_not(pad2d(np.logical_not(roi))),-np.array(si[good_count])));

            bg_estimate = np.sum(np.sum(W*stack[:,:,m]))/np.maximum(np.count_nonzero(roi),1)*0;
            im_reg = ref_norm*crop(Si(pad(stack[:,:,m]-bg_estimate),si[good_count]));


            yi_reg[:,:,good_count] = im_reg;
            good_count = good_count + 1;


    yi_reg = yi_reg[:,:,0:good_count];


    print('done registering\n')
    
    
    return yi_reg,si

def calc_svd(yi_reg,si,rnk,method='nearest'):  
    # NOTE: any value of method except 'nearest' will lead to NaNs being inserted outside the 
    # convex hull of the points given in si. This will lead to lots of problems down the line.
    # An error of "lam value too large" produced by the poisson noise function could be due to these NaNs.
    """
    Parameters:
        yi_reg: shape=(height, width, N) where N is the number of PSFs
        si: dictionary of {field number: [x, y]}
    """
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]


    # si_mat = reshape(cell2mat(si)',[2,Mgood]);
    xq = np.arange(-Nx/2,Nx/2);
    yq = np.arange(-Ny/2,Ny/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    weights_interp = np.zeros((Ny, Nx,rnk));
    xi=[]
    yi=[]
    si_list=list(si.values())

    for i in range(len(si_list)):
        xi.append(si_list[i][0])
        yi.append(si_list[i][1])

    print('interpolating...\n')
    for r in range(rnk):
    #     interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    #     weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
        # BELOW IS WRONG: should be (y,x) rather than (x,y)
        weights_interp[:,:,r]=griddata((xi,yi),weights[:,r],(Xq,Yq),method=method)
        #weights_interp[:,:,r]=griddata((yi,xi),weights[:,r],(Yq,Xq),method=method)

    print('done\n\n')

    return np.flip(comps,-1), np.flip(weights_interp,-1)

def calc_svd_shiftlist(yi_reg,si_list,rnk,method='nearest'):  
    # NOTE: any velue of method except 'nearest' will lead to NaNs being inserted outside the 
    # convex hull of the points given in si. This will lead to lots of problems down the line.
    # An error of "lam value too large" produced by the poisson noise function could be due to these NaNs.
    # this is the same function as calc_svd, but allows us to pass a list of shifts directly
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]


    # si_mat = reshape(cell2mat(si)',[2,Mgood]);
    xq = np.arange(-Nx/2,Nx/2);
    yq = np.arange(-Ny/2,Ny/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    weights_interp = np.zeros((Ny, Nx,rnk));
    xi=[]
    yi=[]
    # si_list passed directly as ordered list
    #si_list=list(si.values())

    for i in range(len(si_list)):
        xi.append(si_list[i][0])
        yi.append(si_list[i][1])

    print('interpolating...\n')
    for r in range(rnk):
    #     interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    #     weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
        # BELOW IS WRONG: should be (y,x) rather than (x,y)
        weights_interp[:,:,r]=griddata((xi,yi),weights[:,r],(Xq,Yq),method=method)
        #weights_interp[:,:,r]=griddata((yi,xi),weights[:,r],(Yq,Xq),method=method)

    print('done\n\n')

    return np.flip(comps,-1), np.flip(weights_interp,-1)

def calc_svd_indexed(yi_reg,si,index_table,rnk,method='nearest'):  
    # NOTE: any value of method except 'nearest' will lead to NaNs being inserted outside the 
    # convex hull of the points given in si. This will lead to lots of problems down the line.
    # An error of "lam value too large" produced by the poisson noise function could be due to these NaNs.
    # this is the same function as calc_svd, but allows us to pass a list of shifts directly
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]


    # si_mat = reshape(cell2mat(si)',[2,Mgood]);
    xq = np.arange(-Nx/2,Nx/2);
    yq = np.arange(-Ny/2,Ny/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    weights_interp = np.zeros((Ny, Nx,rnk));
    xi=[]
    yi=[]
    # si_list passed directly as ordered list
    #si_list=list(si.values())

    for index in index_table:
        xi.append(si[index][0])
        yi.append(si[index][1])

    print('interpolating...\n')
    for r in range(rnk):
    #     interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    #     weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
        # BELOW IS WRONG: should be (y,x) rather than (x,y)
        weights_interp[:,:,r]=griddata((xi,yi),weights[:,r],(Xq,Yq),method=method)
        #weights_interp[:,:,r]=griddata((yi,xi),weights[:,r],(Yq,Xq),method=method)

    print('done\n\n')

    return np.flip(comps,-1), np.flip(weights_interp,-1)

def calc_svd_indexed_sized(yi_reg,si,index_table,rnk, imgdims, method='nearest'):  
    """
    Parameters:
        imgdims.shape = (height, width)
    """
    # NOTE sized implies that this is meant to work for the case that the rectangle
    # over which to interpolate weights is of different dimensions
    # from than the PSF itself
    # For instance, we want weights interpolated over 800x1280, but the PSF is 32x32
    # NOTE: any value of method except 'nearest' will lead to NaNs being inserted outside the 
    # convex hull of the points given in si. This will lead to lots of problems down the line.
    # An error of "lam value too large" produced by the poisson noise function could be due to these NaNs.
    # this is the same function as calc_svd, but allows us to pass a list of shifts directly
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    # NOTE this is where I suspect the problem of ordering happens
    # the original order of the ymat SHOULD be kept; it's the product of the weights and h
    # that matters at the end of the day, and that product should be the same
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]


    # si_mat = reshape(cell2mat(si)',[2,Mgood]);
#     xq = np.arange(-Nx/2,Nx/2);
#     yq = np.arange(-Ny/2,Ny/2);
    yq = np.arange(-imgdims[0]/2,imgdims[0]/2)
    xq = np.arange(-imgdims[1]/2,imgdims[1]/2)
    [Xq, Yq] = np.meshgrid(xq,yq);

    weights_interp = np.zeros((imgdims[0], imgdims[1],rnk));
    xi=[]
    yi=[]
    # si_list passed directly as ordered list
    #si_list=list(si.values())

    for index in index_table:
        xi.append(si[index][0])
        yi.append(si[index][1])

    print('interpolating...\n')
    for r in range(rnk):
    #     interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    #     weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
        # BELOW IS WRONG: should be (y,x) rather than (x,y)
        weights_interp[:,:,r]=griddata((xi,yi),weights[:,r],(Xq,Yq),method=method)
        if method != "nearest":
            weights_interp[:,:,r][np.isnan(weights_interp[:,:,r])] = griddata((xi,yi),weights[:,r],(Xq,Yq),method="nearest")[np.isnan(weights_interp[:,:,r])]
        #weights_interp[:,:,r]=griddata((yi,xi),weights[:,r],(Yq,Xq),method=method)

    print('done\n\n')

    return np.flip(comps,-1), np.flip(weights_interp,-1)
