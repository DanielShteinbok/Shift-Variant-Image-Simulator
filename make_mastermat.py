import mastermat

psfs_directory = "/home/dshteinbok/nV3_PSFs_flat_hd"
#psfs_directory = "/home/dshteinbok/nV3_PSFs_curve"
#psfs_directory = "/home/dshteinbok/nV3_PSFs_flat"
#psfs_directory = "/home/dshteinbok/nV3_PSFs_probe"
#psfs_directory = "/home/dshteinbok/nV3_PSFs_probe_mark"
psf_meta_path = "/home/dshteinbok/nV3_PSFs_flat_meta/metafile_hd.csv"
#psf_meta_path = "/home/dshteinbok/nV3_PSFs_meta/metafile.csv"
#psf_meta_path = "/home/dshteinbok/nV3_PSFs_flat_meta/metafile.csv"
#psf_meta_path = "/home/dshteinbok/nV3_PSFs_flat_meta/metafile_probe_mark.csv"
#psf_meta_path = "/home/dshteinbok/nV3_PSFs_flat_meta/metafile_probe.csv"
img_dims = (800, 1280)
obj_dims = (640, 1024)

#mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_lin_circ_csr.npy", "col_inds_lin_circ_csr.npy", "values_lin_circ_csr.npy"),
        #w_interp_method="linear", s_interp_coords="circular")

#mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_lin_csr.npy", "col_inds_lin_csr.npy", "values_lin_csr.npy"),
        #w_interp_method="linear")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve_csr.npy", "col_inds_curve_csr.npy", "values_curve_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve2_csr.npy", "col_inds_curve2_csr.npy", "values_curve2_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve_nearest_csr.npy", "col_inds_curve_nearest_csr.npy", "values_curve_nearest_csr.npy"),
        #w_interp_method="nearest")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve_rot_csr.npy", "col_inds_curve_rot_csr.npy", "values_curve_rot_csr.npy"),
        #w_interp_method="cubic", rotate_psfs=True)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_diy_csr.npy", "col_inds_diy_csr.npy", "values_diy_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_hd_nearest_csr.npy", "col_inds_hd_nearest_csr.npy", "values_hd_nearest_csr.npy"),
        #w_interp_method="nearest")

mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        savepath = ("row_inds_hd_csr.npy", "col_inds_hd_csr.npy", "values_hd_csr.npy"),
        w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_fm3_csr.npy", "col_inds_fm3_csr.npy", "values_fm3_csr.npy"),
        #w_interp_method="nearest")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_csr.npy", "col_inds_probe_csr.npy", "values_probe_csr.npy"),
        #w_interp_method="cubic")
