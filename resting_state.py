from thalpy.analysis import fc, masks
from thalpy import base
from thalpy.constants import wildcards
import nibabel as nib
import os

MDTB_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/"
dir_tree = base.DirectoryTree(MDTB_DIR)
subjects = base.get_subjects(dir_tree.deconvolve_dir, dir_tree)


n_masker = masks.get_binary_masker(masks.MOREL_PATH)
m_masker = masks.get_roi_masker(masks.SCHAEFER_YEO7_PATH)
print(dir_tree.deconvolve_dir)
fc_data = fc.FcData(
    MDTB_DIR,
    n_masker,
    m_masker,
    "fc_task_residuals",
    subjects=subjects,
    censor=False,
    is_denoise=False,
    bold_dir=dir_tree.deconvolve_dir,
    bold_WC="*FIRmodel_errts_block.nii.gz",
    cores=12,
)
fc_data.calc_fc()
print(fc_data.data.shape)
