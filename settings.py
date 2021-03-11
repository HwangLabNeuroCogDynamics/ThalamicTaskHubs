# Basic settings
BIDS_DIR = 'BIDS/'
SUB_PREFIX = 'sub-'
FUNC_DIR = 'func/'
MRIQC_DIR = 'mriqc/'
FMRIPREP_DIR = 'fmriprep/'
DECONVOLVE_DIR = '3dDeconvolve/'
FREESURFER_DIR = 'freesurfer/'
SESSION = 'SESSION'
LOGS_DIR = 'logs/'

# Replacement Keys
DATASET_KEY = 'DATASET_DIR'
SLOTS_KEY = 'SLOTS'
BIDS_KEY = 'BIDS_DIR'
WORK_KEY = 'WORK_DIR'
SUBS_KEY = 'SUBJECTS'
SGE_KEY = 'SGE_OPTIONS'
OPTIONS_KEY = 'OPTIONS'
BASE_SCRIPT_KEY = 'BASE_SCRIPT'
EFILE_KEY = 'EFILE'
MRIQC_KEY = 'MRIQC_DIR'

# Workflow settings
MRIQC = 'mriqc'
FMRIPREP = 'fmriprep'
DECONVOLVE = '3dDeconvolve'
DEFAULT_WORKFLOW = [MRIQC, FMRIPREP, DECONVOLVE]
JOB_SCRIPTS_DIR = '/Shared/lss_kahwang_hpc/scripts/jobs/'
FIRST_RUN = 'First Run'
FAILED_SUB = 'Failed'
FAILED_SUB_MEM = 'Failed Mem'

# HPC settings
DEFAULT_QUEUE = 'SEASHORE'
LARGE_QUEUE = 'all.q'
QSUB = 'qsub -terse '
ARRAY_QSUB = ' | awk -F. \'{print $1}\''
SUB_BASH = '${subject}'
SUB_DIR_BASH = SUB_PREFIX + SUB_BASH + '/'
LOCALSCRATCH = '/localscratch/Users/'

# mriqc settings
MRIQC_BASHFILE = '/Shared/lss_kahwang_hpc/scripts/preprocessing/mriqc/mriqc_base.sh'
MRIQC_GROUP_BASHFILE = '/Shared/lss_kahwang_hpc/scripts/preprocessing/mriqc/mriqc_group_base.sh'

# fmriprep settings
FMRIPREP_BASHFILE = '/Shared/lss_kahwang_hpc/scripts/preprocessing/fmriprep/fmriprep_base.sh'
MEM_ERROR = ('concurrent.futures.process.BrokenProcessPool: A process in the '
             'process pool was terminated abruptly while the future was running '
             'or pending.')
FAILED_SUB_FILE = 'failed_subjects.txt'
FAILED_SUB_MEM_FILE = 'failed_subjects_mem.txt'
COMPLETED_SUBS_FILE = 'completed_subjects.txt'

# 3dDeconvolve settings
DEFAULT_COLUMNS = ('csf', 'white_matter',  'trans_x',
                   'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')
REGRESSOR_FILE = 'nuisance.1D'
REGRESSOR_WC = 'regressors.tsv'
CENSOR_FILE = 'censor.1D'
EVENTS_WC = 'events.tsv'
STIM_CONFIG = 'stim_config.csv'
BUCKET_FILE = 'FIRmodel_MNI_stats'
ERRTS_FILE = 'FIRmodel_errts.nii.gz'
MASK_FILE = 'combined_mask+tlrc.BRIK'


# singularity settings
SING_RUNCLEAN = 'singularity run --cleanenv'
AFNI_SING_PATH = '/Shared/lss_kahwang_hpc/opt/afni/afni.sif'
