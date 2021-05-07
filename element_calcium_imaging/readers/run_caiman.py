import cv2
import inspect

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import *
from caiman.source_extraction.cnmf import params as params

from element_calcium_imaging.readers.caiman_loader import save_mc


# hacky stuff here:

fit_file_source_code = inspect.getsource(CNMF.fit_file)

# based on the source code of filt_file, build a new source code for "new_fit_file"
# where the motion correction object ("mc") is also returned
new_fit_file = fit_file_source_code.replace('def fit_file',
                                            'def new_fit_file')
new_fit_file = new_fit_file.replace('return cnm2',
                                    'return (cnm2, mc) if motion_correct else (cnm2, None)')
new_fit_file = '\n'.join([line[4:] for line in new_fit_file.split('\n')])

# print(new_fit_file)

exec(new_fit_file)

CNMF.fit_file = new_fit_file


def run_caiman(file_paths, parameters, sampling_rate):
    parameters['fnames'] = file_paths
    parameters['fr'] = sampling_rate

    opts = params.CNMFParams(params_dict=parameters)

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    cnm = CNMF(n_processes, params=opts, dview=dview)
    cnmf_output, mc_output = cnm.fit_file(motion_correct=True, include_eval=True)

    cm.stop_server(dview=dview)

    cnmf_output_file = pathlib.Path(cnmf_output.mmap_file[:-4] + 'hdf5')
    assert cnmf_output_file.exists()

    save_mc(mc_output, cnmf_output_file.as_posix(), parameters['is3D'])
