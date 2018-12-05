import atlascorr.atlas_correlations as ac
import os
import pandas as pd
import nibabel as nib
import numpy as np
import pytest


# Setup files and directories to be used for testins
@pytest.fixture(scope='session')
def deriv_dir(tmpdir_factory):
    bids_dir = tmpdir_factory.mktemp('bids')
    deriv_dir = bids_dir.ensure('derivatives', dir=True)
    return deriv_dir


@pytest.fixture(scope='session')
def prep_dir(deriv_dir):
    return deriv_dir.ensure('prep', 'sub-01', 'ses-pre', 'func', dir=True)


@pytest.fixture(scope='session')
def atlas_dir(deriv_dir):
    return deriv_dir.ensure('atlasCorrelations', 'sub-01', 'ses-pre', 'func', dir=True)


@pytest.fixture(scope='session')
def atlas_file(deriv_dir):
    atlas = np.ones((5, 5, 5), dtype=np.int16)
    atlas_img = nib.Nifti1Image(atlas, np.eye(4))
    atlas_file = deriv_dir.join('atlas.nii.gz')
    atlas_img.to_filename(str(atlas_file))
    return atlas_file

@pytest.fixture(scope='session')
def atlas_lut(deriv_dir):
    atlas_df = pd.DataFrame({'regions': ['region1', 'region2', 'region3']})
    atlas_lut = deriv_dir.join('atlas_lut.tsv')
    atlas_df.to_csv(str(atlas_lut), sep='\t', index=False)
    return atlas_lut

@pytest.fixture(scope='session')
def bold_file(prep_dir):
    data = np.ones((5, 5, 5, 10), dtype=np.float)
    img = nib.Nifti1Image(data, np.eye(4))
    bold_file = prep_dir.join('sub-01_ses-pre_task-rest_bold_preproc.nii.gz')
    img.to_filename(str(bold_file))
    return bold_file


@pytest.fixture(scope='session')
def confounds_file(prep_dir):
    confounds_df = pd.DataFrame({'FramewiseDisplacement': [1.] * 10,
                                 'CSF': [1.] * 10,
                                })
    confounds_file = prep_dir.join('sub-01_ses-pre_task-rest_bold_confounds.tsv')
    confounds_df.to_csv(str(confounds_file), sep='\t', index=False)
    return confounds_file


@pytest.fixture(scope='session')
def brainmask_file(prep_dir):
    mask = np.ones((5, 5, 5), dtype=np.int16)
    brainmask = nib.Nifti1Image(mask, np.eye(4))
    brainmask_file = prep_dir.join('sub-01_ses-pre_task-rest_bold_brainmask.nii.gz')
    brainmask.to_filename(str(brainmask_file))
    return brainmask_file

@pytest.fixture(scope='session')
def corrMatrix_file(atlas_dir):
    corr = np.array([[0.,  1.07, -1.07],
                    [1.07, 0., -1.07],
                    [-1.07, -1.07, 0.]])
    
    corr_df = pd.DataFrame(corr,
                           columns=['region1', 'region2', 'region3'],
                           index=['region1', 'region2', 'region3'])

    corrMatrix_file = atlas_dir.join('sub-01_ses-pre_task-rest_corrMatrix.tsv')
    corr_df.to_csv(str(corrMatrix_file), sep='\t')
    return corrMatrix_file

def test_get_files(bold_file, confounds_file, brainmask_file):
    # FMRIPREP outputs
    assert ac.get_files(str(bold_file)) == (str(confounds_file), str(brainmask_file))


def test_get_confounds(confounds_file):
    confounds = ['FramewiseDisplacement', 'CSF']
    confounds_df = pd.DataFrame({'FramewiseDisplacement': [1.] * 10,
                                 'CSF': [1.] * 10,
                                 })
    # ensure the columns are the same order
    confounds_df = confounds_df[confounds]
    assert confounds_df.equals(ac.proc_confounds(confounds, str(confounds_file)))


def test_extract_ts(bold_file, brainmask_file, confounds_file, atlas_file):
    # setup confounds
    confounds_df = pd.read_csv(str(confounds_file), sep='\t')

    # set highpass and lowpass
    hp = None
    lp = None

    test_array = np.atleast_2d(np.zeros(10)).T
    func_out = ac.extract_ts(str(bold_file), 
                             str(brainmask_file),
                             str(atlas_file),
                             confounds_df, hp, lp)
    assert np.array_equal(test_array, func_out)


def test_make_corr_matrix():
    import numpy as np

    test_ts_matrix = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]).T

    out_zcorr = ac.make_corr_matrix(test_ts_matrix)

    # rounding just to make it easier on me
    out_zcorr = np.round(out_zcorr, decimals=2)

    test_zcorr = np.array([[0.,  1.07, -1.07],
                           [1.07, 0., -1.07],
                           [-1.07, -1.07, 0.]])

    assert np.array_equal(out_zcorr, test_zcorr)


def test_write_out_corr_matrix(atlas_lut, bold_file, deriv_dir):
    import numpy as np
    import os

    test_zcorr = np.array([[0., 1.07, -1.07],
                           [1.07, 0., -1.07],
                           [-1.07, -1.07, 0.]])

    out_file = ac.write_out_corr_matrix(test_zcorr, str(atlas_lut), str(bold_file), str(deriv_dir))

    assert os.path.isfile(out_file)


def test_proc_matrix(corrMatrix_file):
    import pandas as pd

    column_order = ['session_id',
                    'subject_id',
                    'task_id',
                    'region1-region2',
                    'region1-region3',
                    'region2-region3']

    test_df = pd.DataFrame.from_records([{'session_id': 'pre',
                                          'subject_id': '01',
                                          'task_id': 'rest',
                                          'region1-region2': 1.07,
                                          'region1-region3': -1.07,
                                          'region2-region3': -1.07}])

    test_df = test_df[column_order]

    out_df = ac.proc_matrix(str(corrMatrix_file))

    assert test_df.equals(out_df)


def test_merge_dfs():
    import pandas as pd

    column_order = ['subject_id',
                    'session_id',
                    'task_id',
                    'region1-region2',
                    'region1-region3',
                    'region2-region3']

    inpt_df1 = pd.DataFrame.from_records([{'session_id': 'tst2',
                                           'subject_id': 'tst1',
                                           'task_id': 'ppp',
                                           'region1-region2': 1.07,
                                           'region1-region3': -1.07,
                                           'region2-region3': -1.07}])

    inpt_df2 = pd.DataFrame.from_records([{'session_id': 'tst4',
                                           'subject_id': 'tst3',
                                           'task_id': 'ppp',
                                           'region1-region2': .99,
                                           'region1-region3': -.99,
                                           'region2-region3': -.57}])


    dfs = [inpt_df1, inpt_df2]

    test_df = pd.DataFrame.from_records([{'session_id': 'tst2',
                                          'subject_id': 'tst1',
                                          'task_id': 'ppp',
                                          'region1-region2': 1.07,
                                          'region1-region3': -1.07,
                                          'region2-region3': -1.07},
                                         {'session_id': 'tst4',
                                          'subject_id': 'tst3',
                                          'task_id': 'ppp',
                                          'region1-region2': .99,
                                          'region1-region3': -.99,
                                          'region2-region3': -.57},
                                         ])
    test_df = test_df[column_order]

    out_df = ac.merge_dfs(dfs)

    assert test_df.equals(out_df)


def test_write_out_group_tsv(deriv_dir):
    import pandas as pd
    import os

    test_df = pd.DataFrame.from_records([{'session_id': 'tst2',
                                          'subject_id': 'tst1',
                                          'task_id': 'ppp',
                                          'region1-region2': 1.07,
                                          'region1-region3': -1.07,
                                          'region2-region3': -1.07},
                                         {'session_id': 'tst4',
                                          'subject_id': 'tst3',
                                          'task_id': 'ppp',
                                          'region1-region2': .99,
                                          'region1-region3': -.99,
                                          'region2-region3': -.57},
                                         ])
    out_file = ac.write_out_group_tsv(str(deriv_dir), test_df)

    assert os.path.isfile(out_file)
