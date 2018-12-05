import atlas_correlations as ac


def test_get_files():
    # FMRIPREP outputs
    img_file = './test_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_bold_preproc.nii.gz'
    confound = './test_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_bold_confounds.tsv'
    brainmask = './test_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_bold_brainmask.nii.gz'
    assert ac.get_files(img_file) == (confound, brainmask)


def test_get_confounds():
    import pandas as pd

    confounds = ['FramewiseDisplacement', 'CSF']
    confounds_file = './test_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_bold_confounds.tsv'
    confounds_df = pd.DataFrame({'FramewiseDisplacement': [1.] * 10,
                                 'CSF': [1.] * 10,
                                 })
    # ensure the columns are the same order
    confounds_df = confounds_df[confounds]
    assert confounds_df.equals(ac.proc_confounds(confounds, confounds_file))


def test_extract_ts():
    import nibabel as nib
    import numpy as np
    import pandas as pd
    # setup the img
    data = np.ones((5, 5, 5, 10), dtype=np.float)
    img = nib.Nifti1Image(data, np.eye(4))

    # brainmask
    mask = np.ones((5, 5, 5), dtype=np.int16)
    brainmask = nib.Nifti1Image(mask, np.eye(4))

    # atlas_img
    atlas = np.ones((5, 5, 5), dtype=np.int16)
    atlas_img = nib.Nifti1Image(atlas, np.eye(4))

    # setup confounds
    confounds_df = pd.DataFrame({'FramewiseDisplacement': [1.] * 10,
                                 'CSF': [1.] * 10,
                                 })
    # ensure the columns are the same order
    confounds = ['FramewiseDisplacement', 'CSF']
    confounds_df = confounds_df[confounds]

    # set highpass and lowpass
    hp = None
    lp = None

    test_array = np.atleast_2d(np.zeros(10)).T
    func_out = ac.extract_ts(img, brainmask, atlas_img, confounds_df, hp, lp)
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


def test_write_out_corr_matrix():
    import numpy as np
    import os

    test_zcorr = np.array([[0., 1.07, -1.07],
                           [1.07, 0., -1.07],
                           [-1.07, -1.07, 0.]])
    atlas_lut = './test_data/atlas_lut.tsv'
    img = './test_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_bold_preproc.nii.gz'
    output_dir = './test_data'

    out_file = ac.write_out_corr_matrix(test_zcorr, atlas_lut, img, output_dir)

    assert os.path.isfile(out_file)

    # cleanup
    os.remove(out_file)


def test_proc_matrix():
    import pandas as pd

    matrix_tsv = './test_data/proc_data/sub-tst1/ses-tst2/func/sub-tst1_ses-tst2_task-ppp_corrMatrix.tsv'

    column_order = ['session_id',
                    'subject_id',
                    'task_id',
                    'region1-region2',
                    'region1-region3',
                    'region2-region3']

    test_df = pd.DataFrame.from_records([{'session_id': 'tst2',
                                          'subject_id': 'tst1',
                                          'task_id': 'ppp',
                                          'region1-region2': 1.07,
                                          'region1-region3': -1.07,
                                          'region2-region3': -1.07}])

    test_df = test_df[column_order]

    out_df = ac.proc_matrix(matrix_tsv)

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


def test_write_out_group_tsv():
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
    outdir = './test_data'

    out_file = ac.write_out_group_tsv(outdir, test_df)

    assert os.path.isfile(out_file)

    # cleanup
    os.remove(out_file)