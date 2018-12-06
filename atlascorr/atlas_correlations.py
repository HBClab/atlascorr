# pybids has to be greater than 0.5
from nipype.interfaces.io import BIDSDataGrabber
from nipype.pipeline import engine as pe
import json
from argparse import ArgumentParser
from nipype.interfaces import utility as niu
import os


def main():
    opts = get_parser().parse_args()

    # define and create the output directory
    outdir = os.path.join(
        os.path.dirname(
            os.path.abspath(opts.deriv_pipeline)), 'atlasCorrelations')
    os.makedirs(outdir, exist_ok=True)

    # define and create the work directory
    workdir = os.path.join(
        os.path.dirname(
            os.path.abspath(opts.deriv_pipeline)), 'work')
    os.makedirs(workdir, exist_ok=True)

    if opts.bids_config_file:
        bids_config = json.load(open(opts.bids_config_file, 'r'))
        infields = [i['name'] for i in bids_config['entities']]
    else:
        infields = None

    if opts.analysis_level == 'participant':
        # initialize participant workflow
        participant_wf = pe.Workflow(name='participant_wf', base_dir=workdir)
        # initialize connectivity workflow
        connectivity_wf = init_connectivity_wf(workdir, outdir, opts.hp, opts.lp, os.path.abspath(opts.atlas_img),
                                               os.path.abspath(opts.atlas_lut), opts.confounds)

        imgs_criteria = {
                            'imgs':
                            {
                                'space': 'MNI152NLin2009cAsym',
                                'modality': 'func',
                                'type': 'preproc'
                            }
                        }

        # add in optional search criteria
        if opts.session:
            imgs_criteria['matrices']['session'] = opts.session
        if opts.task:
            imgs_criteria['matrices']['task'] = opts.task
        if opts.run:
            imgs_criteria['matrices']['run'] = opts.run
        if opts.variant:
            imgs_criteria['matrices']['variant'] = opts.variant

        input_node = pe.Node(
            BIDSDataGrabber(
                infields=infields,
                output_query=imgs_criteria,
                base_dir=os.path.abspath(opts.deriv_pipeline)),
            name='input_node')

        participant_wf.connect([
            (input_node, connectivity_wf,
                [('imgs', 'input_node.img')]),
        ])

        # run the participant workflow
        participant_wf.run()

    elif opts.analysis_level == 'group':

        # set the input dir (assumed participant level already run).
        input_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(opts.deriv_pipeline)), 'atlasCorrelations')
        # catch if directory doesn't exist
        if not os.path.isdir(input_dir):
            raise OSError('DOES NOT EXIST: {input_dir}'.format(input_dir=input_dir))

        # add in optional search criteria
        matrices_criteria = {
            'matrices':
                {
                    'space': 'MNI152NLin2009cAsym',
                    'modality': 'func',
                    'type': 'corrMatrix',
                }
        }
        if opts.session:
            matrices_criteria['matrices']['session'] = opts.session
        if opts.task:
            matrices_criteria['matrices']['task'] = opts.task
        if opts.run:
            matrices_criteria['matrices']['run'] = opts.run
        if opts.variant:
            matrices_criteria['matrices']['variant'] = opts.variant

        group_wf = pe.Workflow(name='group_wf', base_dir=workdir)

        group_collection_wf = init_group_collection_wf(work_dir=workdir,
                                                       outdir=input_dir)
        input_node = pe.Node(
            BIDSDataGrabber(
                infields=infields,
                output_query=matrices_criteria,
                base_dir=input_dir),
            name='input_node')

        group_wf.connect([
            (input_node, group_collection_wf,
             [('matrices', 'input_node.matrix_tsv')]),
        ])

        group_wf.run()

    else:
        raise NameError('specify either participant or group for analysis level')


def get_parser():
    """Build parser object"""
    parser = ArgumentParser(description='atlas_correlations')
    parser.add_argument('--deriv-pipeline', '-d', action='store', required=True,
                        help='input derivative directory (e.g. fmriprep). '
                             'I assume the inputs are in MNI space.')
    parser.add_argument('--atlas-img', '-a', action='store',
                        help='input atlas nifti')
    parser.add_argument('--atlas-lut', '-l', action='store', required=True,
                        help='atlas look up table formatted with the columns: '
                             'index, regions')
    parser.add_argument('--confounds', '-c', action='store', nargs='+',
                        help='names of confounds to be included in analysis')
    parser.add_argument('analysis_level', choices=['participant', 'group'],
                        help='run participant level analysis, or aggregate '
                             'group level results')
    parser.add_argument('--bids-config-file', action='store', default=None,
                        help='config file to pass into BIDSDataGrabber')
    parser.add_argument('--participant_label', '--participant-label',
                        action='store', nargs='+',
                        help='one or more participant identifiers with the '
                             'sub- prefix removed')
    parser.add_argument('--hp', action='store', default=None,
                        help='highpass filter to apply to the data')
    parser.add_argument('--lp', action='store', default=None,
                        help='lowpass filter to apply to the data')
    parser.add_argument('--variant', action='store',
                        help='only analyze files with a specific variant label')
    parser.add_argument('--run', action='store',
                        help='only analyze files with a specific run label')
    parser.add_argument('--session', action='store',
                        help='only analyze files with a specific session label')
    parser.add_argument('--task', action='store',
                        help='only analyze files with a specific task label')
    return parser


def init_connectivity_wf(work_dir, output_dir, hp, lp,
                         atlas_img, atlas_lut, confounds):
    """
    Generates a connectivity matrix for a bold file

    .. workflow::
        :graph2use: orig
        :simple_form: yes

    from atlascorr.atlas_correlations import init_connectivity_wf
    wf = init_connectivity_wf(
        work_dir='.',
        output_dir='.',
        hp=None,
        lp=None,
        atlas_img='',
        atlas_lut='',
        confounds=[''],
    )

    Parameters
    ----------
    work_dir : str
        full path to directory where intermediate files will be written
    output_dir : str
        full path to directory where output files will be written
    hp : float or None
        high pass filter (frequencies higher than this pass)
    lp : float or None
        low pass filter (frequencies lower than this pass)
    atlas_img : str
        full path and name of the atlas file
    atlas_lut : str
        full path and name to atlas lookup tsv with two columns
        (regions and index)
    confounds : list
        list of confounds to include in the model

    Inputs
    ------
    img : str
        full path and name of the bold file
    atlas_img : str
        full path and name of the atlas file
    atlas_lut : str
        full path and name to atlas lookup tsv with two columns
        (regions and index)

    Outputs
    -------
    dst : str
        full path and name of the correlation matrix
    """
    connectivity_wf = pe.Workflow(name='connectivity_wf')
    connectivity_wf.base_dir = work_dir

    input_node = pe.MapNode(
        niu.IdentityInterface(
            fields=['img', 'atlas_img', 'atlas_lut']),
        iterfield=['img'],
        name='input_node')
    input_node.inputs.atlas_img = atlas_img
    input_node.inputs.atlas_lut = atlas_lut

    get_files_node = pe.MapNode(
        niu.Function(
            function=get_files,
            input_names=['img'],
            output_names=['confounds', 'brainmask']),
        iterfield=['img'],
        name='get_files_node')

    confounds2df_node = pe.MapNode(
        niu.Function(
            function=proc_confounds,
            input_names=['confounds', 'confound_file'],
            output_names=['confounds_df']),
        iterfield=['confound_file'],
        name='confounds2df_node')
    confounds2df_node.inputs.confounds = confounds

    extract_ts_node = pe.MapNode(
        niu.Function(
            function=extract_ts,
            input_names=['img',
                         'brainmask',
                         'atlas_img',
                         'confounds_df',
                         'hp',
                         'lp'],
            output_names=['ts_matrix']),
        iterfield=['img', 'confounds_df', 'brainmask'],
        name='extract_ts_node')

    # initialize highpass and lowpass
    extract_ts_node.inputs.lp = lp
    extract_ts_node.inputs.hp = hp

    make_corr_matrix_node = pe.MapNode(
        niu.Function(
            function=make_corr_matrix,
            input_names=['ts_matrix'],
            output_names=['zcorr_matrix']),
        iterfield=['ts_matrix'],
        name='make_corr_matrix_node')

    write_out_corr_matrix_node = pe.MapNode(
        niu.Function(
            function=write_out_corr_matrix,
            input_names=['corr_matrix', 'atlas_lut', 'img', 'output_dir'],
            output_names=['matrix_tsv']),
        iterfield=['corr_matrix', 'img'],
        name='write_out_corr_matrix_node')
    write_out_corr_matrix_node.inputs.output_dir = output_dir

    connectivity_wf.connect([
        (input_node, get_files_node,
            [('img', 'img')]),
        (get_files_node, confounds2df_node,
            [('confounds', 'confound_file')]),
        (get_files_node, extract_ts_node,
            [('brainmask', 'brainmask')]),
        (confounds2df_node, extract_ts_node,
            [('confounds_df', 'confounds_df')]),
        (input_node, extract_ts_node,
            [('atlas_img', 'atlas_img'),
             ('img', 'img')]),
        (extract_ts_node, make_corr_matrix_node,
            [('ts_matrix', 'ts_matrix')]),
        (make_corr_matrix_node, write_out_corr_matrix_node,
            [('zcorr_matrix', 'corr_matrix')]),
        (input_node, write_out_corr_matrix_node,
            [('atlas_lut', 'atlas_lut'),
             ('img', 'img')]),

    ])

    return connectivity_wf


def init_group_collection_wf(work_dir, outdir):
    """
    Combines correlation matrices derived from the individual
    bold files.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

    from atlascorr.atlas_correlations import init_group_collection_wf
    wf = init_group_collection_wf(
        work_dir='.',
        outdir='.',
    )

    Parameters
    ----------
    work_dir : str
        full path to directory where intermediate files will be written
    outdir : str
        full path to directory where the group tsv will be written

    Inputs
    ------
    matrix_tsv : str
        full path and name to correlation matrix
    """
    group_collection_wf = pe.Workflow(name='group_collection_wf')
    group_collection_wf.base_dir = work_dir

    input_node = pe.MapNode(
        niu.IdentityInterface(
            fields=['matrix_tsv']),
        iterfield=['matrix_tsv'],
        name='input_node')

    matrix_proc_node = pe.MapNode(
        niu.Function(
            function=proc_matrix,
            input_names=['matrix_tsv'],
            output_names=['participant_df']),
        iterfield=['matrix_tsv'],
        name='matrix_proc_node')

    merge_dfs_node = pe.Node(
        niu.Function(
            function=merge_dfs,
            input_names=['dfs'],
            output_names=['df']),
        name='merge_dfs_node')

    write_out_group_tsv_node = pe.Node(
        niu.Function(
            function=write_out_group_tsv,
            input_names=['outdir', 'df'],
            output_names=['out_file']),
        name='write_out_group_tsv_node')
    write_out_group_tsv_node.inputs.outdir = outdir

    group_collection_wf.connect([
        (input_node, matrix_proc_node,
            [('matrix_tsv', 'matrix_tsv')]),
        (matrix_proc_node, merge_dfs_node,
            [('participant_df', 'dfs')]),
        (merge_dfs_node, write_out_group_tsv_node,
            [('df', 'df')]),
    ])

    return group_collection_wf


def get_files(img):
    """
    Find the brainmask and confound files given the bold file.

    Parameters
    ----------
    img : str
        full path and name of the bold file

    Returns
    -------
    confound : str
        full path and name of the confounds file
    brainmask : str
        full path and name of the brainmask file
    """
    import re
    import os
    PROC_EXPR = re.compile(
        r'^(?P<path>.*/)?'
        r'(?P<subject_id>sub-[a-zA-Z0-9]+)'
        r'(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
        r'(_(?P<task_id>task-[a-zA-Z0-9]+))?'
        r'(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
        r'(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
        r'(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        r'_bold'
        r'(_(?P<space_id>space-[a-zA-Z0-9]+))?'
        r'(_(?P<variant_id>variant-[a-zA-Z0-9]+))?'
        r'_preproc.nii.gz')

    def get_confound(img):
        CONF_REPL = (r'\g<path>'
                     r'\g<subject_id>'
                     r'_\g<session_id>'
                     r'_\g<task_id>'
                     r'_\g<run_id>'
                     r'_bold_confounds.tsv')
        conf_tmp = PROC_EXPR.sub(CONF_REPL, img)
        conf = re.sub('_+', '_', conf_tmp)
        if os.path.isfile(conf):
            return conf
        else:
            raise IOError('cannot find {conf}'.format(conf=conf))

    def get_brainmask(img):
        MASK_REPL = (r'\g<path>'
                     r'\g<subject_id>'
                     r'_\g<session_id>'
                     r'_\g<task_id>'
                     r'_\g<run_id>'
                     r'_bold_\g<space_id>_brainmask.nii.gz')
        bmask = PROC_EXPR.sub(MASK_REPL, img)
        bmask = re.sub('_+', '_', bmask)
        if os.path.isfile(bmask):
            return bmask
        else:
            raise IOError('cannot find {bmask}'.format(bmask=bmask))
    confound = get_confound(img)
    brainmask = get_brainmask(img)
    return confound, brainmask


def proc_confounds(confounds, confound_file):
    """
    Filter confounds file to selected confounds &
    replaces "n/a"s in confounds file with the mean.

    Parameters
    ----------
    confounds : list
        list of confounds to include in the model
    confounds_file : str
        full path and name of the confounds file

    Returns
    -------
    confounds_df : pandas.core.frame.DataFrame
        dataframe containing the selected confounds
    """
    import pandas as pd
    import numpy as np
    confounds_df = pd.read_csv(confound_file, sep='\t', na_values='n/a')
    if 'FramewiseDisplacement' in confounds:
        confounds_df['FramewiseDisplacement'] = confounds_df['FramewiseDisplacement'].fillna(
                                np.mean(confounds_df['FramewiseDisplacement']))
    return confounds_df[confounds]


def extract_ts(img, brainmask, atlas_img, confounds_df, hp=None, lp=None):
    """
    Extract timeseries from each region of interest described by an atlas.

    Parameters
    ----------
    img : str
        full path and name of the bold file
    brainmask : str
        full path and name of the brainmask file
    atlas_img : str
        full path and name of the atlas file
    confounds_df : pandas.core.frame.DataFrame
        dataframe containing confound measures
    hp : float or None
        high pass filter (frequencies higher than this pass)
    lp : float or None
        low pass filter (frequencies lower than this pass)

    Returns
    -------
    signals : numpy.ndarray
        2D numpy array with each column representing an atlas region
        and each row representing a volume (time point)
    """
    from nilearn.input_data import NiftiLabelsMasker
    if hp:
        hp = float(hp)
    if lp:
        lp = float(lp)
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize=True, mask_img=brainmask,
        low_pass=lp, high_pass=hp, t_r=2.0)
    return masker.fit_transform(img, confounds=confounds_df.values)


def make_corr_matrix(ts_matrix):
    """
    Make a symmetric pearson's r->z transforme correlation matrix.

    Parameters
    ----------
    ts_matrix : numpy.ndarray
        2D numpy array with each column representing an atlas region
        and each row representing a volume (time point)

    Returns
    -------
    zcorr_matrix : numpy.ndarray
        2D symmetric matrix measuring region-region correlations
        main diagnal is all zeros
    """
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np

    def fisher_r_to_z(r):
        import math
        if r == 1.:
            return 0.
        else:
            return math.log((1. + r)/(1. - r))/2.
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_matrix = correlation_measure.fit_transform([ts_matrix])[0]
    vfisher_r_to_z = np.vectorize(fisher_r_to_z)
    # fisher's r to z
    zcorr_matrix = vfisher_r_to_z(corr_matrix)
    return zcorr_matrix


def write_out_corr_matrix(corr_matrix, atlas_lut, img, output_dir):
    """
    Write out a symmetric correlation matrix using BIDS naming conventions

    Parameters
    ----------
    corr_matrix : numpy.ndarray
        2D symmetric matrix measuring region-region correlations
        main diagnal is all zeros
    atlas_lut : str
        full path and name to atlas lookup tsv with two columns
        (regions and index)
    img : str
        full path and name of the bold file
    output_dir : str
        full path to the base directory where all correlation matrices
        will be written out to.

    Returns
    -------
    dst : str
        full path and name of the correlation matrix
    """
    import pandas as pd
    import os
    import re

    PROC_EXPR = re.compile(
        r'^(?P<path>.*/)?'
        r'(?P<subject_id>sub-[a-zA-Z0-9]+)'
        r'(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
        r'(_(?P<task_id>task-[a-zA-Z0-9]+))?'
        r'(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
        r'(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
        r'(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        r'_bold'
        r'(_(?P<space_id>space-[a-zA-Z0-9]+))?'
        r'(_(?P<variant_id>variant-[a-zA-Z0-9]+))?'
        r'_preproc.nii.gz')

    name_dict = PROC_EXPR.search(img).groupdict()

    bids_output_dir = os.path.join(output_dir,
                                   name_dict['subject_id'],
                                   name_dict['session_id'],
                                   'func')
    os.makedirs(bids_output_dir, exist_ok=True)

    fname = '_'.join([name_dict['subject_id'], name_dict['session_id']])

    key_order = ['task_id', 'acq_id', 'rec_id', 'run_id', 'space_id', 'variant_id']

    for key in key_order:
        if name_dict[key]:
            fname = '_'.join([fname, name_dict[key]])

    dst = os.path.join(bids_output_dir, fname + '_corrMatrix.tsv')

    atlas_lut_df = pd.read_csv(atlas_lut, sep='\t')
    regions = atlas_lut_df['regions']
    corr_matrix_df = pd.DataFrame(corr_matrix, index=regions, columns=regions)
    corr_matrix_df.to_csv(dst, sep='\t')
    return dst


def proc_matrix(matrix_tsv):
    """
    Vectorize symmetric correlation matrix so that
    each unique region-region correlation gets a column.

    Parameters
    ----------
    matrix_tsv : str
        full path and name of the correlation matrix

    Returns
    -------
    flat_df : pandas.core.frame.DataFrame
        a flat dataframe that is one entry and has as many columns
        as there are unique region-region pairs
    """
    import pandas as pd
    import numpy as np
    import re
    import os
    # process data:
    # read in tsv into a pandas dataframe
    tmp_df = pd.read_csv(matrix_tsv, sep='\t', index_col=0)
    # make the dataframe into a numpy array
    tmp_arr = tmp_df.as_matrix()
    # get the upper triangle (excluding the diagonal
    upper_triangle_idx = np.triu_indices(len(tmp_df), 1)
    # extract the values from the symmetric 2D matrix into a 1D matrix
    flat_arr = tmp_arr[upper_triangle_idx]

    # collector for header names
    header_list = []
    # this has to mutable to not repeat combinations
    row_headers = list(tmp_df.index)
    for col_header in tmp_df.columns.values:
        row_headers.remove(col_header)
        for row_header in row_headers:
            header_list.append('-'.join([col_header, row_header]))

    # makes a wide dataframe with one entry
    data_df = pd.DataFrame(data=np.atleast_2d(flat_arr), columns=header_list)

    # process filename
    MAT_EXPR = re.compile(
        r'^(?P<path>.*/)?'
        r'(?P<subject_id>sub-[a-zA-Z0-9]+)'
        r'(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
        r'(_(?P<task_id>task-[a-zA-Z0-9]+))?'
        r'(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
        r'(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
        r'(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        r'(_(?P<space_id>space-[a-zA-Z0-9]+))?'
        r'(_(?P<variant_id>variant-[a-zA-Z0-9]+))?'
        r'_corrMatrix.tsv')
    name_dict = MAT_EXPR.search(os.path.basename(matrix_tsv)).groupdict()
    info_dict = {k: v.split('-')[1] for k, v in name_dict.items() if v is not None}
    info_df = pd.DataFrame.from_records([info_dict])

    # returns a one row many column dataframe
    return pd.concat([info_df, data_df], axis=1)


def merge_dfs(dfs):
    """
    Merge a list of dataframes where each contains one row
    showing all unique region-region pairs.

    Parameters
    ----------
    dfs : list
        list of dataframes where each contains one row
        showing all unique region-region pairs

    Returns
    -------
    out_df : pandas.core.frame.DataFrame
        merged dataframe where each row represents a unique scan
    """
    import pandas as pd
    out_df = pd.concat(dfs, copy=False, ignore_index=True)
    headers = list(out_df.columns.values)
    # if any of these columns exist in the dataframe, move them to the front
    if 'variant_id' in headers:
        headers.insert(0, headers.pop(headers.index('variant_id')))
    if 'space_id' in headers:
        headers.insert(0, headers.pop(headers.index('space_id')))
    if 'run_id' in headers:
        headers.insert(0, headers.pop(headers.index('run_id')))
    if 'rec_id' in headers:
        headers.insert(0, headers.pop(headers.index('rec_id')))
    if 'acq_id' in headers:
        headers.insert(0, headers.pop(headers.index('acq_id')))
    if 'task_id' in headers:
        headers.insert(0, headers.pop(headers.index('task_id')))
    if 'session_id' in headers:
        headers.insert(0, headers.pop(headers.index('session_id')))
    if 'subject_id' in headers:
        headers.insert(0, headers.pop(headers.index('subject_id')))

    out_df = out_df[headers]

    return out_df


def write_out_group_tsv(outdir, df):
    """
    outdir : str
        full path to the output directory for the group tsv
    df : pandas.core.frame.DataFrame
        dataframe where each row represents a unique scan
        and each column is a unique region-region pair
    """
    import os
    out_file = os.path.join(outdir, 'group.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return out_file


if __name__ == '__main__':
    main()
