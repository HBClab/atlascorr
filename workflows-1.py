from atlascorr.atlas_correlations import init_connectivity_wf
wf = init_connectivity_wf(
    work_dir='.',
    output_dir='.',
    hp=None,
    lp=None,
    atlas_img='',
    atlas_lut='',
    confounds=[''])