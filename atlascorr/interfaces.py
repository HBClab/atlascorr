from nipype.interfaces.io import BIDSDataGrabber, BIDSDataGrabberInputSpec
from nipype.interfaces.base import traits, isdefined, Undefined
# import bids as bidslayout
from nipype import logging
from packaging import version

have_pybids = True
try:
    import bids
    bids_ver = version.parse(bids.__version__)
except ImportError:
    have_pybids = False

if have_pybids:
    try:
        from bids import layout as bidslayout
    except ImportError:
        from bids import grabbids as bidslayout

iflogger = logging.getLogger('nipype.interface')


class BIDSDataGrabberInputSpecPatch(BIDSDataGrabberInputSpec):
    derivatives = traits.Bool(desc='use derivative entities in layout')


class BIDSDataGrabberPatch(BIDSDataGrabber):
    input_spec = BIDSDataGrabberInputSpecPatch

    def _list_outputs(self):
        exclude = None
        if self.inputs.strict:
            exclude = ['derivatives/', 'code/', 'sourcedata/']
        if self.inputs.derivatives:
            domains = ['bids', 'derivatives']
        else:
            domains = ['bids']

        if bids_ver < version.parse('0.5'):
            raise ImportError("pybids must be >= 0.5")
        elif bids_ver >= version.parse('0.5') and bids_ver < version.parse('0.6'):
            layout = bidslayout.BIDSLayout(self.inputs.base_dir, config=domains, exclude=exclude)
        else:
            layout = bidslayout.BIDSLayout((self.inputs.base_dir, domains), exclude=exclude)

        # If infield is not given nm input value, silently ignore
        filters = {}
        for key in self._infields:
            value = getattr(self.inputs, key)
            if isdefined(value):
                filters[key] = value

        outputs = {}
        for key, query in self.inputs.output_query.items():
            args = query.copy()
            args.update(filters)
            filelist = layout.get(return_type=self.inputs.return_type, **args)
            if len(filelist) == 0:
                msg = 'Output key: %s returned no files' % key
                if self.inputs.raise_on_empty:
                    raise IOError(msg)
                else:
                    iflogger.warning(msg)
                    filelist = Undefined

            outputs[key] = filelist
        return outputs
