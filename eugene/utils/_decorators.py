import numpy as np
import inspect
from functools import wraps
from ..dataload import SeqData


def get_default_args(func):
    """
    Helper function to get default arguments for a function
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def track(func):
    """
    Track changes in SeqData object after applying function.
    """
    @wraps(func)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(func)
        kwargs.update(kwds)
        if "sdata" in kwargs and kwargs["sdata"] is not None:
            sdata = kwargs["sdata"]
        else:
            if len(args) > 0:
                if type(args[0]) == SeqData:
                    sdata = args[0]
                elif args[0].__repr__()[:7] == "SeqData":
                    sdata = args[0]
                elif isinstance(args[0], SeqData):
                    sdata = args[0]
                elif type(args[1]) == SeqData:
                    sdata = args[1]
                elif args[1].__repr__()[:7] == "SeqData":
                    sdata = args[1]
                elif isinstance(args[1], SeqData):
                    sdata = args[1]
                else:
                    raise ValueError("No SeqData object found for tracking.")
        old_attr = list_attributes(sdata)

        if kwargs["copy"]:
            out_sdata = func(*args, **kwds)
            new_attr = list_attributes(out_sdata)
        else:
            func(*args, **kwds)
            new_attr = list_attributes(sdata)

        # Print differences between new and oldsdata
        out = ""
        out += "SeqData object modified:"

        modified = False
        for attr in old_attr.keys():
            if attr == "n_obs":
                if old_attr["n_obs"] != new_attr["n_obs"]:
                    out += f"\n\tn_obs: {old_attr['n_obs']} -> {new_attr['n_obs']}"
                    modified = True
            elif attr in ["seqs", "names", "rev_seqs", "ohe_seqs", "ohe_rev_seqs"]:
                if not np.array_equal(old_attr[attr], new_attr[attr]):
                    out += f"\n\t{attr}: {old_attr[attr]} -> {len(new_attr[attr])} {attr} added"
                    modified = True
            elif attr in ["seqs_annot", "pos_annot"]:
                if old_attr[attr] is None or new_attr[attr] is None:
                    continue
                else:
                    # print(old_attr[attr])
                    removed = list(old_attr[attr] - new_attr[attr])
                    added = list(new_attr[attr] - old_attr[attr])

                    if len(removed) > 0 or len(added) > 0:
                        modified = True
                        out += f"\n    {attr}:"
                        if len(removed) > 0:
                            out += f"\n        - {', '.join(removed)}"
                        if len(added) > 0:
                            out += f"\n        + {', '.join(added)}"

        if modified:
            print(out)
        return out_sdata if kwargs["copy"] else None
    return wrapper


def list_attributes(sdata):
    found_attr = dict(n_obs=sdata.n_obs)
    for attr in [
        "seqs",
        "names",
        "rev_seqs",
        "ohe_seqs",
        "ohe_rev_seqs",
        "seqs_annot",
        "pos_annot",
        "seqsm",
    ]:
        if getattr(sdata, attr) is None:
            found_attr[attr] = None
            continue
        if attr in ["seqs", "names", "rev_seqs", "ohe_seqs", "ohe_rev_seqs"]:
            vals = getattr(sdata, attr)
            found_attr[attr] = vals
        elif attr in ["seqs_annot", "pos_annot"]:
            keys = set(getattr(sdata, attr).keys())
            found_attr[attr] = keys
    return found_attr
