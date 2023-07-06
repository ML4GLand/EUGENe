from ._fit import fit, fit_sequence_module
try:
    from ._hyperopt import hyperopt
    
    RAY_AVAILABLE = True

except ImportError:
    RAY_AVAILABLE = False
    def no_ray():
        raise ImportError(
            "Install ray to use functionality EUGENe's hyperopt functionality."
        )
    hyperopt = no_ray
    