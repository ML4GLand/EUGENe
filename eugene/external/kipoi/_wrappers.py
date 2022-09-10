def get_model_names(pattern):
    try:
        import kipoi
    except ImportError:
        raise ImportError('Please install kipoi dependencies and git `pip install eugene[kipoi]` and `conda install -c anaconda git`')
    kipoi_model_list = kipoi.list_models()
    model_names = kipoi_model_list[kipoi_model_list["model"].str.contains(pattern)]["model"]
    return model_names


def get_model(model_name):
    try:
        import kipoi
    except ImportError:
        raise ImportError('Please install kipoi dependencies and git `pip install eugene[kipoi]` and `conda install -c anaconda git`')
    model = kipoi.get_model(model_name).model
    return model