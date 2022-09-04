import kipoi
from . import kipoi_model_list

def get_model_names(pattern):
    model_names = kipoi_model_list[kipoi_model_list["model"].str.contains(pattern)]["model"]
    return model_names


def get_model(model_name):
    model = kipoi.get_model(model_name).model
    return model