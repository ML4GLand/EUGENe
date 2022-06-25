import yaml
import inspect
import questionary
import importlib

def infer_type(user_input):
    if "[" in user_input:
        try:
            return [int(i) for i in user_input.strip('][').split(', ')]
        except:
            pass
    if "." in user_input:
        try:
            return float(user_input)
        except:
            pass
    else:
        try:
            return int(user_input)
        except:
            pass
    return user_input

def generate_model_config(model_type):
    config_params = {}
    questionary.print("Extracting model parameters")
    if model_type in ["fcn", "cnn", "rnn"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type}"), model_type.upper())
    elif model_type in ["hybrid"]:
        module = getattr(importlib.import_module(f"eugene.models.{model_type.lower()}"), model_type.lower())
    higher_params = []
    init_params = inspect.signature(module.__init__).parameters
    for key in init_params.keys():
        param_name = init_params[key].name
        param_default = init_params[key].default if type(init_params[key].default) == str else ""
        if param_name == "self":
            continue
        if param_name in ["fc_kwargs", "conv_kwargs", "rnn_kwargs"]:
            module_config = {}
            questionary.print(f"Extracting {param_name} parameters")
            if param_name == "fc_kwargs":
                mod = getattr(importlib.import_module("claim.modules._base_modules"), "BasicFullyConnectedModule")
                module_params = inspect.signature(mod.__init__).parameters
            elif param_name == "conv_kwargs":
                mod = getattr(importlib.import_module("claim.modules._base_modules"), "BasicConv1D")
                module_params = inspect.signature(mod.__init__).parameters
            elif param_name == "rnn_kwargs":
                mod = getattr(importlib.import_module("claim.modules._base_modules"), "BasicRecurrent")
                module_params = inspect.signature(mod.__init__).parameters
            for module_key in module_params:
                module_name = module_params[module_key].name
                module_default = module_params[module_key].default if module_params[module_key].default != inspect._empty else ""
                if module_name == "self":
                    continue
                if module_name in higher_params:
                    continue
                user_mod_param = infer_type(questionary.text(f"{module_name} (default: {module_default})").ask())
                if user_mod_param == "":
                    continue
                module_config[module_name] = user_mod_param
            config_params[param_name] = module_config
            continue
        user_param = infer_type(questionary.text(f"{param_name} (default: {param_default})").ask())
        if user_param == "":
            continue
        higher_params.append(param_name)
        config_params[param_name] = user_param
    return config_params


def generate_data_config():
    pass


model_type = questionary.select("Select model", choices=["fcn", "cnn", "rnn", "hybrid"]).ask()
yml_dict = generate_model_config(model_type)
print(yml_dict)
with open('config.yaml', 'w') as yaml_file:
    yaml.dump(yml_dict, yaml_file, sort_keys=False, default_flow_style=False)
