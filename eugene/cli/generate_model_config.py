import sys
import inspect
import importlib
import questionary
import yaml


def infer_type(user_input):
    if "[" in user_input:
        try:
            return [int(i) for i in user_input.strip("][").split(", ")]
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


def _generate_model_config(model_type):
    config_params = {}
    questionary.print(f"Extracting hyperparameters for {model_type} architecture")
    module = getattr(importlib.import_module("eugene.models"), model_type)
    higher_params = []
    init_params = inspect.signature(module.__init__).parameters
    for key in init_params.keys():
        param_name = init_params[key].name
        param_default = (
            init_params[key].default if type(init_params[key].default) == str else ""
        )
        if param_name == "self":
            continue
        if param_name in ["fc_kwargs", "conv_kwargs", "rnn_kwargs"]:
            module_config = {}
            questionary.print(f"Extracting {param_name} parameters")
            if param_name == "fc_kwargs":
                mod = getattr(
                    importlib.import_module("eugene.models.base"),
                    "BasicFullyConnectedModule",
                )
                module_params = inspect.signature(mod.__init__).parameters
            elif param_name == "conv_kwargs":
                mod = getattr(
                    importlib.import_module("eugene.models.base"), "BasicConv1D"
                )
                module_params = inspect.signature(mod.__init__).parameters
            elif param_name == "rnn_kwargs":
                mod = getattr(
                    importlib.import_module("eugene.models.base"), "BasicRecurrent"
                )
                module_params = inspect.signature(mod.__init__).parameters
            for module_key in module_params:
                module_name = module_params[module_key].name
                module_default = (
                    module_params[module_key].default
                    if module_params[module_key].default != inspect._empty
                    else ""
                )
                if module_name == "self":
                    continue
                if module_name in higher_params:
                    continue
                user_mod_param = infer_type(
                    questionary.text(f"{module_name} (default: {module_default})").ask()
                )
                if user_mod_param == "":
                    continue
                module_config[module_name] = user_mod_param
            config_params[param_name] = module_config
            continue
        user_param = infer_type(
            questionary.text(f"{param_name} (default: {param_default})").ask()
        )
        if user_param == "":
            continue
        higher_params.append(param_name)
        config_params[param_name] = user_param
    return config_params


if __name__ == "__main__":
    out = sys.argv[1]
    model_type = questionary.select(
        "Model type", choices=["FCN", "CNN", "RNN", "Hybrid", "DeepBind"]
    ).ask()
    config_params = _generate_model_config(model_type)
    final_config = {"model": config_params}
    print(final_config)
    with open(out, "w") as yaml_file:
        yaml.dump(final_config, yaml_file, sort_keys=False, default_flow_style=False)
