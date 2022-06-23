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

def generate_data_config(dataloader):
    config_params = {}
    questionary.print("Extracting dataloader arguments")
    module = getattr(importlib.import_module(f"eugene.dataloading.dataloaders._{dataloader}DataModule"), f"{dataloader}DataModule")
    higher_params = []
    init_params = inspect.signature(module.__init__).parameters
    for key in init_params.keys():
        param_name = init_params[key].name
        param_default = init_params[key].default if init_params[key].default != inspect._empty else ""
        if param_name == "self":
            continue
        if param_name == "load_kwargs":
            module_config = {}
            file_type = questionary.select(f"Which file types do you want to load?", choices=["csv", "numpy", "fasta"]).ask()
            func = getattr(importlib.import_module("eugene.dataloading._io"), f"load_{file_type}")
            #module_params = inspect.signature(mod.__init__).parameters
            module_params = inspect.getfullargspec(func)
            #print(module_params)
            for module_key in range(len(module_params.args)):
                #print(module_key)
                module_name = module_params.args[module_key]
                #module_default = module_params.defaults[module_key] if module_params.defaults[module_key] != inspect._empty else ""
                module_default = "" # needs a fix, FullArgSpec does not have defaults for required params
                #print(module_default)
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


#loader_type = questionary.select("Select dataloader", choices=["Seq"]).ask()
#yml_dict = generate_data_config(loader_type)
#print(yml_dict)
#with open('config.yaml', 'w') as yaml_file:
#    yaml.dump(yml_dict, yaml_file, sort_keys=False, default_flow_style=False)