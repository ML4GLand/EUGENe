import sys
import questionary
from ._utils import generate_model_config

if __name__ == "__main__":
    out = sys.argv[1]
    model_type = questionary.select("Model type", choices=["FCN", "CNN", "RNN", "Hybrid", "DeepBind"]).ask()
    config_params = generate_model_config(model_type)
    with open(out, 'w') as yaml_file:
        config_params.dump(config_params, yaml_file, sort_keys=False, default_flow_style=False)
