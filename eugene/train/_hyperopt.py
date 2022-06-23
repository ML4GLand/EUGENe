# Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# PL
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI

# EUGENE
from ..dataloading.dataloaders import SeqDataModule
from ..models import CNN


# Hyperoptimization objective function
def objective(trial: optuna.trial.Trial, pl_cli):
    config = cli.config
    
    model_kwgs = config["model"]
    if config["hyperopt_model"]:
        # Conv layer hps
        conv_kwgs = model_kwgs["conv_kwargs"]
        conv_channels, conv_kernels, pool_kernels = [conv_kwgs["channels"][0]], [], []
        for i in range(0, len(conv_kwgs["conv_kernels"])):
            conv_channels.append(trial.suggest_categorical("conv_channels_{}".format(i), 
                                                           [x*4 if x != 0 else 1 for x in range(0, conv_kwgs["channels"][i+1]//4 + 1)]))
            conv_kernels.append(trial.suggest_categorical("conv_kernel_{}".format(i), 
                                                          [x*4 if x != 0 else 1 for x in range(0, conv_kwgs["conv_kernels"][i]//4 + 1)]))
            pool_kernels.append(trial.suggest_categorical("pool_kernel_{}".format(i),
                                                          [x*2 if x != 0 else 1 for x in range(0, conv_kwgs["pool_kernels"][i]//2 + 1)]))
        conv_dropout = trial.suggest_float("conv_dropout_rate", 0.1, conv_kwgs["dropout_rates"])
        conv_batchnorm = trial.suggest_categorical("conv_batchnorm", [True, False])
        cnn = dict(input_len=conv_kwgs["input_len"], 
                   channels=conv_channels, 
                   conv_kernels=conv_kernels, 
                   pool_kernels=pool_kernels,
                   dropout_rates=conv_dropout,
                   batchnorm=conv_batchnorm)

        # RNN layer hps
        rnn_kwgs = model_kwgs["rnn_kwargs"]
        rnn_output_dim = trial.suggest_categorical("rnn_output_dim", [x*64 if x != 0 else 1 for x in range(0, rnn_kwgs["output_dim"]//64 + 1)])
        rnn = dict(output_dim=rnn_output_dim, batch_first=True)

        # FC layer hps
        fc_kwgs = model_kwgs["fc_kwargs"]
        fc_layers = trial.suggest_int("fc_n_layers", 1, len(fc_kwgs["hidden_dims"]))
        fc_hidden_dims = [trial.suggest_categorical("fc_n_units_l{}".format(i+1), [x*64 if x != 0 else 1 for x in range(0, fc_kwgs["hidden_dims"][i]//64 + 1)]) for i in range(fc_layers)]
        fc_dropout = trial.suggest_float("fc_dropout", 0.1, fc_kwgs["dropout_rate"])
        fc_batchnorm = trial.suggest_categorical("fc_batchnorm", [True, False])
        fc = dict(output_dim=1, 
                  hidden_dims=fc_hidden_dims, 
                  dropout_rate=fc_dropout,
                  batchnorm=fc_batchnorm)
        starting_lr = trial.suggest_float("starting_lr", 1e-5, model_kwgs["learning_rate"], log=True)
        eugene = dsEUGENE(conv_kwargs=cnn, rnn_kwargs=rnn, fc_kwargs=fc, learning_rate=starting_lr)
    else:
        eugene = dsEUGENE(model_kwgs)        

    # Data params
    data_kwgs = config["data"]
    if config["hyperopt_data"]:
        batch_size = trial.suggest_categorical("batch_size", [x*64 if x != 0 else 1 for x in range(0, data_kwgs["batch_size"][i]//64 + 1)])
        low = trial.suggest_float("low_threshold", 0, data_kwgs["load_kwargs"]["low_thresh"])
        high = trial.suggest_float("high_threshold", data_kwgs["load_kwargs"]["high_thresh"], 1)
        load = dict(target_col=data_kwgs["load_kwargs"]["target_col"], low_thresh=low, high_thresh=high)
        mod = SeqDataModule(seq_file=data_kwgs["seq_file"],
                             batch_size=batch_size,
                             num_workers=data_kwgs["num_workers"],
                             split=data_kwgs["split"],
                             load_kwargs=load)
    else:
        mod = SeqDataModule(**data_kwgs)
        
    logger_kwgs = config["trainer"]["logger"]["init_args"]
    logger = TensorBoardLogger(logger_kwgs["save_dir"], name=logger_kwgs["name"], version="trial_{}".format(trial.number))
    print(logger.version)
    trainer = pl.Trainer(gpus=config["trainer"]["gpus"],
                         max_epochs=config["trainer"]["max_epochs"],
                         logger=logger,
                         callbacks=pl_cli.trainer.callbacks)
    trainer.fit(model=eugene, datamodule=mod)  #, logger=logger)
    return trainer.callback_metrics["hp_metric"].item()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--num_trials", default=50)
        parser.add_argument("--hyperopt_model", default=True)
        parser.add_argument("--hyperopt_data", default=False)
        parser.add_argument("--pruning", action="store_true",
                            help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.")
        
if __name__ == "__main__":
    cli = MyLightningCLI(dsEUGENE, SeqDataModule, run=False)
    print(cli.config["model"])
    print(cli.config["num_trials"])
    
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner() if cli.config["pruning"] else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial=trial, pl_cli=cli), n_trials=cli.config["num_trials"], timeout=1200)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    