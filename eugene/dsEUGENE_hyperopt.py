import optuna
from optuna.integration import PyTorchLightningPruningCallback
from MPRADataModule import MPRADataModule
from dsEUGENE import dsEUGENE
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI

# Hyperoptimization objective function
def objective(trial: optuna.trial.Trial, pl_cli):
    config = cli.config
    
    conv_kwgs = config["model"]["conv_kwargs"]
    conv_channels = [4]
    conv_kernels, pool_kernels = [], []
    for i in range(1, len(conv_kwgs["channels"])):
        conv_channels.append(trial.suggest_int("conv_{}".format(i), 1, conv_kwgs["channels"][i], 8))
        conv_kernels.append(trial.suggest_int("conv_kernel_{}".format(i), 1, conv_kwgs["conv_kernels"][i], 8))
        pool_kernels.append(trial.suggest_int("pool_kernel_{}".format(i), 1, conv_kwgs["pool_kernels"][i], 8))   
    cnn=dict(input_len=conv_kwgs["input_len"], channels=conv_channels, conv_kernels=conv_kernels, pool_kernels=pool_kernels)
    
    rnn_kwgs = config["model"]["rnn_kwargs"]
    rnn_output_dim = trial.sugges_int("rnn_output_dim", 1, rnn_kwgs["output_dim"], 16)
    rnn=dict(output_dim=rnn_output_dim, batch_first=True)
    
    fc_kwgs = config["model"]["fc_kwargs"]
    fcn_layers = trial.suggest_int("fcn_n_layers", 1, len(fc_kwgs["hidden_dims"]))
    fcn_dropout = trial.suggest_float("fcn_dropout", 0.1, fc_kwgs["dropout_rate"])
    fcn_output_dims = [trial.suggest_int("fcn_n_units_l{}".format(i), 4, 128, log=True) for i in range(fcn_layers)]
    fc=dict(output_dim=1, hidden_dims=fcn_output_dims, dropout_rate=fcn_dropout)
    eugene = dsEUGENE(conv_kwargs=cnn, rnn_kwargs=rnn, fc_kwargs=fc)
    
    mod = MPRADataModule()
    logger = TensorBoardLogger(tb_name, name="dsEUGENE", version="trial_{}".format(trial.number))
    
    pl_cli.trainer.fit(pl_cli=eugene, datamodule=mod, logger=logger)
    return pl_cli.trainer.callback_metrics["hp_metric"].item()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--num_trials", default=50)
        parser.add_argument("--num_conv_layers", default=3)
        parser.add_argument("--num_channels", default=3)
        
        
if __name__ == "__main__":
    cli = MyLightningCLI(dsEUGENE, MPRADataModule, run=False)
    print(cli.config["model"])
    print(cli.config["num_channels"])
    
    #pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner())
    #study = optuna.create_study(direction="maximize", pruner=pruner)
    #study.optimize(lambda trial: objective(trial=trial, pl_cli=cli), n_trials=5, timeout=600)

    #print("Number of finished trials: {}".format(len(study.trials)))

    #print("Best trial:")
    #trial = study.best_trial

    #print("  Value: {}".format(trial.value))

    #print("  Params: ")
    #for key, value in trial.params.items():
        #print("    {}: {}".format(key, value))
    