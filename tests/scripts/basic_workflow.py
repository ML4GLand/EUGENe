import eugene as eu

# Configure EUGENe for a test spin
eu.settings.dataset_dir = "eugene_data"
eu.settings.output_dir = "eugene_ouput"
eu.settings.logging_dir = "eugene_logs"
eu.settings.config_dir = "eugene_configs"

# Extract the data
sdata = eu.datasets.random1000()

# Preprocess sequences for training
eu.pp.prepare_seqs_sdata(sdata)

# Instantiate the model
model = eu.DeepSEA(input_len=100, output_dim=1)
    
# Initialize the model
eu.models.init_weights(model)

# Train the model
eu.train.fit(
    model=model, 
    sdata=sdata, 
    epochs=100,
    ...
)
        
# Evaluate the model
eu.evaluate.train_val_predictions(
    model=model,
    sdata=sdata,
    target_vars="activity_0",
    ...
)

# Intepret the model
eu.interpret.feature_attribution_sdata(
    model=model,
    sdata=sdata,
    method="DeepLift",
    ...
)
    

