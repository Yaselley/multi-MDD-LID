# main.py
#%%
from modeling_MDD.run_training import train_model
import json
#%%

# Read configuration from a JSON file
with open('config.json') as config_file:
    config = json.load(config_file)

# Train the model using the provided configuration
train_model(config)
