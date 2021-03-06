from __future__ import absolute_import
import os
from . import model

def get_model(init_config):
    return model.FergusNModel.from_config(init_config)
# def TrainingModel():
#     print("Retrieving training model")
#     config = global_config()
#     config_file = config['fergus_n_train_config']
#     config_file = os.path.join(config['config_dir'], config_file)
#     return model.FergusNModel.from_yaml(config_file)
    
# def DevelopmentModel():
#     print("Retrieving development model")
#     config = global_config()
#     config_file = config['fergus_n_dev_config']
#     config_file = os.path.join(config['config_dir'], config_file)
#     return model.FergusNModel.from_yaml(config_file)

# def TestingModel():
#     print("Retrieving testing model")
#     config = global_config()
#     config_file = config['fergus_n_test_config']
#     config_file = os.path.join(config['config_dir'], config_file)
#     return model.FergusNModel.from_yaml(config_file)

# def globally_set_model():
#     config = global_config()
#     return {'dev': DevelopmentModel,
#             'test': TestingModel,
#             'train': TrainingModel}[config['tag_mode']]()