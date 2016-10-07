import yaml
import os
import sys
from .. import ROOTPATH


def get_config(filename):
    filename = os.path.join(ROOTPATH, "configs", filename)
    config = {}

    with open(filename) as fp:
        for key, value in yaml.load(fp.read()).items():
            if "_rel" in key:
                config[key.replace("_rel","")] = os.path.join(ROOTPATH, value)
            elif "_abs" in key:
                config[key.replace("_abs","")] = value
            else:
                config[key] = value
    return config
        
def global_config(filename='premade_confs/global.conf'):
    return get_config(filename)
    
    
def compose_configs(data=None, model=None, embedding=None, use_premades=True):
    config = global_config()
    prefix = ''
    format_conf = lambda *args: os.path.join(*args)+".conf"
    if use_premades:
        prefix = 'premade_confs'
    
    if data:
        config.update(get_config(format_conf(prefix, data)))
    if model:
        config.update(get_config(format_conf(prefix, model)))
    if embedding:
        config['embedding_type'] = embedding
    
    if 'saving_prefix' in config and 'embedding_type' in config:
        config['saving_prefix'] = "{}_{}".format(config['saving_prefix'], 
                                                 config['embedding_type'])
    
    return config
    
    