import yaml
import os
import sys
PATH = os.path.dirname(os.path.realpath(__file__))


def get_config(filename):
    filename = os.path.join(PATH, filename)
    config = {}

    with open(filename) as fp:
        for key, value in yaml.load(fp.read()).items():
            if "dir_rel" in key:
                config[key.replace("_rel","")] = os.path.join(PATH, value)
            elif "dir_abs" in key:
                config[key.replace("_abs","")] = value
            else:
                config[key] = value
    return config
        
def global_config(filename='global_config.conf'):
    return get_config(filename)
    