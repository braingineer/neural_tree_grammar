import yaml

class Blob(object):
    def __init__(self, conf=None, conf_file=None):
        if conf is None and conf_file is not None:
            with open(conf_file) as fp:
                conf = yaml.load(fp)
        elif conf is None:
            raise Exception("Incorrect usage of config container")

        self.__dict__.update(conf.items())


config = Blob("pipeline_settings.conf")



