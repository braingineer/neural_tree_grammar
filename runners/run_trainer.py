"""An implemenation of FERGUS's supertagger

Usage:
    run_trainer.py (fergusr|fergusn) (convolutional|token|shallowconv|minimaltoken)
    run_trainer.py fergusn 
    run_trainer.py (-h | --help)

    fergusr                   train the fergus-r model
    fergusn                   train the fergus-n model
    
    
Notes: 
as per Bangalore & Rambow, 2000 
"Exploiting a Probabilistic Hierarchical Model for Generation"
"""
import os
import sys
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, '..'))
import fergus

from ikelos.data import Vocabulary
from fergus.models import fergus_recurrent, fergus_neuralized
from fergus.configs import compose_configs
from os.path import join
from docopt import docopt
import yaml
import json
import traceback

import theano
from theano.compile.nanguardmode import NanGuardMode


if __name__ == "__main__":

    args = docopt(__doc__, version='FERGUS trainer; October 2016')
    try:
        if args['convolutional']:
            embed_type = 'convolutional'
        elif args['token']:
            embed_type = 'token'
        elif args['minimaltoken']:
            embed_type = 'minimaltoken'
        elif args['shallowconv']:
            embed_type = 'shallowconv'
        else:
            raise Exception("bad embedding argument")
        model_type = "fergusn" if args['fergusn'] else 'fergusr'
        model_factory = {'fergusn': fergus_neuralized, 
                         'fergusr': fergus_recurrent}[model_type]
        config = compose_configs(data='train', model=model_type, embedding=embed_type)
    
        model = model_factory.get_model(config)
        model.train()

    except KeyboardInterrupt as e:
        print("Exceptional Keyboard interruption... saving model now")
        model.model.save("model_safetynet.h5")
    except Exception as e:
        print("Unknown exception: {}".format(e))
        print("Some more info: {}".format(sys.exc_info()))
        print(traceback.format_exc())
          
        # elif args['debug']:
        #     kwargs = {"mode":NanGuardMode(nan_is_error=True, 
        #                                   inf_is_error=True, 
        #                                   big_is_error=True)}
        #     #stage(igor)
        #     #model = TrainingModel(igor)
        #     #model.make(kwargs)
        #     #model.debug()
        #     #model.train()

        #     model = TrainingModel.from_yaml(args['<config_file>'], kwargs)
        #     model.train()

        # elif args['profile']:
        #     kwargs = {"profile": True}
        #     stage(igor)
        #     model = TrainingModel(igor)
        #     model.make(kwargs)
        #     model.profile(num_iterations=1)

        # elif args['visualize']:
        #     stage(igor)
        #     model= TrainingModel(igor)
        #     model.make()
        #     model.plot()
