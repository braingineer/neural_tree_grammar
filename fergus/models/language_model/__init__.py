from __future__ import absolute_import

from .model import LanguageModel

def from_config(config):
    return LanguageModel.from_config(config)