'''
organization of algorithm:
    - tagger.Tagger serves as the entry point for the algorithm
    - tagger.Tagger runs the GPU model
    - tagger.Tagger can either have multiple methods or be subclasses
    - tagger_node.* handle the specific node propertis for Fergus-N and Fergus-R
    - astar_core.run handles the astar algorithm proper
    - astar_state.* should implement specific state information for Fergus-N and Fergus-R
    -
'''
from __future__ import absolute_import
from .tagger import Tagger

