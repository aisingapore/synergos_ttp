#!/usr/bin/env python

####################
# Required Modules #
####################


# Generic/Built-in
import abc
import logging
from typing import Dict

# Libs


# Custom


##################
# Configurations #
##################


########################################
# Abstract Tuner Class - AbstractTuner #
########################################

class AbstractTuner(abc.ABC):

    @abc.abstractmethod
    def tune(self):
        """ Performs federated training using a pre-specified model as a 
            template, across initialised worker nodes, coordinated by a ttp 
            node, across a specified range of hyperparameters, to obtain optimal
            performance
        """
        pass
