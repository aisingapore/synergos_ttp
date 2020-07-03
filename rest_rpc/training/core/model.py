#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import importlib
import inspect
import logging
from collections import OrderedDict

# Libs
import torch as th
from torch import nn

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

MODULE_OF_LAYERS = "torch.nn"
MODULE_OF_ACTIVATIONS = "torch.nn.functional"

###################################
# Model Abstraction Class - Model #
###################################

class Model(nn.Module):
    """
    The Model class serves to automate the building of structured deep neural
    nets, given specific layer configurations. Being a parent class of sy.Plan,
    this makes it more efficient to deploy in terms of communication costs.

    Args:
        owner (VirtualWorker/WebsocketClientWorker): Handler of this model
        structure (OrderedDict): Configurations used to build the achitecture of the NN
        is_condensed (bool): Toggles Binary or Multiclass prediction

    Attributes:
        is_condensed  (bool): Toggles Binary or Multiclass prediction
        layers (OrderedDict): Maps specific layers to their respective activations
        + <Specific layer configuration dynamically defined>
    """
    def __init__(self, structure):
        super(Model, self).__init__()
        
        self.layers = OrderedDict()

        for layer, params in enumerate(structure):

            # Construct layer name
            layer_name = f"nnl_{layer}" # neural network layer

            # Detect if input layer
            is_input_layer = params['is_input']

            # Detect layer type
            layer_type = params['l_type']

            # Extract layer structure and initialise layer
            layer_structure = params['structure']
            setattr(
                self, 
                layer_name,
                self.__parse_layer_type(layer_type)(**layer_structure)
            )

            # Detect activation function & store it for use in .forward()
            # Note: In more complex models, other layer types will be declared,
            #       ones that do not require activation intermediates (eg. 
            #       batch normalisation). Hence, skip activation if undeclared
            layer_activation = params['activation']
            if layer_activation:
                self.layers[layer_name] = self.__parse_activation_type(
                    layer_activation
                )

    ###########
    # Helpers #
    ###########

    @staticmethod
    def __parse_layer_type(layer_type):
        """ Detects layer type of a specified layer from configuration

        Args:
            layer_type (str): Layer type to initialise
        Returns:
            Layer definition (Function)
        """
        try:
            layer_modules = importlib.import_module(MODULE_OF_LAYERS)
            layer = getattr(layer_modules, layer_type)
            return layer

        except AttributeError:
            logging.error(f"Specified layer type '{layer_type}' is not supported!")


    @staticmethod
    def __parse_activation_type(activation_type):
        """ Detects activation function specified from configuration

        Args:
            activation_type (str): Activation function to use
        Returns:
            Activation definition (Function)
        """
        try:
            activation_modules = importlib.import_module(MODULE_OF_ACTIVATIONS)
            activation = getattr(activation_modules, activation_type)
            return activation

        except AttributeError:
            logging.error(f"Specified activation type '{activation_type}' is not supported!")

    ##################
    # Core Functions #
    ##################

    def forward(self, x):
        
        # Apply the appropiate activation functions
        for layer, a_func in self.layers.items():
            curr_layer = getattr(self, layer)
            x = a_func(curr_layer(x))

        return x


#########
# Tests #
#########

if __name__ == "__main__":

    from pprint import pprint
    from config import model_params

    for model_name, model_structure in model_params.items():
        
        model = Model(model_structure)
        pprint(model.__dict__)
        pprint(model.state_dict())