#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import importlib
import inspect
import logging
from collections import OrderedDict
from typing import Tuple

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
        self.__SPECIAL_CASES = [
            'RNNBase', 'RNN', 'RNNCell',
            'LSTM', 'LSTMCell',
            'GRU', 'GRUCell'
        ]
        
        self.layers = OrderedDict()

        for layer, params in enumerate(structure):

            # Detect if input layer
            is_input_layer = params['is_input']

            # Detect layer type
            layer_type = params['l_type']

            # Construct layer name (eg. nnl_0_linear)
            layer_name = self.__construct_layer_name(layer, layer_type)

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
    def __construct_layer_name(layer_idx: int, layer_type: str) -> str:
        """ This function was created as a means for formatting the layer name
            to facilitate finding & handling special cases during forward
            propagation

        Args:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        Returns:
            layer name (str)
        """
        return f"nnl_{layer_idx}_{layer_type.lower()}" 


    @staticmethod
    def __parse_layer_name(layer_name: str) -> Tuple[str, str]:
        """ This function was created as a means for reversing the formatting
            done during name creation to facilitate finding & handling special 
            cases during forward propagation

        Args:
            layer name (str)
        Returns:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        """
        _, layer_idx, layer_type = layer_name.split('_')
        return layer_idx, layer_type.capitalize()


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
        for layer_name, a_func in self.layers.items():
            curr_layer = getattr(self, layer_name)

            _, layer_type = self.__parse_layer_name(layer_name)

            # Check if current layer is a recurrent layer
            if layer_type in self.__SPECIAL_CASES:
                x, _ = a_func(curr_layer(x))
            else:
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