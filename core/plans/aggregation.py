#!/usr/bin/env python

####################
# Required Modules #
####################

# Generics
import logging
from collections import OrderedDict

# Libs
import syft as sy
import torch as th

##################
# Configurations #
##################

hook = sy.TorchHook(th)

#########################################
# Weight Aggregation Class - Aggregator #
#########################################

class Aggregator(sy.Plan):
    """
    The Aggregator class applies a selected weight aggregation algorithm upon
    all locally trained models in a federated network, so as to obtained an 
    updated parameter set for the global model per round.

    Args:
        data   (pd.DataFrame): Data to be processed
        schema         (dict): Datatypes of features found within dataset
        seed            (int): Seed to fix random state for testing consistency
        *args:
        **kwargs:
        
    Attributes:
        __ (CatBoost): CatBoost Classifier to impute categories

        data   (pd.DataFrame): Loaded data to be processed
    """
    def __init__(self):
        pass

    ###########
    # Getters #
    ###########


    ###########
    # Setters #
    ###########


    ###########
    # Helpers #
    ###########

    def __perform_basic_fedavg(self):
        pass

    def __perform_custom_fedavg(self):
        pass

    ##################
    # Core functions #
    ##################

    def aggregate(self):
        """ Aggregates weights from trained locally trained models after a round.
        
        Args:
            global_model   (nn.Module): Global model to be trained federatedly
            models   (dict(nn.Module)): Simulated local models (after distribution)
            datasets (dict(th.utils.data.DataLoader)): Distributed training datasets
        Returns:
            Aggregated parameters (OrderedDict)
        """
        return self.__perform_basic_fedavg()
        
"""
def calculate_global_params(global_model, models, datasets):

    param_types = global_model.state_dict().keys()
    model_states = {w: m.state_dict() for w,m in models.items()}

    # Find size of all distributed datasets for calculating scaling factor
    obs_counts = {}
    for batch_idx, batch in enumerate(datasets):
        for worker, (data, labels) in batch.items():
            curr_count = len(data)
            if worker in obs_counts.keys():
                obs_counts[worker] += curr_count
            else:
                obs_counts[worker] = curr_count
    
    # Calculate scaling factors for each worker
    scale_coeffs = {w: c/sum(obs_counts.values()) for w,c in obs_counts.items()}

    # PyTorch models can only swap weights of the same structure
    # Hence, aggregate weights while maintaining original layering structure
    aggregated_params = OrderedDict()
    for p_type in param_types:
 
        param_states = [
            th.mul(
                model_states[w][p_type],
                scale_coeffs[w]
            ).get().get() for w in workers
        ]

        param_states = [
            th.mul(
                ms[p_type], 
                scale_coeffs[w]
            ).get().get() for w, ms in model_states.items()
        ]
        
        layer_shape = tuple(global_model.state_dict()[p_type].shape)
        
        aggregated_params[p_type] = th.add(*param_states).view(*layer_shape)
        #print(p_type, th.add(*param_states).view(*layer_shape).shape)

    return aggregated_params
"""

#########
# Tests #
#########

if __name__ == "__main__":
    aggregator  =  Aggregator()
    print(aggregator.aggregate())
