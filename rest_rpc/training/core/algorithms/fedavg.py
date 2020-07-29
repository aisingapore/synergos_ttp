#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import copy
import json
import logging
import os
from collections import OrderedDict
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
import torch as th
import tensorflow as tft
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from config import seed_everything
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.early_stopping import EarlyStopping
from rest_rpc.training.core.model import Model
from rest_rpc.training.core.algorithms.base import BaseAlgorithm

##################
# Configurations #
##################


###########################################
# Federated Algorithm Base Class - FedAvg #
###########################################

class FedAvg(BaseAlgorithm):
    """ 
    Contains baseline functionality to all algorithms. Other specific 
    algorithms will inherit all functionality for handling basic federated
    mechanisms. Extensions of this class overrides 5 key methods 
    (i.e. `fit`, `evaluate`, `analyse` and `export`)

    """
    
    def __init__(
        self, 
        crypto_provider: sy.VirtualWorker,
        workers: List[WebsocketClientWorker],
        arguments: Arguments,
        train_loader: sy.FederatedDataLoader,
        eval_loader: sy.FederatedDataLoader,
        test_loader: sy.FederatedDataLoader,
        global_model: Model,
        local_models: Dict[str, Model] = {},
        out_dir: str = '.',
    ):
        # For custom criterion to behave like the vanilla fedavg, the 
        # coefficient mu MUST be 0 (i.e. No FedProx effect) 
        assert self.arguments.mu == 0.0

        super().__init__(
            crypto_provider=crypto_provider,
            workers=workers,
            arguments=arguments,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader,
            global_model=global_model,
            local_models=local_models,
            out_dir=out_dir
        )

    ##################
    # Core functions #
    ##################

    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        raise NotImplementedError

