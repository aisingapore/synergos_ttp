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



##################################################
# Federated Algorithm Base Class - BaseAlgorithm #
##################################################

class FedSwarm(BaseAlgorithm):
    """ 
    Implements the FedSwarm algorithm.

    Contains baseline functionality to all algorithms. Other specific 
    algorithms will inherit all functionality for handling basic federated
    mechanisms. Extensions of this class overrides 5 key methods 
    (i.e. `fit`, `evaluate`, `analyse`, `export` and `restore`)

    Attributes:
        <Insert your attribute documentations here>
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
        # enter logging statement:  initialize fedswarm algo 
        
        # Insert your own attributes here!


    ############
    # Checkers #
    ############

    # Declare all helper functions that check on the current state of any 
    # attribute in this section


    ###########
    # Setters #
    ###########

    # Declare all helper functions that modify custom attributes in this section


    ###########
    # Helpers #
    ###########

    # Declare all overridden or custom helper functions in this section


    ##################
    # Core functions #
    ##################

    # Override the 5 core functions `fit`, `evaluate`, `analyse`, `export` &
    # restore. Make sure that the class is self-consistent!

    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
        """
        raise NotImplementedError

    
    def evaluate(
        self,
        metas: List[str] = [], 
        workers: List[str] = []
    ) -> Tuple[Dict[str, Dict[str, th.Tensor]], Dict[str, th.Tensor]]:
        """ Using the current instance of the global model, performs inference 
            on pre-specified datasets.

        Args:
            metas (list(str)): Meta tokens indicating which datasets are to be
                evaluated. If empty (default), all meta datasets (i.e. training,
                validation and testing) will be evaluated
            workers (list(str)): Worker IDs of workers whose datasets are to be
                evaluated. If empty (default), evaluate all workers' datasets. 
        Returns:
            Inferences (dict(worker_id, dict(result_type, th.Tensor)))
            losses (dict(str, th.Tensor))
        """
        raise NotImplementedError
    

    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        raise NotImplementedError
    

    def export(self, out_dir: str = None, excluded: List[str] = []) -> dict:
        """ Snapshots the current state of federated cycle and exports all 
            models to file. A dictionary is produced as a rous

            An archive's structure looks like this:
            {
                'global': {
                    'origin': <crypto_provider ID>,
                    'path': <path(s) to exported final global model(s)>,
                    'loss_history': <path(s) to final global loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported globalmodel(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to global exported model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                'local_<idx>': {
                    'origin': <worker ID>,
                    'path': <path(s) to exported final local model(s)>,
                    'loss_history': <path(s) to final local loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                ...
            }

        Args:
            out_dir (str): Path to output directory for export
            excluded (list(str)): Federated attributes to skip when exporting.
                Attribute options are as follows:
                1) 'global': Skips current state of the global model
                2) 'local': Skips current states of all local models
                3) 'loss': Skips current state of global & local loss histories
                4) 'checkpoint': Skips all checkpointed metadata
        Returns:
            Archive (dict)
        """
        raise NotImplementedError


    def restore( 
        self, 
        archive: dict, 
        version: Tuple[str, str] = None
    ):
        """ Restores model states from a previously archived training run. If 
            version is not specified, then restore the final state of the grid.
            If version is specified, restore the state of all models conforming
            to that version's snapshot.

            An archive's structure looks like this:
            {
                'global': {
                    'origin': <crypto_provider ID>,
                    'path': <path(s) to exported final global model(s)>,
                    'loss_history': <path(s) to final global loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported globalmodel(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to global exported model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                'local_<idx>': {
                    'origin': <worker ID>,
                    'path': <path(s) to exported final local model(s)>,
                    'loss_history': <path(s) to final local loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                ...
            }

        Args:
            archive (dict): Dictionary containing versioned histories of 
                exported filepaths corresponding to the state of models within a
                training cycle
            version (tuple(str)): A tuple where the first index indicates the
                round index and the second the epoch index 
                (i.e. (round_<r_idx>, epoch_<e_idx>))
        """
        raise NotImplementedError
