#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import copy
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Libs
import numpy as np
import pandas as pd
import syft as sy
import torch as th
import tensorflow as tft
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, tnrange, tqdm_notebook
from tqdm.notebook import trange

# Custom
from .arguments import Arguments
from .early_stopping import EarlyStopping

##################
# Configurations #
##################


###############################################
# Abstract Training Class - FederatedLearning #
###############################################

class FederatedLearning:
    """
    The main class that coodinates federated training across a PySyft-driven grid.
    
    Args:
        arguments (Arguments): Arguments to be passed into each FL function
        crypto_provider (VirtualWorker): Trusted Third Party coordinating FL
        workers (WebsocketClientWorker): All particiating WS-CONNECTED workers
        model (Model): Model to be trained federatedly

    Attributes:
        arguments (Arguments): Arguments to be passed into each FL function
        crypto_provider (VirtualWorker): Trusted Third Party coordinating FL
        workers (list(WebsocketClientWorker)): All particiating WS-CONNECTED workers
        grid (sy.PrivateGridNetwork): A grid to facilitate dataset searches
        _aliases (dict): ID-to-worker mappings for ease of reference
        
        train_loader (sy.FederatedLoader): Training data in configured batches
        test_loader (sy.FederatedLoader): Testing data in configured batches
        
        global_model (Model): Federatedly-trained Global model
    """
    def __init__(self, arguments, crypto_provider, workers, model):
        
        # Network attributes
        self.arguments = arguments
        self.crypto_provider = crypto_provider
        self.workers = workers
        self.grid = sy.PrivateGridNetwork(crypto_provider, *workers)
        self._aliases = {w.id: w for w in self.grid.workers}
        
        # Data attributes
        self.train_loader = None
        self.test_loader = None
        
        # Model attributes
        self.global_model = model
        self.loss_history = {'local': {}, 'global': {}}

    ############
    # Checkers #
    ############

    def is_data_loaded(self):
        """ Checks if data has already been loaded 
        
        Returns:
            loaded state (bool)
        """
        return (self.train_loader is not None and self.test_loader is not None)

    ###########
    # Helpers #
    ###########

    def secret_share(self, tensor):
        """ Transform to fixed precision and secret share a tensor 
        
        Args:
            tensor (PointerTensor): Pointer to be shared
        Returns:
            MPC-shared pointer tensor (PointerTensor)
        """
        return (
            tensor
            .fix_precision(precision_fractional=self.arguments.precision_fractional)
            .share(
                *self.workers, 
                crypto_provider=self.crypto_provider, 
                requires_grad=True
            )
        )
    
    
    def setup_FL_env(self, is_shared=False):
        """ Sets up a basic federated learning environment using virtual workers,
            with a allocated arbiter (i.e. TTP) to faciliate in model development
            & utilisation, and deploys datasets to their respective workers
            
        Args:
            is_shared (bool): Toggles if SMPC encryption protocols are active
        Returns:
            training_datasets  (dict(sy.BaseDataset))
            testing_datasets  (sy.BaseDataset)
        """
        
        def convert_to_datasets(*tags):
            """ Takes in tags to query across all workers, and groups X & y 
                pointer tensors into datasets
            
            Args:
                *tags (str): Tags to query on
            Returns:
                datasets (dict(WebsocketClientWorker, sy.BaseDataset))
            """
            # Retrieve Pointer Tensors to remote datasets
            pointers = self.grid.search(*tags)
        
            datasets = {}
            for worker_id, data in pointers.items():

                curr_worker = self._aliases[worker_id]
                data_ptr = sy.BaseDataset(*data)
                datasets[self._aliases[worker_id]] = data_ptr

            return datasets

        # Ensure that TTP does not have residual tensors
        self.crypto_provider.clear_objects()
        assert len(self.crypto_provider._objects) == 0
        
        # Retrieve Pointer Tensors to remote datasets
        training_datasets = convert_to_datasets("#train")
        testing_datasets = convert_to_datasets("#evaluate")
        
        return training_datasets, testing_datasets
    
    
    def convert_to_FL_batches(self, training_datasets, testing_datasets): 
        """ Supplementary function to convert initialised datasets into their
            SGD compatible dataloaders in the context of PySyft's federated learning
            (NOTE: This is based on the assumption that querying database size does
                   not break FL abstraction (i.e. not willing to share quantity))
        Args:
            training_datasets (dict(sy.BaseDataset)): Distributed datasets for training
            testing_datasets  (sy.BaseDataset): Distributed dataset for verifying performance
        Returns:
            train_loaders (sy.FederatedDataLoader)
            test_loader   (sy.FederatedDataLoader)
        """
    
        def construct_FL_loader(dataset, **kwargs):
            """ Cast paired data & labels into configured tensor dataloaders
            Args:
                dataset (list(sy.BaseDataset)): A tuple of X features & y labels
                kwargs: Additional parameters to configure PyTorch's Dataloader
            Returns:
                Configured dataloader (th.utils.data.DataLoader)
            """
            federated_dataset = sy.FederatedDataset(dataset)

            federated_data_loader = sy.FederatedDataLoader(
                federated_dataset, 
                batch_size=(
                    self.arguments.batch_size 
                    if self.arguments.batch_size 
                    else len(federated_dataset)
                ), 
                shuffle=True,
                iter_per_worker=True, # for subsequent parallelization
                **kwargs
            )

            return federated_data_loader

        # Load training datasets into a configured federated dataloader
        train_loader = construct_FL_loader(training_datasets.values())

        # Load testing datasets into a configured federated dataloader
        test_loader = construct_FL_loader(testing_datasets.values())

        return train_loader, test_loader
 

    def perform_FL_training(self, datasets, is_shared=False):
        """ Performs a remote federated learning cycle leveraging PySyft.

        Args:
            datasets (sy.FederatedDataLoader): Distributed training datasets
        Returns:
            trained global model (nn.Module)
        """
        
        class SurrogateCriterion(self.arguments.criterion):
            """ A wrapper class to augment a specified PyTorch criterion to 
                suppport FedProx 
            
            Args:
                mu (float): Regularisation term for gamma-inexact minimizer
                **kwargs: Keyword arguments to pass to parent criterion
                
            Attributes:
                mu (float): Regularisation term for gamma-inexact minimizer
            """
            def __init__(self, mu, l1_lambda, l2_lambda, **kwargs):
                super(SurrogateCriterion, self).__init__(**kwargs)
                self.__temp = [] # tracks minibatches
                self._cache = [] # tracks epochs
                self.mu = mu
                self.l1_lambda = l1_lambda
                self.l2_lambda = l2_lambda

            def forward(self, outputs, labels, w, wt):
                # Calculate normal criterion loss
                loss = super().forward(outputs, labels)
            
                # Calculate regularisation terms
                fedprox_reg_terms = []
                l1_reg_terms = []
                l2_reg_terms = []
                for layer, layer_w in w.items():
                    
                    # Extract corresponding global layer weights
                    layer_wt = wt[layer]
                    
                    # Calculate FedProx regularisation
                    norm_diff = th.norm(layer_w - layer_wt)
                    fp_reg_term = self.mu * 0.5 * (norm_diff**2)
                    fedprox_reg_terms.append(fp_reg_term)
                    
                    # Calculate L1 regularisation
                    l1_norm = th.norm(layer_w, p=1)
                    l1_reg_term = self.l1_lambda * l1_norm
                    l1_reg_terms.append(l1_reg_term)
                    
                    # Calculate L2 regularisation
                    l2_norm = th.norm(layer_w, p=2)
                    l2_reg_term = self.l2_lambda * 0.5 * (l2_norm)**2
                    l2_reg_terms.append(l2_reg_term)
                
                # Retrieve worker involved
                assert outputs.location is labels.location
                worker = labels.location
                
                # Summing up from a list instead of in-place changes 
                # prevents the breaking of the autograd's computation graph
                fedprox_loss = th.stack(
                    fedprox_reg_terms
                ).sum().requires_grad_().send(worker)
                
                l1_loss = th.stack(
                    l1_reg_terms
                ).sum().requires_grad_().send(worker)
                
                l2_loss = th.stack(
                    l2_reg_terms
                ).sum().requires_grad_().send(worker)
                        
                # Add up all losses involved
                surrogate_loss = loss + fedprox_loss + l1_loss + l2_loss
                
                # Store result in cache
                self.__temp.append(surrogate_loss.clone().detach().get())
                
                return surrogate_loss
            
            def log(self):
                """ Computes mean loss across all current runs & caches the result """
                avg_loss = th.mean(th.stack(self.__temp))
                self._cache.append(avg_loss)
                self.__temp.clear()
                return avg_loss
            
            def reset(self):
                self.__temp = []
                self._cache = []
                return self
            
            
        def perform_parallel_training(datasets, models, optimizers, schedulers, criterions, stoppers, epochs):
            """ Parallelizes training across each distributed dataset (i.e. simulated worker)
                Parallelization here refers to the training of all distributed models per
                epoch.
                NOTE: Current approach does not have early stopping implemented

            Args:
                datasets   (dict(DataLoader)): Distributed training datasets
                models     (dict(nn.Module)): Local models (after distribution)
                optimizers (dict(th.optim)): Local optimizers (after distribution)
                schedulers (dict(lr_scheduler)): Local LR schedulers (after distribution)
                criterions (dict(th.nn)): Custom local objective function (after distribution)
                stoppers   (dict(EarlyStopping)): Local early stopping drivers
                epochs (int): No. of epochs to train each local model
            Returns:
                trained local models
            """ 
            # Global model weights from previous round for subsequent FedProx Comparison
            PREV_ROUND_GLOBAL_MODELS = {w:copy.deepcopy(m) for w,m in models.items()}
            
            # Tracks which workers have reach an optimal/stagnated model
            WORKERS_STOPPED = []
            
            for e in range(epochs):

                for batch_idx, batch in enumerate(datasets):

                    for worker, (data, labels) in batch.items():
                        
                        # Check if worker has been stopped
                        if worker in WORKERS_STOPPED:
                            break
                        
                        curr_global_model = PREV_ROUND_GLOBAL_MODELS[worker]
                        curr_local_model = models[worker]
                        curr_optimizer = optimizers[worker]
                        curr_criterion = criterions[worker]

                        # Zero gradients to prevent accumulation  
                        curr_local_model.train()
                        curr_optimizer.zero_grad()

                        # Forward Propagation
                        predictions = curr_local_model(data.float())

                        if self.arguments.is_condensed:
                            loss = curr_criterion(
                                predictions, 
                                labels.float(),
                                w=curr_local_model.state_dict(),
                                wt=curr_global_model.state_dict()
                            )
                        else:
                            # To handle multinomial context
                            loss = curr_criterion(
                                predictions, 
                                th.max(labels, 1)[1],
                                w=curr_local_model.state_dict(),
                                wt=curr_global_model.state_dict()
                            )

                        # Backward propagation
                        loss.backward()
                        curr_optimizer.step()

                        # Update models, optimisers & losses
                        models[worker] = curr_local_model
                        optimizers[worker] = curr_optimizer
                        criterions[worker] = curr_criterion

                        assert (models[worker] == curr_local_model and 
                                optimizers[worker] == curr_optimizer and 
                                criterions[worker] == curr_criterion)
                        
                # Check if early stopping is possible for each worker
                for worker in self.workers:
                    
                    # Retrieve final loss computed for this epoch
                    trained_criterion = criterions[worker]
                    final_batch_loss = trained_criterion.log()

                    # Retrieve model mutated in this epoch
                    trained_model = models[worker]

                    # Retrieve respective stopper
                    curr_stopper = stoppers[worker]

                    curr_stopper(final_batch_loss, trained_model)

                    # If model is deemed to have stagnated, stop training
                    if curr_stopper.early_stop:
                        WORKERS_STOPPED.append(worker)
                        
                    # else, perform learning rate decay
                    else:
                        curr_scheduler = schedulers[worker]
                        curr_scheduler.step()
                        
                    # Update criterions, stoppers & schedulers
                    criterions[worker] = trained_criterion
                    stoppers[worker] = curr_stopper
                    schedulers[worker] = curr_scheduler
                    
                    assert (criterions[worker] == trained_criterion and
                            stoppers[worker] == curr_stopper and
                            schedulers[worker] == curr_scheduler)

            # Retrieve all models from their respective workers
            trained_models = {w: m.get() for w,m in models.items()}

            return trained_models, optimizers, schedulers, criterions, stoppers
        
        
        def calculate_global_params(global_model, models, datasets):
            """ Aggregates weights from trained locally trained models after a round.

            Args:
                global_model   (nn.Module): Global model to be trained federatedly
                models   (dict(nn.Module)): Simulated local models (after distribution)
                datasets (dict(th.utils.data.DataLoader)): Distributed training datasets
            Returns:
                Aggregated parameters (OrderedDict)
            """
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
                    ) for w in self.workers
                ]

                layer_shape = tuple(global_model.state_dict()[p_type].shape)

                #aggregated_params[p_type] = th.add(*param_states).view(*layer_shape)
                aggregated_params[p_type] = th.stack(
                    param_states,
                    dim=0
                ).sum(dim=0).view(*layer_shape)

            return aggregated_params
        
        th.manual_seed(self.arguments.seed)

        ###########################
        # Implementation Footnote #
        ###########################

        # However, due to certain PySyft nuances (refer to Part 4, section 1: Frame of
        # Reference) there is a need to choose a conceptual representation of the overall 
        # architecture. Here, the node agnostic variant is implemented.
        # Model is stored in the server -> Client (i.e. 'Me') does not interact with it
        
        # Note: If MPC is requested, global model itself cannot be shared, only its
        # copies are shared. This is due to restrictions in PointerTensor mechanics.
        
        global_stopper = EarlyStopping(**self.arguments.early_stopping_params)
        
        rounds = 0
        pbar = tqdm(total=self.arguments.rounds, desc='Rounds', leave=True)
        while rounds < self.arguments.rounds:

            # Generate K copies of template model, representing local models for each worker,
            # and send them to their designated worker
            # Note: This step is crucial because it is able prevent pointer mutation, which
            #       comes as a result of copying pointers (refer to Part 4, section X), 
            #       specifically if the global pointer was copied directly.
            
            local_models = {w: copy.deepcopy(self.global_model).send(w) for w in self.workers}

            optimizers = {
                w: self.arguments.optimizer(
                    params=model.parameters(), 
                    **self.arguments.optimizer_params
                ) for w, model in local_models.items()
            }

            schedulers = {
                w: CyclicLR(optimizer, **self.arguments.lr_decay_params) 
                if self.arguments.use_CLR
                else LambdaLR(optimizer, **self.arguments.lr_decay_params)
                for w, optimizer in optimizers.items()
            }

            criterions = {
                w: SurrogateCriterion(
                    **self.arguments.criterion_params
                ) for w,m in local_models.items()
            }
            
            stoppers = {
                w: EarlyStopping(
                    **self.arguments.early_stopping_params
                ) for w,m in local_models.items()
            }
            
            trained_models, _, _, _, _= perform_parallel_training(
                datasets, 
                local_models, 
                optimizers, 
                schedulers,
                criterions, 
                stoppers,
                self.arguments.epochs
            )
            
            aggregated_params = calculate_global_params(
                self.global_model, 
                trained_models, 
                datasets
            )
            
            # Update weights with aggregated parameters 
            self.global_model.load_state_dict(aggregated_params)
            
            # Check if early stopping is possible for global model
            final_local_losses = [c._cache[-1] for c in criterions.values()]
            global_loss = th.mean(th.stack(final_local_losses))

            # Store losses for analysis
            self.loss_history['local'].update({rounds: final_local_losses})
            self.loss_history['global'].update({rounds: global_loss})

            global_stopper(global_loss, self.global_model)
            
            # If global model is deemed to have stagnated, stop training
            if global_stopper.early_stop:
                break
            
            rounds += 1
            pbar.update(1)
        
        pbar.close()

        return self.global_model

    ##################
    # Core functions #
    ##################
    
    def load(self):
        """ Load all remote datasets into Federated dataloaders for batch
            operations if data has not already been loaded. Note that loading 
            is only done once, since searching for datasets within a grid will
            cause query results to accumulate exponentially on TTP. Hence, this
            function will only load datasets ONLY if it has NOT already been
            loaded.
            
        Returns:
            train_loader (sy.FederatedLoader): Training data in configured batches
            test_loader (sy.FederatedLoader): Testing data in configured batches
        """
        if not self.is_data_loaded():
            
            # Extract data pointers from workers
            training_datasets, testing_datasets = self.setup_FL_env()

            # Generate federated minibatches via loaders 
            train_loader, test_loader = self.convert_to_FL_batches(
                training_datasets, 
                testing_datasets
            )

            # Store federated data loaders for subsequent use
            self.train_loader = train_loader
            self.test_loader = test_loader

        return self.train_loader, self.test_loader

        
    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
            
        Returns:
            Trained global model (Model)
        """
        if not self.is_data_loaded():
            raise RuntimeError("Grid data has not been aggregated! Call '.load()' first & try again.")
            
        # Train global model on aggregated training data
        self.perform_FL_training(self.train_loader, is_shared=True)
        
        return self.global_model


    def reset(self):
        """ Original intention was to make this class reusable, but it seems 
            like remote modification of remote datasets is not allowed/does not
            work. Only in the local machine itself do the following functions 
            perform as documented:
            
            1) rm_obj(self, remote_key:Union[str, int])
            2) force_rm_obj(self, remote_key:Union[str, int])
            3) de_register_obj(self, obj:object, _recurse_torch_objs:bool=True)
            
            In hindside, this makes sense since the system becomes more stable. 
            Clients using the grid cannot modify the original datasets in remote 
            workers, mitigating possibly malicious intent. However, this also 
            means that residual tensors will pile up after each round of FL, 
            which will consume more resources. TTP can clear its own objects, 
            but how to inform remote workers to refresh their
            WebsocketServerWorkers?
            
            A possible solution is to leverage on the external Flask 
            interactions.
        """
        raise NotImplementedError
    
    
    def export(self, out_dir):
        """ Exports the global model state dictionary to file
        
        Returns:
            Path-to-file (str)
        """
        model_out_path = os.path.join(out_dir, "trained_global_model.pt")
        th.save(self.global_model.state_dict(), model_out_path)

        loss_out_path = os.path.join(out_dir, "loss_history.json")
        with open(loss_out_path, 'w') as lp:
            json.dump(self.loss_history, lp)

        return model_out_path, loss_out_path
    