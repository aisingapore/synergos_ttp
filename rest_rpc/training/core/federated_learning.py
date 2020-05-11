#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import asyncio
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
        local_models (dict(str,Models)): Most recent cache of local models
        loss_history (dict): Local & global losses tracked throughout FL training
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
        self.eval_loader = None
        self.test_loader = None
        
        # Model attributes
        self.global_model = model
        self.local_models = {}
        self.loss_history = {'local': {}, 'global': {}}
        self.cache = []

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
            train_datasets (dict(sy.BaseDataset))
            eval_datasets  (dict(sy.BaseDataset))
            test_datasets  (dict(sy.BaseDataset))
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
                
                # Ensure that X & y pointers are arranged sequentially
                sorted_data = sorted(data, key=lambda x: sorted(list(x.tags)))

                curr_worker = self._aliases[worker_id]
                data_ptr = sy.BaseDataset(*sorted_data)
                #datasets[self._aliases[worker_id]] = data_ptr
                datasets[curr_worker] = data_ptr

            return datasets

        # Ensure that TTP does not have residual tensors
        self.crypto_provider.clear_objects()
        assert len(self.crypto_provider._objects) == 0
        
        # Retrieve Pointer Tensors to remote datasets
        train_datasets = convert_to_datasets("#train")
        eval_datasets = convert_to_datasets("#evaluate")
        test_datasets = convert_to_datasets("#predict")
        
        return train_datasets, eval_datasets, test_datasets
    
    
    def convert_to_FL_batches(self, train_datasets, eval_datasets, test_datasets): 
        """ Supplementary function to convert initialised datasets into their
            SGD compatible dataloaders in the context of PySyft's federated learning
            (NOTE: This is based on the assumption that querying database size does
                   not break FL abstraction (i.e. not willing to share quantity))
        Args:
            train_datasets (dict(sy.BaseDataset)): Distributed datasets for training
            eval_datasets  (dict(sy.BaseDataset)): Distributed dataset for verifying performance
            test_datasets  (dict(sy.BaseDataset)): Distributed dataset for to be tested on
        Returns:
            train_loader (sy.FederatedDataLoader)
            eval_loader  (sy.FederatedDataLoader)
            test_loader  (sy.FederatedDataLoader)
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
                iter_per_worker=False,#True, # for subsequent parallelization
                **kwargs
            )

            return federated_data_loader
        """
        # Load training datasets into a configured federated dataloader
        train_loader = construct_FL_loader(train_datasets.values())

        # Load validation datasets into a configured federated dataloader
        eval_loader = construct_FL_loader(eval_datasets.values())

        # Load testing datasets into a configured federated dataloader
        test_loader = construct_FL_loader(test_datasets.values())
        
        return train_loader, eval_loader, test_loader
        """
        train_loaders = {
            worker: construct_FL_loader([dataset]) 
            for worker, dataset in train_datasets.items()
        }

        eval_loaders = {
            worker: construct_FL_loader([dataset]) 
            for worker, dataset in eval_datasets.items()
        }

        test_loaders = {
            worker: construct_FL_loader([dataset]) 
            for worker, dataset in test_datasets.items()
        }

        return train_loaders, eval_loaders, test_loaders
        

    def distribute_federated_datasets(self, train_loader, test_loader):
        """ Adds batches of datasets constructed from deep object retrieval via
            grid search in to workers' dataset caches for subsequent 
            asynchronous training.
            
            Fine details:
            There are 2 main data cataloguing mechanisms at play in PySyft, Tags
            and Datasets. Adding adding a tagged tensor integrates the tensor
            deep into the worker's object registry. On the other hand, adding a
            set of tensors via `worker.add_dataset()` is a shallow operation in
            that the Dataset object is not integrated into the worker's object
            registry as its own dataset, but is merely cached in a dictionary
            attribute for subsequent retrieval/operation.

        Args: 
            train_loader (sy.FederatedDataLoader): Training batched datasets
            eval_loader  (sy.FederatedDataLoader): Evaluation batched datasets
            test_loader  (sy.FederatedDataLoader): Testing batched datasets
        Returns:

        """

        def distribute_batches(data_loader, workers, meta):
            for batch_idx, batch in enumerate(data_loader):
                pass
        pass


    async def fit_model_on_worker(self, worker, local_model, global_model, criterion):
        """Send the model to the worker and fit the model on the worker's training data.
        Args:
            worker: Remote location, where the model shall be trained.
            traced_model: Model which shall be trained.
            batch_size: Batch size of each training step.
            curr_round: Index of the current training round (for logging purposes).
            max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
            lr: Learning rate of each training step.
        Returns:
            A tuple containing:
                * worker_id: Union[int, str], id of the worker.
                * improved model: torch.jit.ScriptModule, model after training at the worker.
                * loss: Loss on last training batch, torch.tensor.
        """

        def reparsed_criterion(target, pred):
            """ Reparse custom surrogate criterion to accept `target` &
                `pred` as keyword arguments for `_fit` in 
                `WebsocketServerWorker`, when `async_fit` is called
            """
            if self.arguments.is_condensed:
                print("Is condensed?", self.arguments.is_condensed)
                return criterion(
                    outputs=pred, 
                    labels=target.float(),
                    w=local_model.state_dict(),
                    wt=global_model.state_dict()
                )
            else:
                # To handle multinomial context
                return criterion(
                    outputs=pred, 
                    labels=th.max(target, 1)[1],
                    w=local_model.state_dict(),
                    wt=global_model.state_dict()
                )


        # Serialise model into TorchScript object
        traced_model = th.jit.script(local_model)

        # Serialise criterion into TorchScript object
        traced_criterion = th.jit.script(reparsed_criterion)

        train_config = sy.TrainConfig(
            model=traced_model,
            loss_fn=traced_criterion,
            batch_size=self.arguments.batch_size,
            shuffle=True,
            #max_nr_batches=max_nr_batches,
            epochs=1,
            optimizer="SGD",
            optimizer_args=self.arguments.optimizer_params
        )
        train_config.send(worker)

        loss = await worker.async_fit(dataset_key="#train", return_ids=[0])
        
        updated_model = train_config.model_ptr.get().obj
        updated_criterion = train_config.loss_fn_ptr.get().obj
        print("Updated Criterion:", updated_criterion)
        return updated_model, criterion, loss


    def perform_FL_training(self, datasets, is_shared=False):
        """ Performs a remote federated learning cycle leveraging PySyft.

        Args:
            datasets (sy.FederatedDataLoader): Distributed training datasets
        Returns:
            trained global model (Model)
            Cached local models  (dict(Model))
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
                print("loss", loss, loss.clone().detach().get())

                print("Mu:", self.mu, "L1:", self.l1_lambda, "L2:", self.l2_lambda)

                # Calculate regularisation terms
                fedprox_reg_terms = []
                l1_reg_terms = []
                l2_reg_terms = []
                for layer, layer_w in w.items():
                    
                    # Extract corresponding global layer weights
                    layer_wt = wt[layer]

                    print("Layer W:", layer_w.clone().detach().get())
                    print("Layer WT:", layer_wt.clone().detach().get())
                    print("layer W - Layer WT:", layer_w.clone().detach().get() - layer_wt.clone().detach().get())
                    print("Norm diff:", th.norm(layer_w.clone().detach().get() - layer_wt.clone().detach().get()))

                    print("Layer w - layer wt --> pointer:", layer_w - layer_wt)
                    print("Layer w - layer wt (retrieved):", (layer_w - layer_wt).clone().detach().get())
                    print("Norm diff --> pointer:", th.norm(layer_w - layer_wt))
                    print("Max value --> pointer:", th.max(layer_w -layer_wt))

                    # Note: In syft==0.2.4, 
                    # 1) `th.norm(<PointerTensor>)` will always return 
                    #    `tensor(0.)`, hence the need to manually apply the 
                    #    regularisation formulas. However, in future versions 
                    #    when this issue is solved, revert back to cleaner 
                    #    implementation using `th.norm`.

                    # Calculate FedProx regularisation
                    """ 
                    [REDACTED in syft==0.2.4]
                    norm_diff = th.norm(layer_w - layer_wt)
                    fp_reg_term = self.mu * 0.5 * (norm_diff**2)
                    """
                    norm_diff = th.pow((layer_w - layer_wt), 2).sum()
                    print("Manual norm diff:", norm_diff.clone().detach().get())
                    fp_reg_term = self.mu * 0.5 * norm_diff # exp cancelled out
                    print("Manual calculation of fedprox reg term:", fp_reg_term, fp_reg_term.clone().detach().get())
                    fedprox_reg_terms.append(fp_reg_term)
                    
                    # Calculate L1 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l1_norm = th.norm(layer_w, p=1)
                    l1_reg_term = self.l1_lambda * l1_norm
                    """
                    l1_norm = layer_w.abs().sum()
                    print("Manual L1 Norm:", l1_norm)
                    l1_reg_term = self.l1_lambda * l1_norm
                    print("Manual calculation of l1 reg term:", l1_reg_term, l1_reg_term.clone().detach().get())
                    l1_reg_terms.append(l1_reg_term)
                    
                    # Calculate L2 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l2_norm = th.norm(layer_w, p=2)
                    l2_reg_term = self.l2_lambda * 0.5 * (l2_norm)**2
                    """
                    l2_norm = th.pow(layer_w, 2).sum()
                    print("Manual L2 Norm:", l2_norm)
                    l2_reg_term = self.l2_lambda * 0.5 * l2_norm
                    print("Manual calculation of l2 reg term:", l2_reg_term, l2_reg_term.clone().detach().get())
                    l2_reg_terms.append(l2_reg_term)
                
                print("Fedprox reg terms", fedprox_reg_terms)
                print("l1 reg terms", l1_reg_terms)
                print("l2 reg terms", l2_reg_terms)

                # Retrieve worker involved
                assert outputs.location is labels.location
                worker = labels.location
                
                # Summing up from a list instead of in-place changes 
                # prevents the breaking of the autograd's computation graph
                fedprox_loss = th.stack(fedprox_reg_terms).sum().requires_grad_()
                print("Fedprox contribution", fedprox_loss, fedprox_loss.numel())

                l1_loss = th.stack(l1_reg_terms).sum().requires_grad_()
                print("l1 contribution", l1_loss, l1_loss.numel())


                l2_loss = th.stack(l2_reg_terms).sum().requires_grad_()
                print("l2 contribution", l2_loss, l2_loss.numel())
                        
                # Add up all losses involved
                surrogate_loss = loss + fedprox_loss + l1_loss + l2_loss
                print("Surrogate loss", surrogate_loss.clone().detach().get(), surrogate_loss.numel())
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
            
            
        def generate_local_models(is_snn=False):
            """ Abstracts the generation of local models in a federated learning
                context. For default FL training (i.e. non-SNN/FedAvg/Fedprox),
                local models generated are clones of the previous round's global
                model. Conversely, in SNN, the local models are instances of
                participant-specified models with supposedly pre-optimised
                architectures.

            Args:
                is_snn (bool): Toggles which type of context-specific local 
                               models to generate
            Returns:
                Distributed context-specific local models (dict(str, Model))
            """
            if not is_snn:
                local_models = {
                    w: copy.deepcopy(self.global_model).send(w)
                    for w in self.workers
                }

            else:
                raise NotImplementedError("SNN training not yet supported!")
            
            return local_models


        def perform_parallel_training(datasets, models, cache, optimizers, 
                                      schedulers, criterions, stoppers, epochs):
            """ Parallelizes training across each distributed dataset 
                (i.e. simulated worker) Parallelization here refers to the 
                training of all distributed models per epoch.
                Note: All objects involved in this set of operations have
                      already been distributed to their respective workers

            Args:
                datasets   (dict(DataLoader)): Distributed training datasets
                models     (dict(nn.Module)): Local models
                cache      (dict(nn.Module)): Cached models from previous rounds
                optimizers (dict(th.optim)): Local optimizers
                schedulers (dict(lr_scheduler)): Local LR schedulers
                criterions (dict(th.nn)): Custom local objective function
                stoppers   (dict(EarlyStopping)): Local early stopping drivers
                epochs (int): No. of epochs to train each local model
            Returns:
                trained local models
            """ 
            # Tracks which workers have reach an optimal/stagnated model
            WORKERS_STOPPED = []

            async def train_worker(worker, data_loader):
                """
                """
                # Extract essentials for training
                curr_global_model = cache[worker]
                curr_local_model = models[worker]
                curr_optimizer = optimizers[worker]
                curr_criterion = criterions[worker]

                # Extract essentials for adaptation
                curr_scheduler = schedulers[worker]
                curr_stopper = stoppers[worker]

                print(f"{worker} - Current GM:", [p.clone().detach().get() for p in list(curr_global_model.parameters())])
                print(f"{worker} - Current LM:", [p.clone().detach().get() for p in list(curr_local_model.parameters())])

                # Check if worker has been stopped
                if worker not in WORKERS_STOPPED:

                    for data, labels in data_loader:
                        
                        # Zero gradients to prevent accumulation  
                        curr_local_model.train()
                        curr_optimizer.zero_grad()

                        # Forward Propagation
                        print("Data shape:", data.clone().detach().get().shape)
                        print("Labels shape:", labels.clone().detach().get().shape)
                        print("Curr_local model:", curr_local_model)
                        predictions = curr_local_model(data.float())

                        print("Prediction shape:", predictions.clone().detach().get().shape)

                        if self.arguments.is_condensed:
                            print("Is condensed?", self.arguments.is_condensed)
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

                        print("Updated loss:", loss)

                        # Backward propagation
                        loss.backward()
                        curr_optimizer.step()
                        
                    # Retrieve final loss computed for this epoch for evaluation
                    final_batch_loss = curr_criterion.log()
                    curr_stopper(final_batch_loss, curr_local_model)

                    # If model is deemed to have stagnated, stop training
                    if curr_stopper.early_stop:
                        WORKERS_STOPPED.append(worker)
                        
                    # else, perform learning rate decay
                    else:
                        curr_scheduler.step()

                    # Update models, optimisers & losses
                    models[worker] = curr_local_model
                    optimizers[worker] = curr_optimizer
                    criterions[worker] = curr_criterion

                    assert (models[worker] == curr_local_model and 
                            optimizers[worker] == curr_optimizer and 
                            criterions[worker] == curr_criterion)

            async def train_on_batch(datasets):
                """
                """
                # Apply asynchronous training to each batch
                futures = [
                    train_worker(
                        worker=worker, 
                        data_loader=data_loader
                    ) for worker, data_loader in datasets.items()
                ]
                await asyncio.gather(*futures)


            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                for e in range(epochs):

                    asyncio.get_event_loop().run_until_complete(
                        train_on_batch(datasets=datasets)
                    )

            finally:
                loop.close()

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
            for worker, data_loader in datasets.items():
                for data, labels in data_loader:
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

        # However, due to certain PySyft nuances (refer to Part 4, section 1: 
        # Frame of Reference) there is a need to choose a conceptual 
        # representation of the overall architecture. Here, the node agnostic 
        # variant is implemented. Model is stored in the server -> Client 
        # (i.e. 'Me') does not interact with it
        
        # Note: If MPC is requested, global model itself cannot be shared, only 
        # its copies are shared. This is due to restrictions in PointerTensor 
        # mechanics.
        
        global_train_stopper = EarlyStopping(**self.arguments.early_stopping_params)
        global_val_stopper = EarlyStopping(**self.arguments.early_stopping_params)

        rounds = 0
        pbar = tqdm(total=self.arguments.rounds, desc='Rounds', leave=True)
        while rounds < self.arguments.rounds:

            # Generate K copies of template model, representing local models for
            # each worker in preparation for parallel training, and send them to
            # their designated workers
            # Note: This step is crucial because it is able prevent pointer 
            #       mutation, which comes as a result of copying pointers (refer
            #       to Part 4, section X), specifically if the global pointer 
            #       was copied directly.
            local_models = generate_local_models(is_snn=self.arguments.is_snn)

            # Model weights from previous round for subsequent FedProx 
            # comparison. Due to certain nuances stated below, they have to be
            # specified here. 
            # Note - In syft==0.2.4: 
            # 1) copy.deepcopy(PointerTensor) causes "TypeError: clone() got an 
            #    unexpected keyword argument 'memory_format'"
            # 2) Direct cloning of dictionary of models causes "TypeError: can't 
            #    pickle module objects"
            prev_models = generate_local_models(is_snn=self.arguments.is_snn)

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

            """
            criterions = {
                w: SurrogateCriterion(
                    **self.arguments.criterion_params
                ) for w,m in local_models.items()
            }
            """

            criterions = {
                w: SurrogateCriterion(**self.arguments.criterion_params)
                for w,m in local_models.items()
            }
            
            stoppers = {
                w: EarlyStopping(
                    **self.arguments.early_stopping_params
                ) for w,m in local_models.items()
            }
            
            trained_models, _, _, _, _= perform_parallel_training(
                datasets=datasets, 
                models=local_models,
                cache=prev_models,
                optimizers=optimizers, 
                schedulers=schedulers,
                criterions=criterions, 
                stoppers=stoppers,
                epochs=self.arguments.epochs
            )

            aggregated_params = calculate_global_params(
                self.global_model, 
                trained_models, 
                datasets
            )

            # Update weights with aggregated parameters 
            self.global_model.load_state_dict(aggregated_params)
            
            # Update cache for local models
            self.local_models = {w.id:lm for w,lm in trained_models.items()}

            # Check if early stopping is possible for global model
            final_local_losses = {
                w.id: c._cache[-1]
                for w,c in criterions.items()
            }
            global_loss = th.mean(th.stack(tuple(final_local_losses.values())))

            # Store local losses for analysis
            for w_id, loss in final_local_losses.items():
                try:
                    self.loss_history['local'][w_id].update({rounds: loss.item()})
                except KeyError:
                    self.loss_history['local'][w_id] = {rounds: loss.item()}

            # Store global losses for analysis
            self.loss_history['global'].update({rounds: global_loss.item()})


            """
                        predictions = curr_local_model(data.float())

                        print("Prediction shape:", predictions.clone().detach().get().shape)

                        if self.arguments.is_condensed:
                            print("Is condensed?", self.arguments.is_condensed)
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
            """


            global_train_stopper(global_loss, self.global_model)
            global_val_stopper(global_loss, self.global_model)
            
            # If global model is deemed to have stagnated, stop training
            if global_train_stopper.early_stop or global_val_stopper.early_stop:
                break
            
            rounds += 1
            pbar.update(1)
        
        pbar.close()

        return self.global_model, self.local_models

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
            train_datasets, eval_datasets, test_datasets = self.setup_FL_env()

            # Generate federated minibatches via loaders 
            train_loader, eval_loader, test_loader = self.convert_to_FL_batches(
                train_datasets, 
                eval_datasets,
                test_datasets
            )

            # Store federated data loaders for subsequent use
            self.train_loader = train_loader
            self.eval_loader = eval_loader
            self.test_loader = test_loader

        return self.train_loader, self.eval_loader, self.test_loader

        
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

    
    def validate(self):
        """ Performs validation on the global model using pre-specified 
            validation datasets

        Returns:
            
        """
        pass
        

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
        out_paths = {}

        # Export global model to file
        global_model_out_path = os.path.join(out_dir, "global_model.pt")
        th.save(self.global_model.state_dict(), global_model_out_path)

        # Export global loss history to file
        global_loss_out_path = os.path.join(out_dir, "global_loss_history.json")
        with open(global_loss_out_path, 'w') as glp:
            print("Global Loss History:", self.loss_history['global'])
            json.dump(self.loss_history['global'], glp)

        # Package global metadata for storage
        out_paths['global'] = {
            "origin": self.crypto_provider.id,
            "path": global_model_out_path,
            "loss_history": global_loss_out_path
        }

        for idx, (worker_id, local_model) in enumerate(self.local_models.items(), start=1):

            # Export local model to file
            local_model_out_path = os.path.join(
                out_dir, 
                f"local_model_{worker_id}.pt"
            )
            th.save(local_model.state_dict(), local_model_out_path)
            
            # Export local loss history to file
            local_loss_out_path = os.path.join(
                out_dir, 
                f"local_loss_history_{worker_id}.json"
            )
            with open(local_loss_out_path, 'w') as llp:
                print("Local Loss History:", self.loss_history['local'][worker_id])
                json.dump(self.loss_history['local'][worker_id], llp)

            # Package local metadata for storage
            out_paths[f'local_{idx}'] = {
                "origin": worker_id,
                "path": local_model_out_path,
                "loss_history": local_loss_out_path
            }

        return out_paths








##############
# Deprecated #
##############

"""

"""