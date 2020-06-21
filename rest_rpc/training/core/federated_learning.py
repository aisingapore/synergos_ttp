#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import aiohttp
import asyncio
import copy
import json
import logging
import os
from collections import OrderedDict, defaultdict
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
import torch as th
import tensorflow as tft
from sklearn.metrics import (
    accuracy_score, 
    roc_curve,
    roc_auc_score, 
    auc, 
    precision_recall_curve, 
    precision_score,
    recall_score,
    f1_score, 
    confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix
from syft.messaging.message import ObjectMessage
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from tqdm import tqdm

# Custom
from config import seed_everything
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.early_stopping import EarlyStopping
from rest_rpc.training.core.model import Model

##################
# Configurations #
##################

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

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
    def __init__(
        self, 
        arguments: Arguments, 
        crypto_provider: sy.VirtualWorker, 
        workers: list, 
        model: Model,
        out_dir: str = '.',
        loop=None
    ):
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
        self.loss_history = {
            'global': {
                'train': {},
                'evaluate': {}
            },
            'local': {}
        }

        # Optimisation attributes
        self.loop = loop

        # Export Attributes
        self.out_dir = out_dir
        self.checkpoints = {}

        # Lock random states within server
        seed_everything(seed=self.arguments.seed)

    ############
    # Checkers #
    ############

    def is_data_loaded(self):
        """ Checks if data has already been loaded 
        
        Returns:
            loaded state (bool)
        """
        return (
            self.train_loader is not None and 
            self.eval_loader is not None and 
            self.test_loader is not None
        )

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


    def setup_FL_env(
        self, 
        is_shared: bool=False
    ) -> Tuple[Dict[sy.WebsocketClientWorker, sy.BaseDataset], ...]:

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
                datasets[curr_worker] = data_ptr

            return datasets

        ###########################
        # Implementation Footnote # 
        ###########################
    
        # TTP should not have residual tensors during training, but will have
        # them during evaluation, as a result of trained models being loaded.
        
        # Retrieve Pointer Tensors to remote datasets
        train_datasets = convert_to_datasets("#train")
        eval_datasets = convert_to_datasets("#evaluate")
        test_datasets = convert_to_datasets("#predict")
        
        return train_datasets, eval_datasets, test_datasets
    
    
    def convert_to_FL_batches(self, 
        train_datasets: dict, 
        eval_datasets: dict, 
        test_datasets: dict,
        shuffle: bool=True
    ) -> Tuple[sy.FederatedDataLoader, ...]: 
        """ Supplementary function to convert initialised datasets into SGD
            compatible dataloaders in the context of PySyft's federated learning
            
        Args:
            train_datasets (dict(sy.BaseDataset)): 
                Distributed datasets for training
            eval_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for verifying performance
            test_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for to be tested on
            shuffle (bool): Toggles the way the minibatches are generated
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
                shuffle=shuffle,
                iter_per_worker=True,   # for LVL 1A parallelization
                #iter_per_worker=False,  # for LVL 1B parallelization
                **kwargs
            )

            return federated_data_loader

        # Load datasets into a configured federated dataloader
        train_loader = construct_FL_loader(train_datasets.values())
        eval_loader = construct_FL_loader(eval_datasets.values())
        test_loader = construct_FL_loader(test_datasets.values())
        
        return train_loader, eval_loader, test_loader

        # train_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in train_datasets.items()
        # }

        # eval_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in eval_datasets.items()
        # }

        # test_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in test_datasets.items()
        # }

        # return train_loaders, eval_loaders, test_loaders


    def build_custom_criterion(self):
        """ Augments a selected criterion with the ability to use FedProx

        Returns:
            Surrogate criterion (SurrogateCriterion)
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
                logging.debug(f"BCE Loss: {loss.location}")

                # Calculate regularisation terms
                # Note: All regularisation terms have to be collected in some 
                #       iterable first before summing up because in-place 
                #       operation break PyTorch's computation graph
                fedprox_reg_terms = []
                l1_reg_terms = []
                l2_reg_terms = []
                for layer, layer_w in w.items():
                    
                    # Extract corresponding global layer weights
                    layer_wt = wt[layer]

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
                    fp_reg_term = self.mu * 0.5 * norm_diff # exp cancelled out
                    fedprox_reg_terms.append(fp_reg_term)
                    
                    # Calculate L1 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l1_norm = th.norm(layer_w, p=1)
                    l1_reg_term = self.l1_lambda * l1_norm
                    """
                    l1_norm = layer_w.abs().sum()
                    l1_reg_term = self.l1_lambda * l1_norm
                    l1_reg_terms.append(l1_reg_term)
                    
                    # Calculate L2 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l2_norm = th.norm(layer_w, p=2)
                    l2_reg_term = self.l2_lambda * 0.5 * (l2_norm)**2
                    """
                    l2_norm = th.pow(layer_w, 2).sum()
                    l2_reg_term = self.l2_lambda * 0.5 * l2_norm
                    l2_reg_terms.append(l2_reg_term)
                
                # Summing up from a list instead of in-place changes 
                # prevents the breaking of the autograd's computation graph
                fedprox_loss = th.stack(fedprox_reg_terms).sum()
                l1_loss = th.stack(l1_reg_terms).sum()
                l2_loss = th.stack(l2_reg_terms).sum()

                # Add up all losses involved
                surrogate_loss = loss + fedprox_loss + l1_loss + l2_loss

                # Store result in cache
                self.__temp.append(surrogate_loss)
                
                return surrogate_loss

            def log(self):
                """ Computes mean loss across all current runs & caches the result """
                avg_loss = th.mean(th.stack(self.__temp), dim=0)
                self._cache.append(avg_loss)
                self.__temp.clear()
                return avg_loss
            
            def reset(self):
                self.__temp = []
                self._cache = []
                return self

        return SurrogateCriterion


    def perform_FL_training(self, datasets, is_shared=False):
        """ Performs a remote federated learning cycle leveraging PySyft.

        Args:
            datasets (sy.FederatedDataLoader): Distributed training datasets
        Returns:
            trained global model (Model)
            Cached local models  (dict(Model))
        """
        
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
                    w: copy.deepcopy(self.global_model)#.send(w)
                    for w in self.workers
                }

            else:
                raise NotImplementedError("SNN training not yet supported!")
            
            return local_models
        
        def perform_parallel_training(
            datasets: dict, 
            models: dict, 
            cache: dict, 
            optimizers: dict, 
            schedulers: dict, 
            criterions: dict, 
            stoppers: dict, 
            rounds: int,
            epochs: int
        ):
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
                rounds (int): Current round of training
                epochs (int): No. of epochs to train each local model
            Returns:
                trained local models
            """ 
            # Tracks which workers have reach an optimal/stagnated model
            WORKERS_STOPPED = Manager().list()

            async def train_worker(packet):
                """ Train a worker on its single batch, and does an in-place 
                    updates for its local model, optimizer & criterion 
                
                Args:
                    packet (dict):
                        A single packet of data containing the worker and its
                        data to be trained on 

                """ 
                worker, (data, labels) = packet

                logging.debug(f"Data: {data}, {type(data)}, {data.shape}")
                logging.debug(f"Labels: {labels}, {type(labels)}, {labels.shape}")

                for i in list(self.global_model.parameters()):
                    logging.debug(f"Model parameters: {i}, {type(i)}, {i.shape}")

                # Extract essentials for training
                curr_global_model = cache[worker]
                curr_local_model = models[worker]
                curr_optimizer = optimizers[worker]
                curr_criterion = criterions[worker]

                # Check if worker has been stopped
                if worker.id not in WORKERS_STOPPED:

                    logging.debug(f"Before training - Local Gradients for {worker}:\n {list(curr_local_model.parameters())[0].grad}")
                    # curr_global_model = self.secret_share(curr_global_model)
                    # curr_local_model = self.secret_share(curr_local_model)
                    curr_global_model = curr_global_model.send(worker)
                    curr_local_model = curr_local_model.send(worker)

                    # Zero gradients to prevent accumulation  
                    curr_local_model.train()
                    curr_optimizer.zero_grad()

                    # Forward Propagation
                    predictions = curr_local_model(data.float())

                    if self.arguments.is_condensed:
                        # To handle binomial context
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

                    curr_global_model = curr_global_model.get()
                    curr_local_model = curr_local_model.get()
                    logging.debug(f"After training - Local Gradients for {worker}:\n {list(curr_local_model.parameters())[0].grad}")

                # Update all involved objects
                assert models[worker] is curr_local_model
                assert optimizers[worker] is curr_optimizer
                assert criterions[worker] is curr_criterion

            async def train_batch(batch):
                """ Asynchronously train all workers on their respective 
                    allocated batches 

                Args:
                    batch (dict): 
                        A single batch from a sliced dataset stratified by
                        workers and their respective packets. A packet is a
                        tuple pairing of the worker and its data slice
                        i.e. (worker, (data, labels))
                """
                for worker_future in asyncio.as_completed(
                    map(train_worker, batch.items())
                ):
                    await worker_future

            async def check_for_stagnation(worker):
                """ After a full epoch, check if training for worker has 
                    stagnated

                Args:
                    worker (WebsocketServerWorker): Worker to be evaluated
                """
                # Extract essentials for adaptation
                curr_local_model = models[worker]
                curr_criterion = criterions[worker]
                curr_scheduler = schedulers[worker]
                curr_stopper = stoppers[worker]

                # Check if worker has been stopped
                if worker.id not in WORKERS_STOPPED:

                    # Retrieve final loss computed for this epoch for evaluation
                    final_batch_loss = curr_criterion.log()
                    curr_stopper(final_batch_loss, curr_local_model)

                    # If model is deemed to have stagnated, stop training
                    if curr_stopper.early_stop:
                        WORKERS_STOPPED.append(worker.id)
                        
                    # else, perform learning rate decay
                    else:
                        curr_scheduler.step()

                assert schedulers[worker] is curr_scheduler
                assert stoppers[worker] is curr_stopper 

            async def train_datasets(datasets):
                """ Train all batches in a composite federated dataset """
                # Note: All TRAINING must be synchronous w.r.t. each batch, so
                #       that weights can be updated sequentially!
                for batch in datasets:
                    logging.debug("-"*90)
                    await train_batch(batch)
                
                logging.debug(f"Before stagnation evaluation: Workers stopped: {WORKERS_STOPPED}")
                stagnation_futures = [
                    check_for_stagnation(worker) 
                    for worker in self.workers
                ]
                await asyncio.gather(*stagnation_futures)
                logging.debug(f"After stagnation evaluation: Workers stopped: {WORKERS_STOPPED}")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                for epoch in range(epochs):

                    asyncio.get_event_loop().run_until_complete(
                        train_datasets(datasets=datasets)
                    )

                    # Update cache for local models
                    self.local_models = {w.id:lm for w,lm in models.items()}

                    # Export ONLY local models. Losses will be accumulated and
                    # cached. This is to prevent the autograd computation graph
                    # from breaking and interfering with weight updates
                    round_key = f"round_{rounds}"
                    epoch_key = f"epoch_{epoch}"
                    checkpoint_dir = os.path.join(
                        self.out_dir, 
                        "checkpoints",
                        round_key, 
                        epoch_key
                    )
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    grid_checkpoint = self.export(checkpoint_dir
                    )
                    for _, logs in grid_checkpoint.items():
                        origin = logs.pop('origin')
                        worker_archive = self.checkpoints.get(origin, {})
                        round_archive = worker_archive.get(round_key, {})
                        round_archive.update({epoch_key: logs})
                        worker_archive.update(round_archive)
                        self.checkpoints.update(worker_archive)

            finally:
                loop.close()

            return models, optimizers, schedulers, criterions, stoppers

    
        def calculate_global_params(global_model, models, datasets):
            """ Aggregates weights from locally trained models after a round.

                Note: 
                    This is based on the assumption that querying database size 
                    does not break FL abstraction (i.e. unwilling to share 
                    quantity)

            Args:
                global_model (nn.Module): Global model to be trained federatedly
                models (dict(nn.Module)): Trained local models
                datasets (dict(sy.FederatedDataLoader)): Distributed datasets
            Returns:
                Aggregated parameters (OrderedDict)
            """
            param_types = global_model.state_dict().keys()
            model_states = {w: m.state_dict() for w,m in models.items()}

            # Find size of all distributed datasets for computing scaling factor
            obs_counts = {}
            for batch in datasets:
                for worker, (data, _) in batch.items():
                    obs_counts[worker] = obs_counts.get(worker, 0) + len(data)

            # Calculate scaling factors for each worker
            scale_coeffs = {
                worker: local_count/sum(obs_counts.values()) 
                for worker, local_count in obs_counts.items()
            }

            # PyTorch models can only swap weights of the same structure. Hence,
            # aggregate weights while maintaining original layering structure
            aggregated_params = OrderedDict()
            for p_type in param_types:

                param_states = [
                    th.mul(
                        model_states[w][p_type],
                        scale_coeffs[w]
                    ) for w in self.workers
                ]

                layer_shape = tuple(global_model.state_dict()[p_type].shape)

                aggregated_params[p_type] = th.stack(
                    param_states,
                    dim=0
                ).sum(dim=0).view(*layer_shape)

            return aggregated_params
 
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

        global_val_stopper = EarlyStopping(**self.arguments.early_stopping_params)

        rounds = 0
        pbar = tqdm(total=self.arguments.rounds, desc='Rounds', leave=True)
        while rounds < self.arguments.rounds:

            logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
            logging.debug(f"Global Gradients:\n {list(self.global_model.parameters())[0].grad}")

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

            criterions = {
                w: self.build_custom_criterion()(
                    **self.arguments.criterion_params
                ) for w,m in local_models.items()
            }
            
            stoppers = {
                w: EarlyStopping(
                    **self.arguments.early_stopping_params
                ) for w,m in local_models.items()
            }
            
            (retrieved_models, _, _, _, _) = perform_parallel_training(
                datasets=datasets, 
                models=local_models,
                cache=prev_models,
                optimizers=optimizers, 
                schedulers=schedulers,
                criterions=criterions, 
                stoppers=stoppers,
                rounds=rounds,
                epochs=self.arguments.epochs
            )

            # Retrieve all models from their respective workers
            logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
            aggregated_params = calculate_global_params(
                self.global_model, 
                retrieved_models, 
                datasets
            )

            # Update weights with aggregated parameters 
            self.global_model.load_state_dict(aggregated_params)
            logging.debug(f"New global model:\n {self.global_model.state_dict()}")

            final_local_losses = {
                w.id: c._cache[-1].get()
                for w,c in criterions.items()
            }

            # Store local losses for analysis
            for w_id, loss in final_local_losses.items():
                local_loss_archive = self.loss_history['local'].get(w_id, {})
                local_loss_archive.update({rounds: loss.item()})
                self.loss_history['local'][w_id] = local_loss_archive

            global_train_loss = th.mean(
                th.stack(list(final_local_losses.values())),
                dim=0
            )

            # Validate the global model
            _, evaluation_losses = self.evaluate(metas=['evaluate'])
            global_val_loss = evaluation_losses['evaluate']

            # Store global losses for analysis
            global_loss_archive = self.loss_history['global']
            global_train_losses = global_loss_archive.get('train', {})
            global_train_losses.update({rounds: global_train_loss.item()})
            global_val_losses = global_loss_archive.get('evaluate', {})
            global_val_losses.update({rounds: global_val_loss.item()})
            self.loss_history['global'] = {
                'train': global_train_losses,
                'evaluate': global_val_losses
            }

            # If global model is deemed to have stagnated, stop training
            global_val_stopper(global_val_loss, self.global_model)
            if global_val_stopper.early_stop:
                logging.info("Global model has stagnated. Training round terminated!\n")
                break

            rounds += 1
            pbar.update(1)
        
        pbar.close()

        logging.debug(f"Objects in TTP: {self.crypto_provider}, {len(self.crypto_provider._objects)}")
        logging.debug(f"Objects in sy.local_worker: {sy.local_worker}, {len(sy.local_worker._objects)}")

        return self.global_model, self.local_models


    def perform_FL_evaluation(self, datasets, workers=[], is_shared=True, **kwargs): 
        """ Obtains predictions given a validation/test dataset upon 
            a specified trained global model.
            
        Args:
            datasets (tuple(th.Tensor)): A validation/test dataset
            workers (list(str)): Filter to select specific workers to infer on
            is_shared (bool): Toggles whether SMPC is turned on
            **kwargs: Miscellaneous keyword arguments for future use
        Returns:
            Tagged prediction tensor (sy.PointerTensor)
        """    
        async def evaluate_worker(packet):
            """ Evaluate a worker on its single packet of minibatch data
            
            Args:
                packet (dict):
                    A single packet of data containing the worker and its
                    data to be evaluated upon 

            """ 

            try:
                worker, (data, labels) = packet
            except TypeError:
                (data, labels) = packet
                assert data.location is labels.location
                worker = data.location

            logging.debug(f"Data: {data}, {type(data)}, {data.shape}")
            logging.debug(f"Labels: {labels}, {type(labels)}, {labels.shape}")

            for i in list(self.global_model.parameters()):
                logging.debug(f"Model parameters: {i}, {type(i)}, {i.shape}")

            # Skip predictions if filter was specified, and current worker was
            # not part of the selected workers
            if workers and (worker.id not in workers):
                return {}

            self.global_model = self.global_model.send(worker)
            self.local_models[worker.id] = self.local_models[worker.id].send(worker)

            self.global_model.eval()
            self.local_models[worker.id].eval()
            with th.no_grad():

                outputs = self.global_model(data).detach()

                if self.arguments.is_condensed:
                    predictions = (outputs > 0.5).float()
                    
                else:
                    # Find best predicted class label representative of sample
                    _, predicted_labels = outputs.max(axis=1)
                    
                    # One-hot encode predicted labels
                    predictions = th.FloatTensor(labels.shape)
                    predictions.zero_()
                    predictions.scatter_(1, predicted_labels.view(-1,1), 1)

                # Compute loss
                surrogate_criterion = self.build_custom_criterion()(
                    **self.arguments.criterion_params
                )
                if self.arguments.is_condensed:
                    # To handle binomial context
                    loss = surrogate_criterion(
                        outputs, 
                        labels.float(),
                        w=self.local_models[worker.id].state_dict(),
                        wt=self.global_model.state_dict()
                    )
                else:
                    # To handle multinomial context
                    loss = surrogate_criterion(
                        outputs, 
                        th.max(labels, 1)[1],
                        w=self.local_models[worker.id].state_dict(),
                        wt=self.global_model.state_dict()
                    )

            self.local_models[worker.id] = self.local_models[worker.id].get()
            self.global_model = self.global_model.get()

            #############################################
            # Inference V1: Assume TTP's role is robust #
            #############################################
            # In this version, TTP's coordination is not deemed to be breaking
            # FL rules. Hence predictions & labels can be pulled in locally for
            # calculating statistics, before sending the labels back to worker.

            # labels = labels.get()
            # outputs = outputs.get()
            # predictions = predictions.get()

            # logging.debug(f"labels: {labels}, outputs: {outputs}, predictions: {predictions}")

            # # Calculate accuracy of predictions
            # accuracy = accuracy_score(labels.numpy(), predictions.numpy())
            
            # # Calculate ROC-AUC for each label
            # roc = roc_auc_score(labels.numpy(), outputs.numpy())
            # fpr, tpr, _ = roc_curve(labels.numpy(), outputs.numpy())
            
            # # Calculate Area under PR curve
            # pc_vals, rc_vals, _ = precision_recall_curve(labels.numpy(), outputs.numpy())
            # auc_pr_score = auc(rc_vals, pc_vals)
            
            # # Calculate F-score
            # f_score = f1_score(labels.numpy(), predictions.numpy())

            # # Calculate contingency matrix
            # ct_matrix = contingency_matrix(labels.numpy(), predictions.numpy())
            
            # # Calculate confusion matrix
            # cf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())
            # logging.debug(f"Confusion matrix: {cf_matrix}")

            # TN, FP, FN, TP = cf_matrix.ravel()
            # logging.debug(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

            # # Sensitivity, hit rate, recall, or true positive rate
            # TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
            # # Specificity or true negative rate
            # TNR = TN/(TN+FP) if (TN+FP) != 0 else 0
            # # Precision or positive predictive value
            # PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
            # # Negative predictive value
            # NPV = TN/(TN+FN) if (TN+FN) != 0 else 0
            # # Fall out or false positive rate
            # FPR = FP/(FP+TN) if (FP+TN) != 0 else 0
            # # False negative rate
            # FNR = FN/(TP+FN) if (TP+FN) != 0 else 0
            # # False discovery rate
            # FDR = FP/(TP+FP) if (TP+FP) != 0 else 0

            # statistics = {
            #     'accuracy': accuracy,
            #     'roc_auc_score': roc,
            #     'pr_auc_score': auc_pr_score,
            #     'f_score': f_score,
            #     'TPR': TPR,
            #     'TNR': TNR,
            #     'PPV': PPV,
            #     'NPV': NPV,
            #     'FPR': FPR,
            #     'FNR': FNR,
            #     'FDR': FDR,
            #     'TP': TP,
            #     'TN': TN,
            #     'FP': FP,
            #     'FN': FN
            # }

            # labels = labels.send(worker)

            # return {worker: statistics}

            ####################################################################
            # Inference V1.5: Assume TTP's role is robust, but avoid violation #
            ####################################################################
            # In this version, while TTP's coordination is also not deemed to be
            # breaking FL rules, the goal is to violate the minimum no. of
            # federated procedures. Here, only outputs & predictions can be 
            # pulled in locally, since they are deemed to be TTP-generated.
            # However, statistical calculation will be orchestrated to be done
            # at worker nodes, and be sent back via a flask payload. This way,
            # the TTP avoids even looking at client's raw data, only interacting
            # with derivative information. 

            outputs = outputs.get()
            predictions = predictions.get()
            loss = loss.get()

            return {worker: {"y_pred": predictions, "y_score": outputs}}, loss

            ####################################################################
            # Inference V2: Strictly enforce federated procedures in inference #
            ####################################################################

            # Override garbage collection to allow for post-inference tracking
            # data.set_garbage_collect_data(False)
            # labels.set_garbage_collect_data(False)
            # outputs.set_garbage_collect_data(False)
            # predictions.set_garbage_collect_data(False)

            # data_id = data.id_at_location
            # labels_id = labels.id_at_location
            # outputs_id = outputs.id_at_location
            # predictions_id = predictions.id_at_location

            # data = data.get()
            # labels = labels.get()
            # outputs = outputs.get()
            # # predictions = predictions.get()

            # logging.debug(f"Before transfer - Worker: {worker}")

            # worker._send_msg_and_deserialize("register_obj", obj=data.tag("#minibatch"))#, obj_id=data_id)
            # worker._send_msg_and_deserialize("register_obj", obj=labels.tag("#minibatch"))#, obj_id=labels_id)
            # worker._send_msg_and_deserialize("register_obj", obj=outputs.tag("#minibatch"))#, obj_id=outputs_id)
            # worker._send_msg_and_deserialize("register_obj", obj=predictions.tag("#minibatch"))#, obj_id=predictions_id)

            # logging.debug(f"After transfer - Worker: {worker}")

            # inferences = {
            #     worker: {
            #         'data': 1,
            #         'labels': 2,
            #         'outputs': 3,
            #         'predictions': 4 
            #     }
            # }

            # # Convert collection of object IDs accumulated from minibatch 
            # inferencer = Inferencer(inferences=inferences, **kwargs["keys"])
            # # converted_stats = inferencer.infer(reg_records=kwargs["registrations"])
            # converted_stats = await inferencer._collect_all_stats(reg_records=kwargs["registrations"])

            # self.global_model = self.global_model.get()
            # return converted_stats

        async def evaluate_batch(batch):
            """ Asynchronously train all workers on their respective 
                allocated batches 

            Args:
                batch (dict): 
                    A single batch from a sliced dataset stratified by
                    workers and their respective packets. A packet is a
                    tuple pairing of the worker and its data slice
                    i.e. (worker, (data, labels))
            """
            logging.debug(f"Batch: {batch}, {type(batch)}")

            batch_evaluations = {}
            batch_losses = []

            # If multiple prediction sets have been declared across all workers,
            # batch will be a dictionary i.e. {<worker_1>: (data, labels), ...}
            if isinstance(batch, dict):

                for worker_future in asyncio.as_completed(
                    map(evaluate_worker, batch)
                ):
                    evaluated_worker_batch, loss = await worker_future
                    batch_evaluations.update(evaluated_worker_batch)
                    batch_losses.append(loss)

            # If only 1 prediction set is declared (i.e. only 1 guest present), 
            # batch will be a tuple i.e. (data, label)
            elif isinstance(batch, tuple):

                evaluated_worker_batch, loss = await evaluate_worker(batch)
                batch_evaluations.update(evaluated_worker_batch)
                batch_losses.append(loss)

            return batch_evaluations, batch_losses

        async def evaluate_datasets(datasets):
            """ Train all batches in a composite federated dataset """
            # Note: Unlike in training, inference does not require any weight
            #       tracking, thus each batch can be processed asynchronously 
            #       as well!
            batch_futures = [evaluate_batch(batch) for batch in datasets]
            all_batch_evaluations = await asyncio.gather(*batch_futures)

            ##########################################################
            # Inference V1: Exercise TTP's role as secure aggregator #
            ##########################################################
            
            # all_worker_stats = {}
            # all_worker_preds = {}

            # for b_count, (batch_evaluations, batch_predictions) in enumerate(
            #     all_batch_evaluations, 
            #     start=1
            # ):
            #     for worker, batch_stats in batch_evaluations.items():

            #         # Manage statistical aggregation
            #         aggregated_stats = all_worker_stats.get(worker.id, {})
            #         for stat, value in batch_stats.items():
                        
            #             if stat in ["TN", "FP", "FN", "TP"]:
            #                 total_val = aggregated_stats.get(stat, 0.0) + value
            #                 aggregated_stats[stat] = total_val

            #             else:
            #                 sliding_stat_avg = (
            #                     aggregated_stats.get(stat, 0.0)*(b_count-1) + value
            #                 ) / b_count
            #                 aggregated_stats[stat] = sliding_stat_avg

            #         all_worker_stats[worker.id] = aggregated_stats

            ####################################################################
            # Inference V1.5: Assume TTP's role is robust, but avoid violation #
            ####################################################################

            all_worker_outputs = {}
            all_losses = []
            for batch_evaluations, batch_losses in all_batch_evaluations:

                for worker, outputs in batch_evaluations.items():

                    aggregated_outputs = all_worker_outputs.get(worker.id, {})
                    for _type, result in outputs.items():

                        aggregated_results = aggregated_outputs.get(_type, [])
                        aggregated_results.append(result)
                        aggregated_outputs[_type] = aggregated_results

                    all_worker_outputs[worker.id] = aggregated_outputs

                all_losses += batch_losses

            # Concatenate all batch outputs for each worker
            all_combined_outputs = {
                worker_id: {
                    _type: th.cat(res_collection, dim=0).numpy().tolist()
                    for _type, res_collection in batch_outputs.items()
                }
                for worker_id, batch_outputs in all_worker_outputs.items()
            }
            
            avg_loss = th.mean(th.stack(all_losses), dim=0)

            return all_combined_outputs, avg_loss

            ####################################################################
            # Inference V2: Strictly enforce federated procedures in inference #
            ####################################################################

            # for batch_evaluations in all_batch_evaluations:
            #     for worker, batch_obj_ids in batch_evaluations.items():

            #         minibatch_ids = all_worker_stats.get(worker.id, [])
            #         minibatch_ids.append(batch_obj_ids)
            #         all_worker_stats[worker.id] = minibatch_ids


        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_combined_outputs, avg_loss = asyncio.get_event_loop().run_until_complete(
                evaluate_datasets(datasets=datasets)
            )

        finally:
            loop.close()

        return all_combined_outputs, avg_loss

    ##################
    # Core functions #
    ##################
    
    def load(self, archive=None, shuffle=True):
        """ Prepares federated environment for training or inference. If archive
            is specified, restore all models tracked, otherwise, the default
            global model is used. All remote datasets will be loaded into 
            Federated dataloaders for batch operations if data has not already 
            been loaded. Note that loading is only done once, since searching 
            for datasets within a grid will cause query results to accumulate 
            exponentially on TTP. Hence, this function will only load datasets 
            ONLY if it has NOT already been loaded.

            Note:
                When `shuffle=True`, federated dataloaders will SHUFFLE datasets
                to ensure that proper class representation is covered in each
                minibatch generated. This an important aspect during training.

                When `shuffle=False`, federated dataloaders will NOT shuffle 
                datasets before minibatching. This is to ensure that the
                prediction labels can be re-assembled, aligned and restored on
                the worker nodes during evaluation/inference
            
        Args:
            archive (dict): Paths to exported global & local models
            shuffle (bool): Toggles the way the minibatches are generated
        Returns:
            train_loader (sy.FederatedLoader): Training data in configured batches
            test_loader (sy.FederatedLoader): Testing data in configured batches
        """
        if archive:

            for _, logs in archive.items():

                logging.debug(f"Logs: {logs}")
                archived_origin = logs['origin']

                if archived_origin == self.crypto_provider.id:
                    archived_model_weights = th.load(logs['path'])
                    self.global_model.load_state_dict(archived_model_weights)
                else:

                    ###########################
                    # Implementation Footnote #
                    ###########################

                    # Because local models in SNN will be optimal models owned
                    # by the participants themselves, there are 2 ways of 
                    # handling model archival - Store the full model, or get 
                    # participants to register the architecture & hyperparameter
                    # sets of their optimal setup, while exporting the model
                    # weights. The former allows models to be captured alongside
                    # their architectures, hence removing the need for tracking 
                    # additional information unncessarily. However, models 
                    # exported this way have limited use outside of REST-RPC, 
                    # since they are pickled relative to the file structure of 
                    # the package. The latter is the more flexible approach, 
                    # since weights will still remain usable even outside the
                    # context of REST-RPC, as long as local model architectures,
                    # are available.  
                    
                    if self.arguments.is_snn:
                        archived_model = th.load(logs['path'])
                    else:
                        archived_model_weights = th.load(logs['path'])
                        archived_model = copy.deepcopy(self.global_model)
                        archived_model.load_state_dict(archived_model_weights)
                    self.local_models[archived_origin] = archived_model          

        if not self.is_data_loaded():
            
            # Extract data pointers from workers
            train_datasets, eval_datasets, test_datasets = self.setup_FL_env()

            # Generate federated minibatches via loaders 
            train_loader, eval_loader, test_loader = self.convert_to_FL_batches(
                train_datasets, 
                eval_datasets,
                test_datasets,
                shuffle=shuffle
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


    def evaluate(
        self, 
        metas: List[str] = [], 
        workers: List[str] = [], 
        **kwargs
    ) -> Dict[str, Dict[str, th.Tensor]]:
        """ Using the current instance of the global model, performs inference 
            on pre-specified datasets.

        Args:
            metas (list(meta)): Meta tokens indicating which datasets are to be
                evaluated. If empty (default), all meta datasets (i.e. training,
                validation and testing) will be evaluated
        Returns:
            Inferences (dict(worker_id, dict(result_type, th.tensor)))
        """
        if not self.is_data_loaded():
            raise RuntimeError("Grid data has not been aggregated! Call '.load()' first & try again.")

        DATA_MAP = {
            'train': self.train_loader,
            'evaluate': self.eval_loader,
            'predict': self.test_loader
        }

        # If no meta filters are specified, evaluate all datasets 
        metas = list(DATA_MAP.keys()) if not metas else metas

        # If no worker filter are specified, evaluate all workers
        workers = [w.id for w in self.workers] if not workers else workers

        # Evaluate global model using datasets conforming to specified metas
        inferences = {}
        losses = {}
        for meta, dataset in DATA_MAP.items():

            if meta in metas:

                worker_meta_inference, avg_loss = self.perform_FL_evaluation(
                    datasets=dataset,
                    workers=workers,
                    is_shared=True,
                    **kwargs
                )

                # inference = worker -> meta -> (y_pred, y_score)
                for worker_id, meta_result in worker_meta_inference.items():

                    worker_results = inferences.get(worker_id, {})
                    worker_results[meta] = meta_result
                    inferences[worker_id] = worker_results

                losses[meta] = avg_loss
        
        return inferences, losses
      
    
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
    
    
    def export(self, out_dir: str = None, excluded: List[str] = []) -> dict:
        """ Exports the global model state dictionary to file
        
        Args:
            out_dir (str): Path to output directory for export
        Returns:
            Path-to-file (str)
        """

        def save_global_model():
            if 'global' in excluded: return None
            # Export global model to file
            global_model_out_path = os.path.join(
                out_dir, 
                "global_model.pt"
            )
            # Only states can be saved, since Model is not picklable
            th.save(self.global_model.state_dict(), global_model_out_path)
            return global_model_out_path

        def save_global_losses():
            if 'loss' in excluded: return None
            # Export global loss history to file
            global_loss_out_path = os.path.join(
                out_dir, 
                "global_loss_history.json"
            )
            with open(global_loss_out_path, 'w') as glp:
                print("Global Loss History:", self.loss_history['global'])
                json.dump(self.loss_history['global'], glp)
            return global_loss_out_path

        def save_worker_model(worker_id, model):
            if 'local' in excluded: return None
            # Export local model to file
            local_model_out_path = os.path.join(
                out_dir, 
                f"local_model_{worker_id}.pt"
            )
            if self.arguments.is_snn:
                # Local models are saved directly to log their architectures
                th.save(model, local_model_out_path)
            else:
                th.save(model.state_dict(), local_model_out_path)
            return local_model_out_path

        def save_worker_losses(worker_id):
            if 'loss' in excluded: return None
            # Export local loss history to file
            local_loss_out_path = os.path.join(
                out_dir, 
                f"local_loss_history_{worker_id}.json"
            )
            with open(local_loss_out_path, 'w') as llp:
                json.dump(self.loss_history['local'].get(worker_id, {}), llp)
            return local_loss_out_path

        # Override cached output directory with specified directory if any
        out_dir = out_dir if out_dir else self.out_dir

        out_paths = {}

        # Package global metadata for storage
        out_paths['global'] = {
            'origin': self.crypto_provider.id,
            'path': save_global_model(),
            'loss_history': save_global_losses(),
            'checkpoints': self.checkpoints.get(self.crypto_provider.id, {})
        }

        for idx, (worker_id, local_model) in enumerate(
            self.local_models.items(), 
            start=1
        ):
            # Package local metadata for storage
            out_paths[f'local_{idx}'] = {
                'origin': worker_id,
                'path': save_worker_model(worker_id, model=local_model),
                'loss_history': save_worker_losses(worker_id),
                'checkpoints': self.checkpoints.get(worker_id, {})
            }

        return out_paths