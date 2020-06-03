#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import asyncio
import copy
import json
import logging
import os
from collections import OrderedDict
from multiprocessing import Manager
from pathlib import Path

# Libs
import syft as sy
import torch as th
import tensorflow as tft
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from tqdm import tqdm

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
    def __init__(self, arguments, crypto_provider, workers, model, loop=None):
        
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

        # Optimisation attributes
        self.loop = loop

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
        #self.crypto_provider.clear_objects()
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
            train_datasets (dict(sy.BaseDataset)): 
                Distributed datasets for training
            eval_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for verifying performance
            test_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for to be tested on
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
                iter_per_worker=True,   # for LVL 1A parallelization
                #iter_per_worker=False,  # for LVL 1B parallelization
                **kwargs
            )

            return federated_data_loader

        # Load training datasets into a configured federated dataloader
        train_loader = construct_FL_loader(train_datasets.values())

        # Load validation datasets into a configured federated dataloader
        eval_loader = construct_FL_loader(eval_datasets.values())

        # Load testing datasets into a configured federated dataloader
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

                # Extract essentials for training
                curr_global_model = cache[worker]
                curr_local_model = models[worker]
                curr_optimizer = optimizers[worker]
                curr_criterion = criterions[worker]

                # Check if worker has been stopped
                if worker.id not in WORKERS_STOPPED:

                    logging.debug(f"Before training - Local Gradients for {worker}:\n {list(curr_local_model.parameters())[0].grad}")
                    curr_global_model = curr_global_model.send(worker.id)
                    curr_local_model = curr_local_model.send(worker.id)

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
                for _ in range(epochs):

                    asyncio.get_event_loop().run_until_complete(
                        train_datasets(datasets=datasets)
                    )

            finally:
                loop.close()

            return models, optimizers, schedulers, criterions, stoppers

    
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
            for batch in datasets:
                for worker, (data, _) in batch.items():
                    obs_counts[worker] = obs_counts.get(worker, 0) + len(data)

            # Calculate scaling factors for each worker
            scale_coeffs = {
                worker: local_count/sum(obs_counts.values()) 
                for worker, local_count in obs_counts.items()
            }

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
        # global_val_stopper = EarlyStopping(**self.arguments.early_stopping_params)

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
                w: SurrogateCriterion(
                    **self.arguments.criterion_params)
                for w,m in local_models.items()
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
                epochs=self.arguments.epochs
            )

            # # Retrieve all models from their respective workers
            # retrieved_models = {w: m.get() for w,m in trained_models.items()}
            logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
            aggregated_params = calculate_global_params(
                self.global_model, 
                retrieved_models, 
                datasets
            )

            # # Update weights with aggregated parameters 
            self.global_model.load_state_dict(aggregated_params)
            logging.debug(f"New global model:\n {self.global_model.state_dict()}")

            # Update cache for local models
            self.local_models = {w.id:lm for w,lm in retrieved_models.items()}

            # Check if early stopping is possible for global model
            final_local_losses = {
                w.id: c._cache[-1].get()
                for w,c in criterions.items()
            }
            global_loss = th.mean(
                th.stack(list(final_local_losses.values())),
                dim=0
            )

            # Store local losses for analysis
            for w_id, loss in final_local_losses.items():
                local_archive = self.loss_history['local'].get(w_id, {})
                local_archive.update({rounds: loss.item()})
                self.loss_history['local'][w_id] = local_archive

            # Store global losses for analysis
            self.loss_history['global'].update({rounds: global_loss.item()})

            # global_train_stopper(global_loss, self.global_model)
            # # global_val_stopper(global_loss, self.global_model)
            
            # # If global model is deemed to have stagnated, stop training
            # if global_train_stopper.early_stop:# or global_val_stopper.early_stop:
            #     break

            rounds += 1
            pbar.update(1)
        
        pbar.close()

        logging.info(f"Objects in TTP: {self.crypto_provider}, {len(self.crypto_provider._objects)}")
        logging.info(f"Objects in sy.local_worker: {sy.local_worker}, {len(sy.local_worker._objects)}")

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