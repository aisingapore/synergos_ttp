#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import os
import random
import timeit
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from config import seed_everything
from rest_rpc.connection.core.utils import RegistrationRecords
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.early_stopping import EarlyStopping
from rest_rpc.training.core.model import Model
from rest_rpc.training.core.algorithms.base import BaseAlgorithm
from rest_rpc.training.core.utils import RPCFormatter
from rest_rpc.evaluation.core.utils import Analyser

##################
# Configurations #
##################

seed_threshold = 0.15
metric = 'accuracy'
auto_align = True
metas=['train']
combination_keys = {
    'project_id': None,
    'expt_id': None,
    'run_id': None
}
max_epochs = 1000
registration_records = RegistrationRecords()
raw_registrations = registration_records.read(**combination_keys)
relevant_registrations = RPCFormatter().strip_keys(raw_registrations)

amundsen_metadata = {}

########################################
# Federated Algorithm Class - FedPrune #
########################################

class FedPrune(BaseAlgorithm):
    """ 
    Contains baseline functionality to all algorithms. Other specific 
    algorithms will inherit all functionality for handling basic federated
    mechanisms. Extensions of this class overrides 5 key methods 
    (i.e. `fit`, `evaluate`, `analyse` and `export`)

    NOTES:
    1)  Any model (i.e. self.global_model or self.local_model[w_id]) passed in 
        has an attribute called `layers`.
        - `layers` is an ordered dict
        - All nn.module layers are set as attributes in the Model object
            eg. 
            
            A model of this structure:
            [
                {
                    "activation": "relu",
                    "is_input": True,
                    "l_type": "Conv2d",
                    "structure": {
                        "in_channels": 1, 
                        "out_channels": 4, # [N, 4, 28, 28]
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1
                    }
                },
                {
                    "activation": None,
                    "is_input": False,
                    "l_type": "Flatten",
                    "structure": {}
                },
                # ------------------------------
                {
                    "activation": "softmax",
                    "is_input": False,
                    "l_type": "Linear",
                    "structure": {
                        "bias": True,
                        "in_features": 4 * 28 * 28,
                        "out_features": 1
                    }
                }

            ]
            will have:
            model.nnl_0_conv2d  --> Conv2d( ... )
            model.nnl_1_flatten --> Flatten()
            model.nnl_2_linear  --> Linear( ... )
        - Structure of `layers`:
            {
                layer_1_name: activation_function_1,
                layer_2_name: activation_function_2,
                ...
            }
        - If a layer is not paired with an activation function, a default
          identity function will be registered.
            {
                layer_1_name: activation_function_1,
                layer_2_name: lambda x: x,              <---
                ...
            }

    """
    
    def __init__(
        self, 
        action: str,
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
            action=action,
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

        ### Existing Attributes (for your reference) ###
        # self.action = action
        # self.crypto_provider = crypto_provider
        # self.workers = workers
        # self.arguments = arguments
        # self.train_loader = train_loader
        # self.eval_loader = eval_loader
        # self.test_loader = test_loader
        # self.global_model = global_model  
        # self.local_models = local_models
        # self.loss_history = {
        #     'global': {
        #         'train': {},
        #         'evaluate': {}
        #     },
        #     'local': {}
        # }
        # self.loop = None
        # self.out_dir = out_dir
        # self.checkpoints = {}


        # General attributes
        self.selected_worker = None

        # Network attributes


        # Data attributes
        self.tensor_dict = None
        self.layer_timings = None
        self.mask = None

        # Model attributes


        # Optimisation attributes


        # Export Attributes



    ###########
    # Helpers #
    ###########

    def parse_layers(self) -> dict:
        """
        Args:
            layers (OrderedDict): Layer attribute of Model
        Returns:
            Name-to-Layer mappings (dict)
        """
        return {
            layer_name: getattr(self.global_model, layer_name)
            for layer_name, _ in self.global_model.layers.items()
        }   


    def initialise_layer_inputs(self, model):
        """
        < Objective - Parse the model to get a list of tensors whose shapes
            correspond to each layer of the specified model.

        Args:

        Returns:
            Tensor dict mapping layer names to the input tensors that 
            correspond to these layers
        """
        name_to_layer_mapping = self.parse_layers()

        # < Insert for loop over here >
        tensor_dict = {}
        for layer_name, layer in name_to_layer_mapping.items():

            # < Given the layer, generate its respective input tensor >            
            input_tensor = None 

            self.tensor_dict[layer_name] = input_tensor

        self.tensor_dict = tensor_dict
        return self.tensor_dict


    def select_optimal_worker(self):
        """ Choose a seeding model that is most generalised out of the entire
            grid

        < For now use random, but need to implement the entropy measure >
        """
        self.selected_worker = random.choice([self.workers])
        return self.selected_worker


    def measure_layer_times(self, worker) -> Dict[str, float]:
        """ 
        < Objective - Obtain measurements for all layers

        How to use:
        selected_worker = random.choice([self.workers])
        layer_timings = self.measure_layer_times(selected_worker)

        Args:
            worker (WebsocketClientWorker)
        """
        name_to_layer_mapping = self.parse_layers()

        layer_timings = OrderedDict()
        for layer_name, input_tensor in self.tensor_dict.items():
            layer = name_to_layer_mapping[layer_name]

            # Send input tensor to specified worker to convert to PointerTensor
            input_tensor = input_tensor.send(worker)

            layer_time = 0
            # < Start timing here >
            layer(input_tensor)
            # < End timing here >

            layer_timings[layer_name] = layer_time            

        self.layer_timings = layer_timings
        return self.layer_timings


    def count_model_parameters(self) -> int:
        """ Counts the total number of parameters in the current global model

        Returns:
            Total parameter count (int)
        """
        name_to_layer_mapping = self.parse_layers()
        return sum([
            th.numel(layer.weight) 
            for _, layer in name_to_layer_mapping
        ])


    def generate_mask(self) -> List[int]:
        """ Generates mask for phase 1 & phase 2 operations

        Returns:
            Mask (list(int))
        """
        num_weight_elements = self.count_model_parameters()
        mask = [1] * num_weight_elements
        return mask


    def perform_pruning(
        self, 
        worker, 
        model,
        layer_timings: Dict[str, float]
    ):
        """
        """
        mask = self.generate_mask()
        # < Insert reconfiguration logic here >
            # < Remember to update the self.mask!
        return self.mask


    def perform_initial_training(
        self,
        selected_worker: WebsocketClientWorker,
        datasets: dict,  
        optimizer: th.optim, 
        scheduler: th.nn, 
        criterion: th.nn, 
        stopper: EarlyStopping, 
        metric: str,
        max_epochs: int = 1000
    ):
        """ Phase 1 dictates initial training to seed the model for better 
            convergence in Phase 2

            {
                "train": {
                    "y_pred": [
                        [0.],
                        [1.],
                        [0.],
                        [1.],
                        .
                        .
                        .
                    ],
                    "y_score": [
                        [0.4561681891162],
                        [0.8616516118919],
                        [0.3218971919191],
                        [0.6919811999489],
                        .
                        .
                        .
                    ]
                },
                "evaluate": {},
                "predict": {}
            }
        """
        for epoch in range(max_epochs):

            # Train global model through the entire batch once
            for batch in datasets:
                for worker, (data, labels) in batch.items():
                    if worker == selected_worker:

                        curr_global_model = curr_global_model.send(worker)
                        curr_local_model = curr_local_model.send(worker)

                        # Zero gradients to prevent accumulation  
                        curr_local_model.train()
                        optimizer.zero_grad() 

                        # Forward Propagation
                        outputs = curr_local_model(data)

                        loss = criterion(
                            outputs=outputs, 
                            labels=labels,
                            w=curr_local_model.state_dict(),
                            wt=curr_global_model.state_dict()
                        )

                        # Backward propagation
                        loss.backward()
                        optimizer.step()

                        curr_global_model = curr_global_model.get()
                        curr_local_model = curr_local_model.get()

            # Use trained initialised global model for inference
            worker_inferences, _ = self.perform_FL_evaluation(
                datasets=datasets,
                workers=[selected_worker],
                is_shared=False
            )

            # Convert collection of object IDs accumulated from minibatch 
            analyser = Analyser(
                auto_align=auto_align, 
                inferences=worker_inferences, 
                metas=metas,
                **combination_keys
            )
            polled_stats = analyser.infer(reg_records=relevant_registrations)

            fl_combination = tuple(combination_keys.keys())
            metric_value = polled_stats[fl_combination][selected_worker][metric] 

            if metric_value > seed_threshold:
                break

        return self.global_model


    def perform_parallel_training(
        self,
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
        WORKERS_STOPPED = []
        gradients = []

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

                # curr_global_model = self.secret_share(curr_global_model)
                # curr_local_model = self.secret_share(curr_local_model)
                curr_global_model = curr_global_model.send(worker)
                curr_local_model = curr_local_model.send(worker)

                # Zero gradients to prevent accumulation  
                curr_local_model.train()
                curr_optimizer.zero_grad() 

                # Forward Propagation
                outputs = curr_local_model(data)

                loss = curr_criterion(
                    outputs=outputs, 
                    labels=labels,
                    w=curr_local_model.state_dict(),
                    wt=curr_global_model.state_dict()
                )

                # Backward propagation
                loss.backward()
                curr_optimizer.step()

                curr_global_model = curr_global_model.get()
                curr_local_model = curr_local_model.get()

                # < Insert gradient extraction & accumulations here >

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
                worker (WebsocketClientWorker): Worker to be evaluated
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
                await train_batch(batch)
            
            stagnation_futures = [
                check_for_stagnation(worker) 
                for worker in self.workers
            ]
            await asyncio.gather(*stagnation_futures)


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
                grid_checkpoint = self.export(
                    out_dir=checkpoint_dir,
                    excluded=['checkpoint']
                )
                for _, logs in grid_checkpoint.items():
                    origin = logs.pop('origin')

                    # Note: Structure - {worker: {round: {epoch: {...}}}}
                    worker_archive = self.checkpoints.get(origin, {}) 
                    round_archive = worker_archive.get(round_key, {}) 
                    round_archive.update({epoch_key: logs})           
                    worker_archive.update({round_key: round_archive})
                    self.checkpoints.update({origin: worker_archive})

            # Insert here for round
            self.perform_pruning(
                worker=self.selected_worker, 
                model=self.global_model,
                layer_timing=self.layer_timings
            )

        finally:
            loop.close()

        return models, optimizers, schedulers, criterions, stoppers


    def finalize_mask(self):
        """ Reconstruct pruned global model according to the finalised mask

        < Modifies the global model inplace >
        """
        pass

    ##################
    # Core functions #
    ##################

    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
            
        Returns:
            Trained global model (Model)
        """
        # < Insert operations for Phase 0 - Timings >
        self.select_optimal_worker()
        layer_timings = self.measure_layer_times(self.selected_worker)

        # < Insert operations for Phase 1 - Initialised Training >
        optimizer = self.arguments.optimizer( 
            **self.arguments.optimizer_params,
            params=self.global_model.parameters()
        )
        scheduler = self.arguments.lr_scheduler(
            **self.arguments.lr_decay_params,
            optimizer=optimizer
        )
        criterion = self.build_custom_criterion()(
            **self.arguments.criterion_params
        )
        stopper = EarlyStopping(**self.arguments.early_stopping_params)
        self.perform_initial_training(
            selected_worker=self.selected_worker,
            datasets=self.train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            stopper=stopper,
            metric=metric,
            max_epochs=max_epochs
        )
        self.perform_pruning(
            worker=self.selected_worker, 
            model=self.global_model,
            layer_timings=layer_timings
        )

        # Phase 2 Operation: Federated masked averaging
        super().fit()
        self.finalize_mask()

        return self.global_model, self.local_models


    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        raise NotImplementedError
