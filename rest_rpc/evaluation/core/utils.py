#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Libs
import aiohttp
import mlflow
import torch as th

# Custom
from rest_rpc import app
from rest_rpc.training.core.utils import (
    UrlConstructor, 
    RPCFormatter, 
    Orchestrator
)
from synarchive.connection import ExperimentRecords, RunRecords
from synarchive.training import ModelRecords
from synarchive.evaluation import MLFRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

db_path = app.config['DB_PATH']
mlflow_dir = app.config['MLFLOW_DIR']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/core/utils.py logged", Description="No Changes")

####################
# Helper Functions #
####################

def replicate_combination_key(expt_id, run_id):
    return str((expt_id, run_id))

############################################
# Inference Orchestration class - Analyser #
############################################

class Analyser(Orchestrator):
    """ 
    Takes in a list of minibatch IDs and sends them to worker nodes. Workers
    will use these IDs to reconstruct their aggregated test datasets with 
    prediction labels mapped appropriately.

    Attributes:
        inferences (dict(str, dict(str, dict(str, th.tensor)))
    """
    def __init__(
        self, 
        collab_id: str,
        project_id: str,
        expt_id: str, 
        run_id: str,
        inferences: dict,
        metas: list = ['train', 'evaluate', 'predict'],
        auto_align: bool = True
    ):
        super().__init__()

        self.metas = metas
        self.collab_id = collab_id
        self.project_id = project_id
        self.expt_id = expt_id
        self.run_id = run_id
        self.inferences = inferences
        self.auto_align = auto_align

    ###########
    # Helpers #
    ###########

    async def _poll_for_stats(
        self, 
        node_info: Dict[str, Any], 
        inferences: Dict[str, Dict[str, th.Tensor]]
    ):
        """ Parses a registration record for participant metadata, before
            submitting minibatch IDs of inference objects to corresponding 
            worker node's REST-RPC service for calculating descriptive
            statistics and prediction exports

        Args:
            reg_record (tinydb.database.Document): Participant-project details
            inferences (dict): List of dicts containing inference object IDs
        Returns:
            Statistics (dict)
        """
        _, _, participant_id = self.parse_keys(node_info)
 
        if not inferences:
            return {participant_id: {meta:{} for meta in self.metas}}

        # Construct destination url for interfacing with worker REST-RPC
        rest_connection = self.parse_rest_info(node_info)
        destination_constructor = UrlConstructor(**rest_connection)
        destination_url = destination_constructor.construct_predict_url(
            collab_id=self.collab_id,
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        ml_action = self.parse_action(node_info)
        data_tags = self.parse_tags(node_info)
        data_alignments = self.parse_alignments(node_info, self.auto_align)

        payload = {
            'action': ml_action, 
            'tags': data_tags,
            'alignments': data_alignments,
            'inferences': inferences
        }

        # Trigger remote inference by posting alignments & ID mappings to 
        # `Predict` route in worker
        resp_inference_data, _ = await self.instruct(
            command='post', 
            url=destination_url, 
            payload=payload
        )
        
        logging.debug(
            f"Participant '{participant_id}' >|< Project '{self.project_id}' -> Experiment '{self.expt_id}' -> Run '{self.run_id}': Polled statistics tracked.",
            description=f"Polled statistics for participant '{participant_id}' under project '{self.project_id}' using experiment '{self.expt_id}' and run '{self.run_id}' tracked.",
            resp_json=resp_inference_data,
            ID_path=SOURCE_FILE,
            ID_class=Analyser.__name__,
            ID_function=Analyser._poll_for_stats.__name__
        )

        # Extract the relevant expt-run results
        expt_run_key = replicate_combination_key(self.expt_id, self.run_id)
        metadata = resp_inference_data['results'][expt_run_key]

        # Filter by selected meta-datasets
        filtered_statistics = {
            meta: stats 
            for meta, stats in metadata.items()
            if meta in self.metas
        }

        return {participant_id: filtered_statistics}


    async def _collect_all_stats(self, grid: List[Dict[str, Any]]) -> dict:
        """ Asynchronous function to submit inference data to registered
            participant servers in return for remote performance statistics

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' statistics (dict)
        """
        sorted_node_info = sorted(
            grid, 
            key=lambda x: self.parse_syft_info(x).get('id')
        )
        sorted_inferences = sorted(self.inferences.items(), key=lambda x: x[0])

        logging.debug(
            "Sorted inferences tracked.",
            sorted_inferences=sorted_inferences,
            ID_path=SOURCE_FILE,
            ID_class=Analyser.__name__,
            ID_function=Analyser._collect_all_stats.__name__
        )

        mapped_pairs = [
            (record, inferences) 
            for record, (_, inferences) in zip(
                sorted_node_info, 
                sorted_inferences
            )
        ]

        all_statistics = {}
        for future in asyncio.as_completed(
            map(lambda args: self._poll_for_stats(*args), mapped_pairs)
        ):
            result = await future
            all_statistics.update(result)

        return all_statistics

    ##################
    # Core Functions #
    ##################

    def infer(self, grid: List[Dict[str, Any]]) -> dict:
        """ Wrapper function for triggering asychroneous remote inferencing of
            participant nodes

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' statistics (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_stats = loop.run_until_complete(
                self._collect_all_stats(grid=grid)
            )
        finally:
            loop.close()

        return all_stats



####################################
# MLFLow logging class - MLFlogger #
####################################

class MLFlogger:
    """ 
    Wrapper class around MLFlow to faciliate experiment & run registrations in
    the REST-RPC setting, where statistical logging is performed during 
    post-mortem analysis
    """
    def __init__(self, db_path: str = db_path):

        # Private attributes
        self.__rpc_formatter = RPCFormatter()
        self.__expt_records = ExperimentRecords(db_path=db_path)
        self.__run_records = RunRecords(db_path=db_path)
        self.__model_records = ModelRecords(db_path=db_path)
        
        # Public attributes
        self.mlf_records = MLFRecords(db_path=db_path)

    ###########
    # Helpers #
    ###########

    def initialise_mlflow_project(
        self, 
        collab_id: str, 
        project_id: str
    ) -> str:
        """ In MLFlow, there is no concept of collaboration-level 
            stratification. While they have MLFlow Projects, using this 
            functionality will come at the cost since this conflicts with 
            REST-RPC's job orchestration. As such, collaboration & project 
            intialisation is done natively, by creating custom project URIs, 
            and switching to them when necessary.

        Args:
            project_id (str): REST-RPC ID of specified project
        Returns:
            Project-specific URI (str)
        """
        project_uri = os.path.join(mlflow_dir, collab_id, project_id)
        Path(project_uri).mkdir(parents=True, exist_ok=True)
        return project_uri


    def delete_mlflow_project(        
        self, 
        collab_id: str, 
        project_id: str
    ) -> str:
        """ Conceptually remove all MLFlow logs made under a specific project.

        Args:
            project_id (str): REST-RPC ID of specified project
        Returns:
            Removed Project-specific URI (str)
        """
        # Remove MLFlow directory corresponding to project's URI
        project_uri = os.path.join(mlflow_dir, collab_id, project_id)
        shutil.rmtree(project_uri)
        return project_uri


    def initialise_mlflow_experiment(
        self, 
        collab_id: str, 
        project_id: str,
        expt_id: str
    ) -> Dict[str, str]:
        """ Initialises an MLFlow experiment at the specified project URI

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
        Returns:
            MLFlow Experiment configuration (dict)
        """
        project_uri = self.initialise_mlflow_project(
            collab_id=collab_id,
            project_id=project_id
        )
        mlflow.set_tracking_uri(project_uri)

        # Check if MLFlow experiment has already been created
        mlflow_details = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            name=expt_id
        )
        if not mlflow_details:
            
            # Initialise MLFlow experiment
            mlflow_id = mlflow.create_experiment(name=expt_id)

            mlflow_details = {
                'collaboration': collab_id,
                'project': project_id,
                'name': expt_id,
                'mlflow_type': 'experiment',
                'mlflow_id': mlflow_id,
                'mlflow_uri': project_uri
            }
            self.mlf_records.create(
                collaboration=collab_id,
                project=project_id,
                name=expt_id, 
                details=mlflow_details
            )

        return mlflow_details


    def delete_mlflow_experiment(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str,
    ) -> str:
        """ Conceptually removes all MLFlow logs made under a specific 
            experiment.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
        Returns:
            Removed MLFlow experiment directory 
        """
        # Delete the details themselves
        deleted_details = self.mlf_records.delete(
            collaboration=collab_id,
            project=project_id, 
            name=expt_id
        )

        # Remove experiment's MLFlow directory
        expt_mlflow_dir = os.path.join(
            deleted_details['mlflow_uri'], 
            deleted_details['mlflow_id']
        )
        shutil.rmtree(expt_mlflow_dir)

        stripped_details = self.__rpc_formatter.strip_keys(
            record=deleted_details, 
            concise=True
        )
        return stripped_details


    def initialise_mlflow_run(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> Dict[str, str]:
        """ Initialises a MLFLow run under a specified experiment of a project.
            Initial run hyperparameters will be logged, and MLFlow run id will
            be stored for subsequent analysis.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
        """
        # Initialise the parent MLFlow experiment
        expt_mlflow_details = self.initialise_mlflow_experiment(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_name=run_id
        ) as mlf_run:

            # Retrieve run details from database
            run_details = self.__run_records.read(
                collab_id=collab_id, 
                project_id=project_id, 
                expt_id=expt_id, 
                run_id=run_id
            )
            stripped_run_details = self.__rpc_formatter.strip_keys(
                record=run_details,
                concise=True
            )

            mlflow.log_params(stripped_run_details)

            # Save the MLFlow ID mapping
            run_mlflow_id = mlf_run.info.run_id
            run_mlflow_details = {
                'collaboration': collab_id,
                'project': project_id,
                'name': run_id,
                'mlflow_type': 'run',
                'mlflow_id': run_mlflow_id,
                'mlflow_uri': expt_mlflow_details['mlflow_uri'] # same as expt
            }
            new_run_mlflow_details = self.mlf_records.create(
                collaboration=collab_id,
                project=project_id,
                name=run_id, 
                details=run_mlflow_details
            )

        stripped_run_mlflow_details = self.__rpc_formatter.strip_keys(
            record=new_run_mlflow_details
        )
        return stripped_run_mlflow_details


    def delete_mlflow_run(self):
        """

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
        Returns:

        """
        pass



    def log_losses(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str, 
        run_id: str
    ):
        """ Registers all cached losses, be it global or local, obtained from
            federated training into MLFlow.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
        Returns:
            Stripped metadata (dict)
        """
        # Initialise the parent MLFlow experiment
        expt_mlflow_details = self.initialise_mlflow_experiment(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        # Search for run session to update entry, not create a new one
        run_mlflow_details = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            name=run_id
        )

        if not run_mlflow_details:
            logging.error(
                "MLFlow run has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow run has not been initialised!")

        # Retrieve all model metadata from storage
        model_metadata = self.__model_records.read(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id, 
            run_id=run_id
        )
        stripped_metadata = self.__rpc_formatter.strip_keys(
            record=model_metadata, 
            concise=True
        ) if model_metadata else model_metadata

        if stripped_metadata:

            with mlflow.start_run(
                experiment_id=expt_mlflow_id, 
                run_id=run_mlflow_details['mlflow_id']
            ) as mlf_run:

                # Extract loss histories
                for m_type, metadata in stripped_metadata.items():

                    loss_history_path = metadata['loss_history']
                    origin = metadata['origin']

                    with open(loss_history_path, 'r') as lh:
                        loss_history = json.load(lh)

                    if m_type == 'global':
                        for meta, losses in loss_history.items():
                            for round_idx, loss, in losses.items():
                                mlflow.log_metric(
                                    key=f"global_{meta}_loss", 
                                    value=loss, 
                                    step=int(round_idx)
                                )

                    else:
                        for round_idx, loss, in losses.items():
                            mlflow.log_metric(
                                key=f"{origin}_local_loss", 
                                value=loss, 
                                step=int(round_idx)
                            )

        return stripped_metadata

    
    def log_model_performance(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str,
        statistics: dict
    ) -> dict:
        """ Using all cached model checkpoints, log the performance statistics 
            of models, be it global or local, to MLFlow at round (for global) or
            epoch (for local) level.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
            statistics (dict): Inference statistics polled from workers
        Returns:
            MLFLow run details (dict)
        """
        # Initialise the parent MLFlow experiment
        expt_mlflow_details = self.initialise_mlflow_experiment(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        # Search for run session to update entry, not create a new one
        run_mlflow_details = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            name=run_id
        )
        run_mlflow_id = run_mlflow_details['mlflow_id']

        if not run_mlflow_details:
            logging.error(
                "Run has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("Run has not been initialised!")

        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_id=run_mlflow_id
        ) as mlf_run:

            # Store output metadata into database
            for _, inference_stats in statistics.items():
                for _, meta_stats in inference_stats.items():

                    # Log statistics to MLFlow for analysis
                    stats = meta_stats.get('statistics', {})
                    for stat_type, stat_value in stats.items():

                        if isinstance(stat_value, list):
                            for val_idx, value in enumerate(stat_value):
                                mlflow.log_metric(
                                    key=f"{stat_type}_class_{val_idx}", 
                                    value=value, 
                                    step=int(val_idx+1)
                                )

                        else:
                            mlflow.log_metric(key=stat_type, value=stat_value)

        stripped_mlflow_run_details = self.__rpc_formatter.strip_keys(
            run_mlflow_details, 
            concise=True
        )
        return stripped_mlflow_run_details

    ##################
    # Core Functions #
    ##################

    def log(self, accumulations: dict) -> List[str]:
        """ Wrapper function that processes statistics accumulated from 
            inference.

        Args:
            accumulations (dict): Accumulated statistics from inferring
                different project-expt-run combinations
        Returns:
            List of MLFlow run IDs from all runs executed (list(str))
        """
        jobs_ran = []
        for combination_key, statistics in accumulations.items():

            collab_id, project_id, expt_id, run_id = combination_key

            run_mlflow_details = self.initialise_mlflow_run(
                collab_id=collab_id,
                project_id=project_id,
                expt_id=expt_id,
                run_id=run_id
            )
            self.log_losses(
                collab_id=collab_id,
                project_id=project_id,
                expt_id=expt_id,
                run_id=run_id
            )
            self.log_model_performance(
                collab_id=collab_id,
                project_id=project_id,
                expt_id=expt_id,
                run_id=run_id,
                statistics=statistics
            )
            jobs_ran.append(run_mlflow_details['mlflow_id'])
        
        return jobs_ran