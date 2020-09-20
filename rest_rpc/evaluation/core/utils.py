#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

# Libs
import aiohttp
import jsonschema
import mlflow

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalRecords, 
    AssociationRecords,
    ExperimentRecords,
    RunRecords
)
from rest_rpc.training.core.utils import (
    UrlConstructor, 
    RPCFormatter,
    ModelRecords
)

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
mlflow_dir = app.config['MLFLOW_DIR']

"""
These are the subject-id-class mappings for the main utility records in 
Prediction:
{
    'MLFlow': {
        'id': 'name',
        'class': MLFRecords
    },
    'Validation': {
        'id': 'val_id',
        'class': ValidationRecords
    },
    'Prediction': {
        'id': 'pred_id',
        'class': PredictionRecords
    }
}
"""

####################
# Helper Functions #
####################

def replicate_combination_key(expt_id, run_id):
    return str((expt_id, run_id))

#########################################
# MLFlow Key storage class - MLFRecords #
#########################################

class MLFRecords(TopicalRecords):
    """ 
    This class solely exists as a persistent storage of `experiment_id/run_id`
    mappings to MLFlow generated experiement IDs & run IDs respectively. This is
    due to the fact that each unique experiment/run name can only be assigned a
    single MLFlow ID. Any attempt to re-initialise a new experiment/run will not
    override the existing registries, raising `mlflow.exceptions.MlflowException`
    """
    def __init__(self, db_path=db_path):
        super().__init__(
            subject="MLFlow",  
            identifier="name", 
            db_path=db_path
        )

    def __generate_key(self, project, name):
        return {"project": project, "name": name}

    def create(self, project, name, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["mlflow_schema"])
        mlf_key = self.__generate_key(project, name)
        new_entry = {'key': mlf_key}
        new_entry.update(details)
        return super().create(new_entry)

    def read(self, project, name):
        mlf_key = self.__generate_key(project, name)
        return super().read(mlf_key)

    def update(self, project, name, updates):
        mlf_key = self.__generate_key(project, name)
        return super().update(mlf_key, updates)

    def delete(self, project, name):
        mlf_key = self.__generate_key(project, name)
        return super().delete(mlf_key)

######################################################
# Data Storage Association class - ValidationRecords #
######################################################

class ValidationRecords(AssociationRecords):
    """ This class catalogues exported changes of both the global & local models
        as federated training is in progress. Unlike PredictionRecords, this
        table does not record statistics, but rather tracked values to be fed
        MLFlow for visualisations, as well as incentive calculation.
    """

    def __init__(self, db_path=db_path):
        super().__init__(
            "Validation",  
            "val_id", 
            db_path,
            [],
            *["Model"]
        )

    def __generate_key(self, participant_id, project_id, expt_id, run_id):
        return {
            "participant_id": participant_id,
            "project_id": project_id,
            "expt_id": expt_id,
            "run_id": run_id
        }

    def create(self, participant_id, project_id, expt_id, run_id, details):
        logging.debug(f"Details: {details}")

        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["prediction_schema"])#schemas["validation_schema"])
        validation_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        new_validation = {'key': validation_key}
        new_validation.update(details)
        return super().create(new_validation)

    def read(self, participant_id, project_id, expt_id, run_id):
        validation_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().read(validation_key)

    def update(self, participant_id, project_id, expt_id, run_id, updates):
        validation_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().update(validation_key, updates)

    def delete(self, participant_id, project_id, expt_id, run_id):
        validation_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().delete(validation_key)

######################################################
# Data Storage Association class - PredictionRecords #
######################################################

class PredictionRecords(AssociationRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Prediction",  
            "pred_id", 
            db_path,
            [],
            *["Model", "Registration", "Tag"]
        )

    def __generate_key(self, participant_id, project_id, expt_id, run_id):
        return {
            "participant_id": participant_id,
            "project_id": project_id,
            "expt_id": expt_id,
            "run_id": run_id
        }

    def create(self, participant_id, project_id, expt_id, run_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["prediction_schema"])
        prediction_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        new_prediction = {'key': prediction_key}
        new_prediction.update(details)
        return super().create(new_prediction)

    def read(self, participant_id, project_id, expt_id, run_id):
        prediction_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().read(prediction_key)

    def update(self, participant_id, project_id, expt_id, run_id, updates):
        prediction_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().update(prediction_key, updates)

    def delete(self, participant_id, project_id, expt_id, run_id):
        prediction_key = self.__generate_key(
            participant_id, 
            project_id, 
            expt_id, 
            run_id
        )
        return super().delete(prediction_key)

############################################
# Inference Orchestration class - Analyser #
############################################

class Analyser:
    """ 
    Takes in a list of minibatch IDs and sends them to worker nodes. Workers
    will use these IDs to reconstruct their aggregated test datasets with 
    prediction labels mapped appropriately.

    Attributes:
        inferences (dict(str, dict(str, dict(str, th.tensor)))
    """
    def __init__(
        self, 
        project_id: str,
        expt_id: str, 
        run_id: str,
        inferences: dict,
        metas: list = ['train', 'evaluate', 'predict']
    ):
        self.__rpc_formatter = RPCFormatter()
        self.metas = metas
        self.project_id = project_id
        self.expt_id = expt_id
        self.run_id = run_id
        self.inferences = inferences

    ###########
    # Helpers #
    ###########

    async def _poll_for_stats(self, reg_record, inferences):
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
        project_action = reg_record['project']['action'] 
        
        participant_details = reg_record['participant'].copy()
        participant_id = participant_details['id']
        participant_ip = participant_details['host']
        participant_f_port = participant_details.pop('f_port') # Flask port

        if not inferences:
            return {participant_id: {meta:{} for meta in self.metas}}

        # Construct destination url for interfacing with worker REST-RPC
        destination_constructor = UrlConstructor(
            host=participant_ip,
            port=participant_f_port
        )
        destination_url = destination_constructor.construct_predict_url(
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )
        
        payload = {'action': project_action, 'inferences': inferences}

        # Trigger remote inference by posting alignments & ID mappings to 
        # `Predict` route in worker
        async with aiohttp.ClientSession() as session:
            async with session.post(
                destination_url,
                json=payload
            ) as response:
                resp_json = await response.json(content_type='application/json')
        
        # Extract the relevant expt-run results
        expt_run_key = replicate_combination_key(self.expt_id, self.run_id)
        metadata = resp_json['data']['results'][expt_run_key]

        # Filter by selected meta-datasets
        filtered_statistics = {
            meta: stats 
            for meta, stats in metadata.items()
            if meta in self.metas
        }

        return {participant_id: filtered_statistics}

    async def _collect_all_stats(self, reg_records):
        """ Asynchronous function to submit inference data to registered
            participant servers in return for remote performance statistics

        Args:
            reg_records (list(tinydb.database.Document))): Participant Registry
        Returns:
            All participants' statistics (dict)
        """
        sorted_reg_records = sorted(
            reg_records, 
            key=lambda x: x['participant']['id']
        )
        sorted_inferences = sorted(
            self.inferences.items(), 
            key=lambda x: x[0]
        )

        logging.debug(f"Sorted inferences: {sorted_inferences}")

        mapped_pairs = [
            (record, inferences) 
            for record, (_, inferences) in zip(
                sorted_reg_records, 
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

    def infer(self, reg_records):
        """ Wrapper function for triggering asychroneous remote inferencing of
            participant nodes

        Args:
            reg_records (list(tinydb.database.Document))): Participant Registry
        Returns:
            All participants' statistics (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_stats = loop.run_until_complete(
                self._collect_all_stats(reg_records)
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

    def initialise_mlflow_project(self, project_id: str) -> str:
        """ In MLFlow, there is no concept of project-level stratification.
            While they have MLFlow Projects, using this functionality will come
            at the cost since this conflicts with REST-RPC's job orchestration.
            As such, project intialisation is done natively, by creating project
            URIs, and switching to them when necessary.

        Args:
            project_id (str): REST-RPC ID of specified project
        Returns:
            Project-specific URI (str)
        """
        project_uri = os.path.join(mlflow_dir, project_id)
        Path(project_uri).mkdir(parents=True, exist_ok=True)
        return project_uri


    def delete_mlflow_project(self, project_id: str) -> str:
        """ Conceptually remove all MLFlow logs made under a specific project.

        Args:
            project_id (str): REST-RPC ID of specified project
        Returns:
            Removed Project-specific URI (str)
        """
        # Remove MLFlow directory corresponding to project's URI
        project_uri = os.path.join(mlflow_dir, project_id)
        shutil.rmtree(project_uri)
        return project_uri


    def initialise_mlflow_experiment(
        self, 
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
        project_uri = self.initialise_mlflow_project(project_id=project_id)
        mlflow.set_tracking_uri(project_uri)

        # Check if MLFlow experiment has already been created
        mlflow_details = self.mlf_records.read(
            project=project_id, 
            name=expt_id
        )
        if not mlflow_details:
            
            # Initialise MLFlow experiment
            mlflow_id = mlflow.create_experiment(name=expt_id)

            mlflow_details = {
                'project': project_id,
                'name': expt_id,
                'mlflow_type': 'experiment',
                'mlflow_id': mlflow_id,
                'mlflow_uri': project_uri
            }
            self.mlf_records.create(
                project=project_id,
                name=expt_id, 
                details=mlflow_details
            )

        return mlflow_details


    def delete_mlflow_experiment(
        self, 
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
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_name=run_id
        ) as mlf_run:

            # Retrieve run details from database
            run_details = self.__run_records.read(project_id, expt_id, run_id)
            stripped_run_details = self.__rpc_formatter.strip_keys(
                record=run_details,
                concise=True
            )

            mlflow.log_params(stripped_run_details)

            # Save the MLFlow ID mapping
            run_mlflow_id = mlf_run.info.run_id
            run_mlflow_details = {
                'project': project_id,
                'name': run_id,
                'mlflow_type': 'run',
                'mlflow_id': run_mlflow_id,
                'mlflow_uri': expt_mlflow_details['mlflow_uri'] # same as expt
            }
            new_run_mlflow_details = self.mlf_records.create(
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



    def log_losses(self, project_id: str, expt_id: str, run_id: str):
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
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        # Search for run session to update entry, not create a new one
        run_mlflow_details = self.mlf_records.read(
            project=project_id, 
            name=run_id
        )

        if not run_mlflow_details:
            raise RuntimeError("Run has not been initialised!")

        # Retrieve all model metadata from storage
        model_metadata = self.__model_records.read(project_id, expt_id, run_id)
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
            project_id=project_id,
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_details['mlflow_id']

        # Search for run session to update entry, not create a new one
        run_mlflow_details = self.mlf_records.read(
            project=project_id, 
            name=run_id
        )
        run_mlflow_id = run_mlflow_details['mlflow_id']

        if not run_mlflow_details:
            raise RuntimeError("Run has not been initialised!")

        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_id=run_mlflow_id
        ) as mlf_run:

            # Store output metadata into database
            for participant_id, inference_stats in statistics.items():
                for meta, meta_stats in inference_stats.items():

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

        return self.__rpc_formatter.strip_keys(run_mlflow_details, concise=True)

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

            curr_project_id = combination_key[0]
            curr_expt_id = combination_key[1]
            curr_run_id = combination_key[2]

            run_mlflow_details = self.initialise_mlflow_run(
                project_id=curr_project_id,
                expt_id=curr_expt_id,
                run_id=curr_run_id
            )
            self.log_losses(
                project_id=curr_project_id,
                expt_id=curr_expt_id,
                run_id=curr_run_id
            )
            self.log_model_performance(
                project_id=curr_project_id,
                expt_id=curr_expt_id,
                run_id=curr_run_id,
                statistics=statistics
            )
            jobs_ran.append(run_mlflow_details['mlflow_id'])
        
        return jobs_ran