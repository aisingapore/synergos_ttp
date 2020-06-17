#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import logging
from datetime import datetime

# Libs
import aiohttp
import jsonschema

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload, AssociationRecords
from rest_rpc.connection.core.datetime_serialization import DateTimeSerializer
from rest_rpc.training.core.utils import UrlConstructor, RPCFormatter

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

"""
These are the subject-id-class mappings for the main utility records in 
Prediction:
{
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
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["validation_schema"])
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
        
        payload = {'inferences': inferences}

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