#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import logging
import uuid
from datetime import datetime

# Libs
import aiohttp
import jsonschema

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload, AssociationRecords
from rest_rpc.connection.core.datetime_serialization import DateTimeSerializer

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

worker_poll_route = app.config['WORKER_ROUTES']['poll']
worker_align_route = app.config['WORKER_ROUTES']['align']
worker_initialise_route = app.config['WORKER_ROUTES']['initialise']
worker_terminate_route = app.config['WORKER_ROUTES']['terminate']
worker_predict_route = app.config['WORKER_ROUTES']['predict']

"""
These are the subject-id-class mappings for the main utility records in 
training:
{
    'Alignment': {
        'id': 'alignment_id',
        'class': AlignmentRecords
    },
    'Model': {
        'id': 'model_id',
        'class': ModelRecords
    }
}
"""
#################################
# Helper Class - UrlConstructor #
#################################

class UrlConstructor:
    def __init__(self, host, port, secure=False):
        self.host = host
        self.port = port
        self.secure = secure

    ###########
    # Helpers #
    ###########

    def construct_url(self, route):
        protocol = "http" if not self.secure else "https"
        base_url = f"{protocol}://{self.host}:{self.port}"
        destination_url = base_url + route
        return destination_url

    ##################
    # Core Functions #
    ##################

    def construct_poll_url(self, project_id):
        destination_url = self.construct_url(route=worker_poll_route)
        custom_poll_url = destination_url.replace("<project_id>", project_id)
        return custom_poll_url
        
    def construct_align_url(self, project_id):
        destination_url = self.construct_url(route=worker_align_route)
        custom_align_url = destination_url.replace("<project_id>", project_id)
        return custom_align_url

    def construct_initialise_url(self, project_id, expt_id, run_id):
        destination_url = self.construct_url(route=worker_initialise_route)
        custom_intialise_url = destination_url.replace(
            "<project_id>", project_id
        ).replace(
            "<expt_id>", expt_id
        ).replace(
            "<run_id", run_id
        )
        return custom_intialise_url

    def construct_terminate_url(self, project_id, expt_id, run_id):
        destination_url = self.construct_url(route=worker_terminate_route)
        custom_terminate_url = destination_url.replace(
            "<project_id>", project_id
        ).replace(
            "<expt_id>", expt_id
        ).replace(
            "<run_id", run_id
        )
        return custom_terminate_url

#####################################################
# Data Storage Association class - AlignmentRecords #
#####################################################

class AlignmentRecords(AssociationRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Alignment",  
            "alignment_id", 
            db_path,
            [],
            *["Registration", "Tag"]
        )

    def __generate_key(self, project_id, participant_id):
        return {
            "project_id": project_id, 
            "participant_id": participant_id
        }

    def create(self, project_id, participant_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["alignment_schema"])
        alignment_key = self.__generate_key(project_id, participant_id)
        new_alignment = {'key': alignment_key}
        new_alignment.update(details)
        return super().create(new_alignment)

    def read(self, project_id, participant_id):
        alignment_key = self.__generate_key(project_id, participant_id)
        return super().read(alignment_key)

    def update(self, project_id, participant_id, updates):
        alignment_key = self.__generate_key(project_id, participant_id)
        return super().update(alignment_key, updates)

    def delete(self, project_id, participant_id):
        alignment_key = self.__generate_key(project_id, participant_id)
        return super().delete(alignment_key)

#################################################
# Data Storage Association class - ModelRecords #
#################################################

class ModelRecords(AssociationRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            subject="Model",  
            identifier="model_id", 
            db_path=db_path,
            relations=["Project", "Experiment", "Run"]
        )

    def __generate_key(self, project_id, expt_id, run_id):
        return {
            "project_id": project_id,
            "expt_id": expt_id,
            "run_id": run_id
        }

    def create(self, project_id, expt_id, run_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["model_schema"])
        model_key = self.__generate_key(project_id, expt_id, run_id)
        new_model = {'key': model_key}
        new_model.update(details)
        return super().create(new_model)

    def read(self, project_id, expt_id, run_id):
        model_key = self.__generate_key(project_id, expt_id, run_id)
        return super().read(model_key)

    def update(self, project_id, expt_id, run_id, updates):
        model_key = self.__generate_key(project_id, expt_id, run_id)
        return super().update(model_key, updates)

    def delete(self, project_id, expt_id, run_id):
        model_key = self.__generate_key(project_id, expt_id, run_id)
        return super().delete(model_key)
           
#################################################
# Data Storage Association class - StatsRecords #
#################################################

class StatsRecords(AssociationRecords):
    def __init__(self, subject, identifier, db_path=db_path, relations=[], *associations):
        super().__init__(subject, identifier, db_path=db_path, relations=relations, *associations)

##########################################
# Data Augmentation class - RPCFormatter #
##########################################

class RPCFormatter:
    """
    Automates the formatting of structured data stored within the TTP into a
    form compatible with the interfaces defined by the worker node's REST-RPC 
    service.
    """
    def strip_keys(self, record, concise=False):
        """ Remove db-specific keys and descriptors

        Args:
            record (tinydb.database.Document): Target record to strip
        Returns:
            Stripped record (tinydb.database.Document)
        """
        record.pop('key')
        record.pop('created_at')
        try:
            record.pop('link')
        except KeyError:
            pass
        if concise:
            record.pop('relations')
        return record

    def aggregate_metadata(self, all_metadata):
        """ Takes a series of metadata and aggregates them in preparation for
            multiple feature alignment

        Args:
            all_metadata (dict(str, tinydb.database.Document)): 
                All retrieved metadata polled from registered participants
        Returns:
            X_data_headers  (list(list(str)))
            y_data_headers  (list(list(str)))
            key_sequences   (list(tuple(str)))
            super_schema    (dict)
        """
        super_schema = {}
        key_sequences = []
        X_data_headers = []
        y_data_headers = []
        for participant_id, metadata in all_metadata.items():

            headers = metadata['headers']
            for meta, data_headers in headers.items():
                if not data_headers:
                    continue
                X_headers = data_headers['X']
                X_data_headers.append(X_headers)
                y_headers = data_headers['y']
                y_data_headers.append(y_headers)
                key_sequences.append((participant_id, meta))

            schemas = metadata['schemas']
            for meta, schema in schemas.items():
                if not schema:
                    continue
                super_schema.update(schema)

        return X_data_headers, y_data_headers, key_sequences, super_schema

    def alignment_to_spacer_idxs(self, X_mf_alignments, y_mf_alignments, key_sequences):
        """ Aggregates feature and target alignments and formats them w.r.t each
            corresponding participant

        Args:
            X_mf_alignments (tuple(list(str))): MFA sequence for features
            y_mf_alignments (tuple(list(str))): MFA sequence for targets
            key_sequence (list(tuple(str))):
                eg. 
                    [
                        ('test_worker_2', 'train'), 
                        ('test_worker_1', 'train'), 
                        ('test_worker_1', 'evaluate')
                    ]
        Returns:
            Spacer collection (dict)
        """
        
        def extract_null_idxs(alignment):
            """
            Args:
                alignment (list(str)): Aligned list of headers from MFA
            Returns:
                Null indexes (list(int))
            """
            return [idx for idx, e in enumerate(alignment) if e == None]

        spacer_collection = {}
        for X_alignment, y_alignment, (participant, meta) in zip(
                X_mf_alignments, 
                y_mf_alignments, 
                key_sequences
            ):

            X_null_idxs = extract_null_idxs(X_alignment)
            y_null_idxs = extract_null_idxs(y_alignment)
            spacer_idxs = {'X': X_null_idxs, 'y': y_null_idxs}
            
            if participant in spacer_collection.keys():
                spacer_collection[participant][meta] = spacer_idxs
            else:
                new_participant_spacers = {participant: {meta: spacer_idxs}}
                spacer_collection.update(new_participant_spacers)
        
        return spacer_collection

#####################################
# Data Orchestration class - Poller #
#####################################

class Poller:
    """
    Polls for headers & schemas, before performing multiple feature alignment,
    to obtain aligned headers alongside a super schema for subsequent reference
    """
    def __init__(self, project_id):
        self.project_id = project_id
        self.__rpc_formatter = RPCFormatter()

    ###########
    # Helpers #
    ###########

    async def _poll_metadata(self, reg_record):
        """ Parses a registration record for participant metadata, before
            polling for data-specific descriptors from corresponding worker 
            node's REST-RPC service

        Args:
            reg_record (tinydb.database.Document)
        Returns:
            headers (dict)
        """
        participant_details = reg_record['participant']
        participant_id = participant_details['id']
        participant_ip = participant_details['host']
        participant_f_port = participant_details.pop('f_port') # Flask port

        # Construct destination url for interfacing with worker REST-RPC
        destination_constructor = UrlConstructor(
            host=participant_ip,
            port=participant_f_port
        )
        destination_url = destination_constructor.construct_poll_url(
            project_id=self.project_id
        )

        # Search for tags using composite project + participant
        relevant_tags = reg_record['relations']['Tag'][0]
        self.__rpc_formatter.strip_keys(relevant_tags)
        payload = {'tags': relevant_tags}

        # Poll for headers by posting tags to `Poll` route in worker
        async with aiohttp.ClientSession() as session:
            async with session.post(
                destination_url,
                json=payload
            ) as response:
                resp_json = await response.json(content_type='application/json')
        
        metadata = resp_json['data']
        return {participant_id: metadata}

    async def _collect_all_metadata(self, reg_records):
        """ Asynchroneous function to poll metadata from registered participant
            servers

        Args:
            reg_records (list(tinydb.database.Document))): Registry of participants
        Returns:
            All participants' metadata (dict)
        """
        all_metadata = {}
        for future in asyncio.as_completed(map(self._poll_metadata, reg_records)):
            result = await future
            all_metadata.update(result)

        return all_metadata

    ##################
    # Core Functions #
    ##################

    def poll(self, reg_records):
        """ Wrapper function for triggering asychroneous polling of registered
            participants' metadata 

        Args:
            reg_records (list(tinydb.database.Document))): Registry of participants
        Returns:
            All participants' metadata (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_metadata = loop.run_until_complete(
                self._collect_all_metadata(reg_records)
            )
        finally:
            loop.close()

        return all_metadata

#######################################
# Data Orchestration class - Governor #
#######################################

class Governor:
    """
    Governs intialisations and terminations of remote WSSW objects via REST-RPC
    
    Args:
        project_id (str): Unique identifier of project
        expt_id (str): Unique identifier of experiment
        run_id (str): Unique identifier of run
        dockerised (bool): Defines if the worker nodes are dockerised or not. 
                           If worker nodes are dockerised, the host IPs & ports
                           used for intialisation default to "0.0.0.0" & "5000"
                           respectively, since actual host IPs & ports are
                           routed though container port mappings.
                           Otherwise, initialise using registered IPs & ports

    Attributes:
        # Private attributes
        __DEFAULT_SERVER_CONFIG (dict): Default server mappings to be used in a
                                        containerised setting
        # Public attributes
        project_id (str): Unique identifier of project
        expt_id (str): Unique identifier of experiment
        run_id (str): Unique identifier of run
        dockerised (bool): Defines if the worker nodes are dockerised or not. 
    """
    def __init__(self, project_id, expt_id, run_id, dockerised=True):
        self.__DEFAULT_SERVER_CONFIG = {
            'host': "0.0.0.0",
            'port': 8020
        }
        self.__rpc_formatter = RPCFormatter()
        self.project_id = project_id
        self.expt_id = expt_id
        self.run_id = run_id
        self.dockerised = dockerised

    ###########
    # Helpers #
    ###########

    @staticmethod
    def is_live(status):
        return status['is_live']

    async def _initialise_participant(self, reg_record):
        """ Parses a registration record for participant metadata, before
            posting to his/her corresponding worker node's REST-RPC service for
            WSSW initialisation

        Args:
            reg_record (tinydb.database.Document)
        Returns:
            State of WSSW object (dict)
        """
        participant_details = reg_record['participant'].copy()
        participant_id = participant_details['id']
        participant_ip = participant_details['host']
        participant_f_port = participant_details.pop('f_port') # Flask port

        # Construct destination url for interfacing with worker REST-RPC
        destination_constructor = UrlConstructor(
            host=participant_ip,
            port=participant_f_port
        )
        destination_url = destination_constructor.construct_initialise_url(
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        relevant_tags = reg_record['relations']['Tag'][0]
        self.__rpc_formatter.strip_keys(relevant_tags)

        relevant_alignments = reg_record['relations']['Alignment'][0]
        self.__rpc_formatter.strip_keys(relevant_alignments)

        payload = {
            'tags': relevant_tags,
            'alignments': relevant_alignments            
        }
        self.__rpc_formatter.strip_keys(participant_details)
        payload.update(participant_details)
        
        # If workers are dockerised, use default container mappings
        if self.dockerised:
            payload.update(self.__DEFAULT_SERVER_CONFIG)

        logging.info(f"rest_rpc.utils.Governor._initialise_participant - Payload: {payload}")
        
        # Initialise WSSW object on participant's worker node by posting tags &
        # alignments to `initialise` route in worker's REST-RPC
        async with aiohttp.ClientSession() as session:
            async with session.post(
                destination_url,
                json=payload
            ) as response:
                resp_json = await response.json(content_type='application/json')
        
        state = resp_json['data']
        return {participant_id: state}

    async def _terminate_participant(self, reg_record):
        """ Parses a registration record for participant metadata, before
            posting to his/her corresponding worker node's REST-RPC service for
            WSSW initialisation

        Args:
            reg_record (tinydb.database.Document)
        Returns:
            State of WSSW object (dict)
        """
        participant_details = reg_record['participant'].copy()
        participant_id = participant_details['id']
        participant_ip = participant_details['host']
        participant_f_port = participant_details.pop('f_port') # Flask port

        # Construct destination url for interfacing with worker REST-RPC
        destination_constructor = UrlConstructor(
            host=participant_ip,
            port=participant_f_port
        )
        destination_url = destination_constructor.construct_terminate_url(
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        # Terminate WSSW object on participant's worker node (if possible) by
        # posting to `terminate` route in worker's REST-RPC
        async with aiohttp.ClientSession() as session:
            async with session.post(destination_url) as response:
                resp_json = await response.json(content_type='application/json')
        
        state = resp_json['data']
        return {participant_id: state}

    async def _operate_on_participants(self, reg_records, operation):
        """ Asynchroneous function to poll metadata from registered participant
            servers

        Args:
            reg_records (list(tinydb.database.Document))): Registry of participants
            operation (str): Operation (either initialise or terminate) to be
                             executed on participant's worker node
        Returns:
            All participants' WSSW object states (dict)
        """
        if operation == "initialise":
            method = self._initialise_participant
        elif operation == "terminate":
            method = self._terminate_participant
        else:
            raise ValueError("Invalid operation specified")

        all_states = {}
        for future in asyncio.as_completed(map(method, reg_records)):
            result = await future
            all_states.update(result)

        return all_states

    ##################
    # Core functions #
    ##################

    def initialise(self, reg_records):
        """ Wrapper function for triggering asychroneous polling of registered
            participants' metadata 

        Args:
            reg_records (list(tinydb.database.Document))): Registry of participants
        Returns:
            All participants' metadata (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_states = loop.run_until_complete(
                self._operate_on_participants(
                    reg_records=reg_records,
                    operation="initialise"
                )
            )
        finally:
            loop.close()

        return all_states

    def terminate(self, reg_records):
        """ Wrapper function for triggering asychroneous polling of registered
            participants' metadata 

        Args:
            reg_records (list(tinydb.database.Document))): Registry of participants
        Returns:
            All participants' metadata (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_states = loop.run_until_complete(
                self._operate_on_participants(
                    reg_records=reg_records,
                    operation="terminate"
                )
            )
        finally:
            loop.close()

        return all_states

##############
# Deprecated #
##############
"""
import mlflow
remote_server_uri = "..." # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
# Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a
# valid path in the workspace
mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
    mlflow.log_param("a", 1)
    mlflow.log_metric("b", 2)

        all_metadata = {}
        async for record in reg_records:
            metadata = await self._poll_metadata(reg_record=record)
            all_metadata.update(metadata)

        all_metadata = await asyncio.gather(map(
            self._poll_metadata, 
            reg_records
        ))


        import requests
        all_metadata = {}
        for reg_record in reg_records:
            participant_details = reg_record['participant']
            participant_id = participant_details['id']
            participant_ip = participant_details['host']
            participant_port = participant_details['port']

            # Construct destination url for interfacing with worker REST-RPC
            destination_constructor = UrlConstructor(
                host=participant_ip,
                port=participant_port
            )
            destination_url = destination_constructor.construct_poll_url(
                project_id=self.project_id
            )
            print(destination_url)

            # Search for tags using composite project + participant
            relevant_tags = reg_record['relations']['Tag'][0]
            self.__rpc_formatter.strip_keys(relevant_tags)
            payload = {'tags': relevant_tags}
            response = requests.post(url=destination_url, json=payload)
            metadata = response.json()
            all_metadata.update({participant_id: metadata})

"""