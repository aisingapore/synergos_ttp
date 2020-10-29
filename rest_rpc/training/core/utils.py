#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import copy
import importlib
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Callable

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
Training:
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
            "<run_id>", run_id
        )
        return custom_intialise_url

    def construct_terminate_url(self, project_id, expt_id, run_id):
        destination_url = self.construct_url(route=worker_terminate_route)
        custom_terminate_url = destination_url.replace(
            "<project_id>", project_id
        ).replace(
            "<expt_id>", expt_id
        ).replace(
            "<run_id>", run_id
        )
        return custom_terminate_url

    def construct_predict_url(self, project_id, expt_id, run_id):
        destination_url = self.construct_url(route=worker_predict_route)
        custom_predict_url = destination_url.replace(
            "<project_id>", project_id
        ).replace(
            "<expt_id>", expt_id
        ).replace(
            "<run_id>", run_id
        )
        return custom_predict_url

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
            relations=["Validation", "Prediction"]
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
           
##########################################
# Data Augmentation class - RPCFormatter #
##########################################

class RPCFormatter:
    """
    Automates the formatting of structured data stored within the TTP into a
    form compatible with the interfaces defined by the worker node's REST-RPC 
    service.
    """
    def strip_keys(self, record, concise: bool = False):
        """ Remove db-specific keys and descriptors

        Args:
            record (tinydb.database.Document): Target record to strip
        Returns:
            Stripped record (tinydb.database.Document)
        """
        copied_record = copy.deepcopy(record)
        copied_record.pop('key')
        copied_record.pop('created_at')
        try:
            copied_record.pop('link')
        except KeyError:
            pass
        if concise:
            copied_record.pop('relations')
        return copied_record

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

        logging.debug(f"X_mf_alignments: {X_mf_alignments}, y_mf_alignments: {y_mf_alignments}")
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
        self.__rpc_formatter = RPCFormatter()
        self.project_id = project_id

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
        project_action = reg_record['project']['action']

        participant_details = reg_record['participant'].copy()
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
        stripped_tags = self.__rpc_formatter.strip_keys(relevant_tags)
        
        payload = {'action': project_action, 'tags': stripped_tags}

        # Poll for headers by posting tags to `Poll` route in worker
        timeout = aiohttp.ClientTimeout(
            total=None, 
            connect=None,
            sock_connect=None, 
            sock_read=None
        ) # `0` or None value to disable timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                destination_url,
                timeout=timeout,
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
        auto_align (bool): Toggles if multiple feature alignments will be used
        dockerised (bool): Defines if the worker nodes are dockerised or not
    """
    def __init__(
        self, 
        project_id: str,
        expt_id: str, 
        run_id: str, 
        auto_align: bool = True,
        dockerised: bool = True
    ):
        self.__DEFAULT_SERVER_CONFIG = {
            'host': "0.0.0.0",
            'port': 8020
        }
        self.__rpc_formatter = RPCFormatter()
        self.project_id = project_id
        self.expt_id = expt_id
        self.run_id = run_id
        self.auto_align = auto_align
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
        project_action = reg_record['project']['action'] 

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
        stripped_tags = self.__rpc_formatter.strip_keys(relevant_tags)

        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # Decoupling of MFA from training should be made more explicit

        # [Problems]
        # Auto-alignment is not scalable to datasets that have too many features
        # and can consume too much computation resources such that the container 
        # will crash.

        # [Solution]
        # Explicitly declare a new state parameter that allows alignments to be
        # be skipped when necessary, provided that the declared model parameters
        # are CORRECT!!! If `auto-align` is true, then MFA indexes will be 
        # loaded into the grid, on the condition that MFA has already been 
        # performed, otherwise a Runtime Error will be thrown. If `auto-align` 
        # is false, then an empty set of MFA indexes will be declared. 

        if self.auto_align:
            try:
                relevant_alignments = reg_record['relations']['Alignment'][0]
                stripped_alignments = self.__rpc_formatter.strip_keys(
                    relevant_alignments
                )
            except (KeyError, AttributeError) as e:
                logging.error(f"Governor._initialise_participant: Error - {e}")
                raise RuntimeError("No prior alignments have been detected! Please run multiple feature alignment first and try again!")
        else:
            stripped_alignments = {
                meta: {"X": [], "y": []} # do not apply any alignment indexes
                for meta, _ in stripped_tags.items()
            }

        payload = {
            'action': project_action,
            'tags': stripped_tags,
            'alignments': stripped_alignments            
        }
        stripped_participant_details = self.__rpc_formatter.strip_keys(participant_details)
        payload.update(stripped_participant_details)
        
        # If workers are dockerised, use default container mappings
        if self.dockerised:
            payload.update(self.__DEFAULT_SERVER_CONFIG)

        logging.info(f"rest_rpc.utils.Governor._initialise_participant - Payload: {payload}")
        
        # Initialise WSSW object on participant's worker node by posting tags &
        # alignments to `initialise` route in worker's REST-RPC
        timeout = aiohttp.ClientTimeout(
            total=None, 
            connect=None,
            sock_connect=None, 
            sock_read=None
        ) # `0` or None value to disable timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                destination_url,
                timeout=timeout,
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
        timeout = aiohttp.ClientTimeout(
            total=None, 
            connect=None,
            sock_connect=None, 
            sock_read=None
        ) # `0` or None value to disable timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                destination_url,
                timeout=timeout
            ) as response:
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


############################################
# Base Configuration Parser Class - Parser #
############################################

class Parser:

    def parse_operation(self, module_str: str, operation_str: str):
        """ Detects layer type of a specified layer from configuration

        Args:
            module_str    (str): String module to search from
            operation_str (str): String operation to translate
        Returns:
            Module operation
        """
        try:
            module = importlib.import_module(module_str)
            operation = getattr(module, operation_str)
            return operation

        except AttributeError:
            logging.error(f"Specified operation '{operation_str}' is not supported!")

############################################
# Configuration Parser Class - TorchParser #
############################################

class TorchParser(Parser):
    """ Dynamically translates string names to PyTorch classes

    Attributes:
        MODULE_OF_LAYERS      (str): Import string for layer modules
        MODULE_OF_ACTIVATIONS (str): Import string for activation modules
        MODULE_OF_OPTIMIZERS  (str): Import string for optimizer modules
        MODULE_OF_CRITERIONS  (str): Import string for criterion modules
        MODULE_OF_SCHEDULERS  (str): Import string for scheduler modules
    """
    
    def __init__(self):
        super().__init__()
        self.MODULE_OF_LAYERS = "torch.nn"
        self.MODULE_OF_ACTIVATIONS = "torch.nn.functional"
        self.MODULE_OF_OPTIMIZERS = "torch.optim"
        self.MODULE_OF_CRITERIONS = "torch.nn"
        self.MODULE_OF_SCHEDULERS = "torch.optim.lr_scheduler"


    def parse_layer(self, layer_str: str) -> Callable:
        """ Detects layer type of a specified layer from configuration

        Args:
            layer_str (str): Layer type to initialise
        Returns:
            Layer definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_LAYERS, layer_str)


    def parse_activation(self, activation_str: str) -> Callable:
        """ Detects activation function specified from configuration

        Args:
            activation_type (str): Activation function to initialise
        Returns:
            Activation definition (Callable)
        """
        if not activation_str:
            return lambda x: x

        return self.parse_operation(self.MODULE_OF_ACTIVATIONS, activation_str)
    

    def parse_optimizer(self, optim_str: str) -> Callable:
        """ Detects optimizer specified from configuration

        Args:
            optim_str (str): Optimizer to initialise
        Returns:
            Optimizer definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_OPTIMIZERS, optim_str)


    def parse_criterion(self, criterion_str: str) -> Callable:
        """ Detects criterion specified from configuration

        Args:
            criterion_str (str): Criterion to initialise
        Returns:
            Criterion definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_CRITERIONS, criterion_str)


    def parse_scheduler(self, scheduler_str: str) -> Callable:
        """ Detects learning rate schedulers specified from configuration

        Args:
            scheduler_str (str): Learning rate scheduler to initialise
        Returns:
            Scheduler definition (Callable)
        """
        if not scheduler_str:
            return self.parse_operation(self.MODULE_OF_SCHEDULERS, "LambdaLR")

        return self.parse_operation(self.MODULE_OF_SCHEDULERS, scheduler_str)