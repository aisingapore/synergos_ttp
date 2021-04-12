#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import copy
import importlib
import os
import re
import time
from string import Template
from typing import Any, Dict, List, Union, Tuple, Callable

# Libs
import aiohttp
from tinydb.database import Document

# Custom
from rest_rpc import app

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

schemas = app.config['SCHEMAS']

retry_interval = app.config['RETRY_INTERVAL']
worker_poll_template = app.config['WORKER_ROUTE_TEMPLATES']['poll']
worker_align_template = app.config['WORKER_ROUTE_TEMPLATES']['align']
worker_initialise_template = app.config['WORKER_ROUTE_TEMPLATES']['initialise']
worker_terminate_template = app.config['WORKER_ROUTE_TEMPLATES']['terminate']
worker_predict_template = app.config['WORKER_ROUTE_TEMPLATES']['predict']

node_id_template = app.config['NODE_ID_TEMPLATE']
node_pid_regex = app.config['NODE_PID_REGEX']
node_nid_regex = app.config['NODE_NID_REGEX']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("training/core/utils.py logged", Description="No Changes")

#################################
# Helper Class - UrlConstructor #
#################################

class UrlConstructor:
    """ 
    Helper class that facilitates the construction of REST-RPC related URLs

    Attributes:
        host (str): IP of the remote node to be contacted
        port (int): Port of the remote node assigned for REST-RPC
        secure (bool): Toggles the security of connection (i.e. http or https)
            Default: False (i.e. http)
    """
    def __init__(self, host: str, port: int, secure: bool = False):
        self.host = host
        self.port = port
        self.secure = secure

    ###########
    # Helpers #
    ###########

    def construct_url(self, route: str) -> str:
        protocol = "http" if not self.secure else "https"
        base_url = f"{protocol}://{self.host}:{self.port}"
        destination_url = base_url + route
        return destination_url

    ##################
    # Core Functions #
    ##################

    def construct_poll_url(self, collab_id: str, project_id: str) -> str:
        custom_poll_url = worker_poll_template.substitute({
            'collab_id': collab_id,
            'project_id': project_id
        })
        destination_url = self.construct_url(route=custom_poll_url)
        return destination_url
        

    def construct_align_url(self, collab_id: str, project_id: str) -> str:
        custom_align_url = worker_align_template.substitute({
            'collab_id': collab_id,
            'project_id': project_id
        })
        destination_url = self.construct_url(route=custom_align_url)
        return destination_url


    def construct_initialise_url(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> str:
        custom_intialise_url = worker_initialise_template.substitute({
            'collab_id': collab_id,
            'project_id': project_id,
            'expt_id': expt_id,
            'run_id': run_id
        })
        destination_url = self.construct_url(route=custom_intialise_url)
        return destination_url


    def construct_terminate_url(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> str:
        custom_terminate_url = worker_terminate_template.substitute({
            'collab_id': collab_id,
            'project_id': project_id,
            'expt_id': expt_id,
            'run_id': run_id
        })
        destination_url = self.construct_url(route=custom_terminate_url)
        return destination_url


    def construct_predict_url(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> str:
        custom_predict_url = worker_predict_template.substitute({
            'collab_id': collab_id,
            'project_id': project_id,
            'expt_id': expt_id,
            'run_id': run_id
        })
        destination_url = self.construct_url(route=custom_predict_url)
        return destination_url


           
##########################################
# Data Augmentation class - RPCFormatter #
##########################################

class RPCFormatter:
    """
    Automates the formatting of structured data stored within the TTP into a
    form compatible with the interfaces defined by the worker node's REST-RPC 
    service. This class stores all core formatting operations that are used
    throughout the system as a consolidated point of reference for easy 
    maintanence.
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
        
                eg.

                {
                    <participant_id_1>: {

                        "headers": {
                            "train": {
                                "X": ["X1_1", "X1_2", "X2_1", "X2_2", "X3"],
                                "y": ["target_1", "target_2"]
                            },
                            ...
                        },

                        "schemas": {
                            "train": {
                                "X1": "int32",
                                "X2": "category", 
                                "X3": "category", 
                                "X4": "int32", 
                                "X5": "int32", 
                                "X6": "category", 
                                "target": "category"
                            },
                            ...
                        },
                        
                        "metadata":{
                            "train":{
                                'src_count': 1000,
                                '_type': "<insert datatype>",
                                <insert type-specific meta statistics>
                                ...
                            },
                            ...
                        }
                    },
                    ...
                }
        
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
        descriptors = {}
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

            dataset_stats = metadata['metadata']
            descriptors[participant_id] = dataset_stats

        return (
            X_data_headers,
            y_data_headers, 
            key_sequences, 
            super_schema,
            descriptors
        )


    def alignment_to_spacer_idxs(
        self, 
        X_mf_alignments: Tuple[List[str]], 
        y_mf_alignments: Tuple[List[str]], 
        key_sequences: List[Tuple[str]]
    ) -> dict:
        """ Aggregates feature and target alignments and formats them w.r.t 
            each corresponding participant

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
            """ Given a list of headers obtained from MFA, generate the
                necessary null indexes that will be required to augment the
                data residing in the remote node

            Args:
                alignment (list(str)): Aligned list of headers from MFA
            Returns:
                Null indexes (list(int))
            """
            return [idx for idx, e in enumerate(alignment) if e == None]

        logging.debug(
            f"Alignments to convert tracked.",
            X_mf_alignments=X_mf_alignments,
            y_mf_alignments=y_mf_alignments,
            ID_path=SOURCE_FILE,
            ID_class=RPCFormatter.__name__,
            ID_function=RPCFormatter.alignment_to_spacer_idxs.__name__
        )

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


    # def generate_node_id(self, participant: str, signature: str) -> str:
    #     """
    #     """
    #     return node_id_template.substitute({
    #         'participant': participant,
    #         'node': signature
    #     })


    # def nodeID_to_participant(self, node_id: str) -> str:
    #     """
    #     """
    #     return re.findall(node_pid_regex, node_id).pop()


    # def nodeID_to_signature(self, node_id: str) -> str:
    #     """
    #     """
    #     return re.findall(node_nid_regex, node_id).pop()


    def extract_grids(self, reg_records: List[Document]):
        """ Given a list of registration records from participants in the same
            project under a specific collaboration, extract N no. of grids, 
            where N is some pre-determined number decided upon by all parties
            before the current federated cycle commences.

            For example, if all parties agreed to hosting 3 grids, then each
            party's registration records should look like this:

            {
                'key': {
                    'collab_id': "test_collab",
                    'project_id': "test_project",
                    'participant_id': "test_participant_1"
                }
                'role': "guest",
                'n_count': 3,
                'node_0': {
                    'host': "111.111.111.111",
                    'port': 8020,
                    'f_port': 5000,
                    'log_msgs': False,
                    'verbose': False
                },
                'node_1': {
                    'host': "222.222.222.222",
                    'port': 8020,
                    'f_port': 5000,
                    'log_msgs': False,
                    'verbose': False
                },
                'node_2': {
                    'host': "333.333.333.333",
                    'port': 8020,
                    'f_port': 5000,
                    'log_msgs': False,
                    'verbose': False
                }
                'collaboration': {...},
                'project': {...},
                'participant': {...},
                'relations': [...]
            }

        After processing, the grids generated will be as follows:

        eg. Grid #1
        [
            {
                'keys': {
                    'collab_id': "test_collab",
                    'project_id': "test_project",
                    'participant_id': "test_participant_1"
                },
                'action': "classify",
                'rest': {
                    'host': "111.111.111.111",  # Node 0 of participant_1
                    'port': 5000
                },
                'syft': {
                    'id': "test_participant_1-[node_0]"
                    'host': "111.111.111.111",
                    'port': 8020,
                    'log_msgs': False,
                    'verbose': False
                },
                'relations': [...]
            }, 
            ...
        ]

        Args:

        Returns:
            Grids
        """
        aligned_n_count = min([record['n_count'] for record in reg_records])

        grids = []
        for n_idx in range(aligned_n_count):
            node_signature = f"node_{n_idx}"

            curr_grid = []
            for record in reg_records:
                keys = record['key']
                action = record['project']['action']

                node = record[node_signature]
                rest_connection = {
                    'host': record[node_signature]['host'],
                    'port': record[node_signature]['f_port']
                }

                node.pop('f_port')
                syft_connection = {'id': record['participant']['id'], **node}

                relations = record['relations']

                node_info = {
                    'keys': keys,
                    'action': action,
                    'rest': rest_connection,
                    'syft': syft_connection,
                    'relations': relations
                }
                curr_grid.append(node_info)

            grids.append(curr_grid)
                
        return grids


    def enumerate_federated_conbinations(
        self,
        action: str,
        experiments: list,
        runs: list,
        auto_align: bool = True,
        dockerised: bool = True,
        log_msgs: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> dict:
        """ Enumerates all registered combinations of experiment models and run
            configurations for a SINGLE project in preparation for bulk operations.

        Args:
            action (str): Type of machine learning operation to be executed
            experiments (list): All experimental models to be reconstructed
            runs (dict): All hyperparameter sets to be used during grid FL inference
            auto_align (bool): Toggles if multiple feature alignments will be used
            dockerised (bool): Toggles if current FL grid is containerised or not. 
                If true (default), hosts & ports of all participants are locked at
                "0.0.0.0" & 8020 respectively. Otherwise, participant specified
                configurations will be used (grid architecture has to be finalised).
            log_msgs (bool): Toggles if messages are to be logged
            verbose (bool): Toggles verbosity of logs for WSCW objects
            **kwargs: Miscellaneous keyword argmuments to 
        Returns:
            Combinations (dict)
        """
        combinations = {}
        for expt_record in experiments:
            curr_expt_id = expt_record['key']['expt_id']

            for run_record in runs:
                run_key = run_record['key']
                collab_id = run_key['collab_id']
                project_id = run_key['project_id']
                expt_id = run_key['expt_id']
                run_id = run_key['run_id']

                if expt_id == curr_expt_id:

                    combination_key = (collab_id, project_id, expt_id, run_id)
                    combination_params = {
                        'keys': run_key,
                        'action': action,
                        'experiment': expt_record,
                        'run': run_record,
                        'auto_align': auto_align,
                        'dockerised': dockerised, 
                        'log_msgs': log_msgs, 
                        'verbose': verbose,
                        **kwargs
                    }
                    combinations[combination_key] = combination_params

        return combinations


##########################################
# Base Orchestrator Class - Orchestrator #
##########################################

class Orchestrator:
    """

    Attributes:
        __rpc_formatter (RPCFormatter): Formatter for structured data in REST-RPC
    """
    def __init__(self):
        self.__rpc_formatter = RPCFormatter()

    ###########
    # Helpers #
    ###########

    def parse_keys(self, node_info: dict) -> Dict[str, str]:
        """

        Args:
            node_info (dict):
        Returns:
            
        """
        node_keys = node_info.get('keys')
        collab_id = node_keys.get('collab_id')
        project_id = node_keys.get('project_id')
        participant_id = node_keys.get('participant_id')
        return collab_id, project_id, participant_id


    def parse_action(self, node_info: dict) -> str:
        """

        Args:
            node_info (dict):
        Returns:
            
        """
        return node_info.get('action')


    def parse_rest_info(self, node_info: dict) -> dict:
        """

        Args:
            node_info (dict):
        Returns:
            
        """
        return node_info.get('rest')

    
    def parse_syft_info(self, node_info: dict) -> dict:
        """

        Args:
            node_info (dict):
        Returns:
            
        """
        return node_info.get('syft')


    def parse_tags(self, node_info: dict) -> dict:
        """

        Args:
            node_info (dict):
        Returns:
            
        """
        node_relations = node_info.get('relations')
        relevant_tags = node_relations['Tag'][0]
        stripped_tags = self.__rpc_formatter.strip_keys(relevant_tags)
        return stripped_tags
        

    def parse_alignments(
        self, 
        node_info: dict, 
        auto_align: bool = True
    ) -> dict:
        """

        Args:
            node_info (dict):
            auto_align (bool): Toggles if dynamic alignment will be applied
        Returns:

        """
        node_relations = node_info.get('relations')
        node_tags = self.parse_tags(node_info) 

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

        if auto_align:
            try:
                relevant_alignments = node_relations['Alignment'][0]
                stripped_alignments = self.__rpc_formatter.strip_keys(
                    relevant_alignments
                )
            except (KeyError, AttributeError) as e:
                logging.error(
                    "No prior alignments have been detected! Please run multiple feature alignment first and try again!", 
                    description=f"{e}",
                    ID_path=SOURCE_FILE,
                    ID_class=Orchestrator.__name__,
                    ID_function=Orchestrator.parse_alignments.__name__
                )
                raise RuntimeError(
                    "No prior alignments have been detected! Please run multiple feature alignment first and try again!"
                )

        else:
            stripped_alignments = {
                meta: {"X": [], "y": []} # do not apply any alignment indexes
                for meta, _ in node_tags.items()
            }

        return stripped_alignments

    ##################    
    # Core Functions #
    ##################

    async def instruct(
        self, 
        command: str, 
        url: str, 
        payload: dict = {}
    ) -> dict:
        """ Sends an order to a remote node for execution. An order is a
            prefined REST command that is triggered when a particular url 
            hosted by the remote machine is contacted.

        Args:
            command (str): Type of request (i.e. 'get', 'post', 'put', 'delete')
            url (str): Endpoint corresponding to the order
            payload (dict): Information required to trigger the order
        Returns:
            Response Data (dict)
            Response Status (int)
        """
        # Disable connection timeout
        timeout = aiohttp.ClientTimeout(
            total=None, 
            connect=None,
            sock_connect=None, 
            sock_read=None
        ) # `0` or None value to disable timeout

        # Submit loading job by POSTING tags to `Poll` route in worker
        async with aiohttp.ClientSession(timeout=timeout) as session:
            order = getattr(session, command)

            async with order(url, timeout=timeout, json=payload) as response:
                resp_json = await response.json(content_type='application/json')
                resp_data = resp_json.get('data')
                resp_status = response.status

        return resp_data, resp_status

            

#####################################
# Data Orchestration class - Poller #
#####################################

class Poller(Orchestrator):
    """
    Polls for headers & schemas, before performing multiple feature alignment,
    to obtain aligned headers alongside a super schema for subsequent reference
    """
    def __init__(self):
        super().__init__()

    ###########
    # Helpers #
    ###########

    async def _poll_metadata(self, node_info: dict) -> dict:
        """ Parses a registration record for participant metadata, before
            polling for data-specific descriptors from corresponding worker 
            node's REST-RPC service

        Args:
            node_info (dict): Metadata describing a single participant node
        Returns:
            headers (dict)
        """
        collab_id, project_id, participant_id = self.parse_keys(node_info)

        # Construct destination url for interfacing with worker REST-RPC
        rest_connection = self.parse_rest_info(node_info)
        destination_constructor = UrlConstructor(**rest_connection)
        destination_url = destination_constructor.construct_poll_url(
            collab_id=collab_id,
            project_id=project_id
        )

        ml_action = self.parse_action(node_info)

        # Search for tags using composite project + participant
        data_tags = self.parse_tags(node_info)
        
        payload = {'action': ml_action, 'tags': data_tags}

        # Submit loading job by POSTING tags to `Poll` route in worker
        load_resp_data, _ = await self.instruct(
            command='post', 
            url=destination_url, 
            payload=payload
        )
        jobs_in_queue = load_resp_data.get('jobs')

        # Check if archives are retrievable every X=1 second
        archive_retrieved = False
        metadata = None
        while not archive_retrieved:

            # Attempt archive retrieval via GET request to `Poll` route in worker
            retrieval_resp_data, resp_status = await self.instruct(
                command='get', 
                url=destination_url, 
                payload=payload
            )
                       
            if resp_status == 200:
                metadata = retrieval_resp_data
                archive_retrieved = True

            elif resp_status == 406:
                sleep_interval = jobs_in_queue * retry_interval

                logging.info(
                    f"Archives for participant '{participant_id}' are not yet loaded. Retrying after {sleep_interval} seconds...",
                    ID_path=SOURCE_FILE,
                    ID_class=Poller.__name__,
                    ID_function=Poller._poll_metadata.__name__
                )

                time.sleep(sleep_interval)

            else:
                logging.error(
                    f"Something went wrong when polling for metadata from participant '{participant_id}'!",
                    description=f"Unexpected response status received: {resp_status}",
                    ID_path=SOURCE_FILE,
                    ID_class=Poller.__name__,
                    ID_function=Poller._poll_metadata.__name__
                )
                raise RuntimeError(f"Something went wrong when polling for metadata from participant '{participant_id}'")
        
        return {participant_id: metadata}


    async def _collect_all_metadata(
        self, 
        grid: List[Dict[str, Union[str, Any]]]
    ) -> Dict[str, Union[str, Any]]:
        """ Asynchroneous function to poll metadata from registered participant
            servers

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' metadata (dict)
        """
        all_metadata = {}
        for future in asyncio.as_completed(map(self._poll_metadata, grid)):
            result = await future
            all_metadata.update(result)

        return all_metadata

    ##################
    # Core Functions #
    ##################

    def poll(self, grid: List[Dict[str, Any]]):
        """ Wrapper function for triggering asychroneous polling of registered
            participants' metadata 

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' metadata (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_metadata = loop.run_until_complete(
                self._collect_all_metadata(grid=grid)
            )
        finally:
            loop.close()

        return all_metadata



#######################################
# Data Orchestration class - Governor #
#######################################

class Governor(Orchestrator):
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
        collab_id (str): Unique identifier of collaboration
        project_id (str): Unique identifier of project
        expt_id (str): Unique identifier of experiment
        run_id (str): Unique identifier of run
        auto_align (bool): Toggles if dynamic alignment will be applied
        dockerised (bool): Defines if the worker nodes are dockerised or not
    """
    def __init__(
        self, 
        collab_id: str,
        project_id: str,
        expt_id: str, 
        run_id: str, 
        auto_align: bool = True,
        dockerised: bool = True
    ):
        super().__init__()

        self.__DEFAULT_SERVER_CONFIG = {
            'host': "0.0.0.0",
            'port': 8020
        }
        self.collab_id = collab_id
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


    async def _initialise_participant(self, node_info: dict) -> dict:
        """ Parses a registration record for participant metadata, before
            posting to his/her corresponding worker node's REST-RPC service for
            WSSW initialisation

        Args:
            node_info (dict): Metadata describing a single participant node
        Returns:
            State of WSSW object (dict)
        """
        # Construct destination url for interfacing with worker REST-RPC
        rest_connection = self.parse_rest_info(node_info)
        destination_constructor = UrlConstructor(**rest_connection)
        destination_url = destination_constructor.construct_initialise_url(
            collab_id=self.collab_id,
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        _, _, participant_id = self.parse_keys(node_info)
        participant_details = self.parse_syft_info(node_info)

        ml_action = self.parse_action(node_info)
        data_tags = self.parse_tags(node_info)
        data_alignments = self.parse_alignments(node_info, self.auto_align)

        logging.warn(f"---> data alignments: {data_alignments}")

        payload = {
            'action': ml_action,
            'tags': data_tags,
            'alignments': data_alignments,      
            **participant_details
        }
        
        # If workers are dockerised, use default container mappings
        if self.dockerised:
            payload.update(self.__DEFAULT_SERVER_CONFIG)

        logging.debug(
            f"Initialisation payload used for governing server worker {participant_id} tracked!",
            payload=payload, 
            ID_path=SOURCE_FILE,
            ID_class=Governor.__name__,
            ID_function=Governor._initialise_participant.__name__
        )
        
        # Initialise WSSW object on participant's worker node by posting tags &
        # alignments to `initialise` route in worker's REST-RPC
        state, _ = await self.instruct(
            command='post', 
            url=destination_url, 
            payload=payload
        )
        return {participant_id: state}


    async def _terminate_participant(self, node_info: dict) -> dict:
        """ Parses a registration record for participant metadata, before
            posting to his/her corresponding worker node's REST-RPC service for
            WSSW initialisation

        Args:
            reg_record (tinydb.database.Document)
        Returns:
            State of WSSW object (dict)
        """
        # Construct destination url for interfacing with worker REST-RPC
        rest_connection = self.parse_rest_info(node_info)
        destination_constructor = UrlConstructor(**rest_connection)
        destination_url = destination_constructor.construct_terminate_url(
            collab_id=self.collab_id,
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        _, _, participant_id = self.parse_keys(node_info)

        # Terminate WSSW object on participant's worker node (if possible) by
        # posting to `terminate` route in worker's REST-RPC
        state, _ = await self.instruct(command='post', url=destination_url)

        return {participant_id: state}


    async def _operate_on_participants(
        self, 
        grid: List[Dict[str, Any]],
        operation: str
    ):
        """ Asynchroneous function to poll metadata from registered participant
            servers

        Args:
            grid (list(dict))): Registry of participants' node information
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
            logging.error(
                "ValueError: Invalid operation specified", 
                ID_path=SOURCE_FILE,
                ID_class=Governor.__name__,
                ID_function=Governor._operate_on_participants.__name__
            )
            raise ValueError("Invalid operation specified")

        all_states = {}
        for future in asyncio.as_completed(map(method, grid)):
            result = await future
            all_states.update(result)

        return all_states

    ##################
    # Core functions #
    ##################

    def initialise(self, grid: List[Dict[str, Any]]) -> dict:
        """ Wrapper function for triggering asychroneous polling of registered
            participants' metadata 

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' metadata (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_states = loop.run_until_complete(
                self._operate_on_participants(
                    grid=grid,
                    operation="initialise"
                )
            )
        finally:
            loop.close()

        return all_states


    def terminate(self, grid: List[Dict[str, Any]]) -> dict:
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
                    grid=grid,
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
    """ 
    Base class that facilitates the loading of modules at runtime given
    their string names
    """

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
            logging.error(
                f"AttributeError: Specified operation '{operation_str}' is not supported!", 
                ID_path=SOURCE_FILE,
                ID_class=Parser.__name__,
                ID_function=Parser.parse_operation.__name__
            )



############################################
# Configuration Parser Class - TorchParser #
############################################

class TorchParser(Parser):
    """ 
    Dynamically translates string names to PyTorch classes

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



###########################################
# Configuration Parser Class - TuneParser #
###########################################

class TuneParser(Parser):
    """ 
    Dynamically translates string names to Tune API callables

    Attributes:
        MODULE_OF_HYPERPARAM_TYPES (str): Import string for hyperparam types
    """
    
    def __init__(self):
        super().__init__()
        self.MODULE_OF_HYPERPARAM_TYPES = "ray.tune"
        self.MODULE_OF_HYPERPARAM_SCHEDULERS = "ray.tune.schedulers"
        self.MODULE_OF_HYPERPARAM_SEARCHERS = "ray.tune.suggest"


    def parse_type(self, type_str: str) -> Callable:
        """ Detects hyperparameter type of a declared hyperparameter from
            configuration

        Args:
            type_str (str): Layer type to initialise
        Returns:
            Type definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_HYPERPARAM_TYPES, type_str)


    def parse_scheduler(self, scheduler_str: str) -> Callable:
        """ Detects hyperparameter scheduler from configuration. This variant
            is important as `ray.tune.create_scheduler` requires kwargs to be
            specified to return a fully instantiated scheduler, whereas this
            way the scheduler parameter signature can be retrieved.

        Args:
            scheduler_str (str): Scheduler type to initialise
        Returns:
            Scheduler definition (Callable)
        """
        return self.parse_operation(
            self.MODULE_OF_HYPERPARAM_SCHEDULERS, 
            scheduler_str
        )


    def parse_searcher(self, searcher_str: str) -> Callable:
        """ Detects hyperparameter searcher from configuration. This variant
            is important as `ray.tune.create_searcher` requires kwargs to be
            specified to return a fully instantiated scheduler, whereas this
            way the scheduler parameter signature can be retrieved.

        Args:
            searcher_str (str): Searcher type to initialise
        Returns:
            Searcher definition (Callable)
        """
        SEARCHER_MAPPINGS = {
            'BasicVariantGenerator': "basic_variant",
            'AxSearch': "ax",
            'BayesOptSearch': "bayesopt",
            'TuneBOHB': "bohb",
            'DragonflySearch': "dragonfly",
            'HEBOSearch': "hebo",
            'HyperOptSearch': "hyperopt",
            'NevergradSearch': "nevergrad",
            'OptunaSearch': "optuna",
            'SigOptSearch': "sigopt",
            'SkOptSearch': "skopt",
            'ZOOptSearch': "zoopt"
        }
        partial_import_str = SEARCHER_MAPPINGS[searcher_str]
        return self.parse_operation(
            '.'.join([self.MODULE_OF_HYPERPARAM_SEARCHERS, partial_import_str]),
            searcher_str
        )
