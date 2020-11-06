#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import importlib
import inspect
import logging

# Libs
import aiohttp
import jsonschema
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload,
    ExperimentRecords,
    RegistrationRecords
)
from rest_rpc.training.core.utils import (
    AlignmentRecords, 
    UrlConstructor, 
    Poller,
    RPCFormatter
)
from rest_rpc.training.core.feature_alignment import MultipleFeatureAligner

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "alignments", 
    description='API to faciliate multiple feature alignment tracking in in a PySyft Grid.'
)

SUBJECT = "Alignment" # table name

db_path = app.config['DB_PATH']
expt_records = ExperimentRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)

worker_poll_route = app.config['WORKER_ROUTES']['poll']
worker_align_route = app.config['WORKER_ROUTES']['align']

rpc_formatter = RPCFormatter()

MODULE_OF_LAYERS = "torch.nn"
MODULE_OF_ACTIVATIONS = "torch.nn.functional"

activation_modules = importlib.import_module(MODULE_OF_ACTIVATIONS)

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

xy_alignment_model = ns_api.model(
    name="xy_alignments",
    model={
        'X': fields.List(fields.Integer(), required=True, default=[]),
        'y': fields.List(fields.Integer(), required=True, default=[])
    }
)

alignment_model = ns_api.model(
    name="alignments",
    model={
        'train': fields.Nested(xy_alignment_model, required=True),
        'evaluate': fields.Nested(xy_alignment_model),
        'predict': fields.Nested(xy_alignment_model),
    }
)

alignment_output_model = ns_api.inherit(
    "alignment_output",
    alignment_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String(),
                    'participant_id': fields.String(),
                    'tag_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, alignment_output_model)

#############
# Resources #
#############
"""
In connection, there was no need for both TTP and participants to interact with
alignments. However, during the training phase, preprocessing is of utmost 
priority. Hence, alignment routes are given to the TTP to allow manual 
triggering of multiple feature alignment.
"""

@ns_api.route('/')
@ns_api.response(404, 'Alignment not found')
@ns_api.response(500, 'Internal failure')
class Alignments(Resource):
    """ Using registered data tags, poll for metadata from worker nodes for 
        multiple feature alignment to be used subsequently for FL training.
    """
    
    @ns_api.doc("get_alignments")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, project_id):
        """ Retrieves all alignments for all registered data under a project """
        retrieved_alignments = alignment_records.read_all(
            filter={'project_id': project_id}
        )

        if retrieved_alignments:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="alignments.get",
                params={
                    'project_id': project_id
                },
                data=retrieved_alignments
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"MFA has not been performed for Project '{project_id}'!"
            )

    @ns_api.doc("trigger_alignments")
    #@ns_api.marshal_with(payload_formatter.plural_model)
    @ns_api.response(201, "New alignments have been created!")
    def post(self, project_id):
        """ Searches for all registered participant under project, and uses
            their registered data tags to trigger the RPC for polling 
            participant metadata for alignment
        """
        try:
            all_relevant_registrations = registration_records.read_all(
                filter={'project_id': project_id}
            )

            poller = Poller(project_id=project_id)
            all_metadata = poller.poll(all_relevant_registrations)

            (X_data_headers, y_data_headers,
             key_sequences, _) = rpc_formatter.aggregate_metadata(all_metadata)

            logging.debug(f"X_data_headers: {X_data_headers}")
            logging.debug(f"y_data_headers: {y_data_headers}")

            X_mfa_aligner = MultipleFeatureAligner(headers=X_data_headers)
            X_mf_alignments = X_mfa_aligner.align()

            y_mfa_aligner = MultipleFeatureAligner(headers=y_data_headers)
            y_mf_alignments = y_mfa_aligner.align()

            spacer_collection = rpc_formatter.alignment_to_spacer_idxs(
                X_mf_alignments=X_mf_alignments,
                y_mf_alignments=y_mf_alignments,
                key_sequences=key_sequences
            )

            retrieved_alignments = []
            for p_id, spacer_idxs in spacer_collection.items():

                new_alignment = alignment_records.create(
                    project_id=project_id,
                    participant_id=p_id,
                    details=spacer_idxs
                ) 
    
                retrieved_alignment = alignment_records.read(
                    project_id=project_id,
                    participant_id=p_id
                )

                assert new_alignment.doc_id == retrieved_alignment.doc_id   
                
                retrieved_alignments.append(retrieved_alignment)

            #############################################
            # Auto-alignment of global inputs & outputs #
            #############################################

            logging.debug(f"Alignment Superset: {X_mfa_aligner.superset} {len(X_mfa_aligner.superset)}")
            layer_modules = importlib.import_module(MODULE_OF_LAYERS)

            all_expts = expt_records.read_all(filter={'project_id': project_id})
            for curr_expt in all_expts:
            
                expt_model = curr_expt['model']

                # Check if input layer needs alignment
                input_config = expt_model.pop(0)
                input_layer = getattr(layer_modules, input_config['l_type'])
                input_params = list(inspect.signature(input_layer.__init__).parameters)
                input_key = input_params[1] # from [self, input, output, ...]
                
                # Only modify model inputs if handling non-image data! An 
                # assumption for now is that all collaborating parties have 
                # images of the same type of color scale (eg. grayscale, RGBA) 
                if "in_channels" not in input_key:
                    aligned_input_size = len(X_mfa_aligner.superset)
                    input_config['structure'][input_key] = aligned_input_size

                expt_model.insert(0, input_config)

                logging.debug(f"Modified input config: {input_config}")
                logging.debug(f"Modified experiment: {expt_model}")

                # Check if output layer needs alignment
                output_config = expt_model.pop(-1)
                output_layer = getattr(layer_modules, output_config['l_type'])
                output_params = list(inspect.signature(output_layer.__init__).parameters)
                output_key = output_params[2] # from [self, input, output, ...]
                aligned_output_size = len(y_mfa_aligner.superset)
                if aligned_output_size <= 2:
                    # Case 1: Regression or Binary classification
                    output_config['structure'][output_key] = 1
                else:
                    # Case 2: Multiclass classification
                    output_config['structure'][output_key] = aligned_output_size
                    
                    # If the no. of class labels has expanded, switch from 
                    # linear activations to softmax variants
                    output_config['activation'] = "softmax"

                expt_model.append(output_config)

                logging.debug(f"Modified output config: {output_config}")
                logging.debug(f"Modified experiment: {expt_model}")

                expt_records.update(
                    **curr_expt['key'], 
                    updates={'model': expt_model}
                )

                logging.debug(f"Updated records: {expt_records.read(**curr_expt['key'])}")

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="alignments.post",
                params={
                    'project_id': project_id
                },
                data=retrieved_alignments
            )
            
            return success_payload, 201

        except RuntimeError as e:
            ns_api.abort(
                code=417,
                message="Inappropriate conditions available for multiple feature alignment!"
            )
