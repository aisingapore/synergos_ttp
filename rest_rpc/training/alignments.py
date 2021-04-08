#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import importlib
import inspect
import os
import random
from logging import NOTSET

# Libs
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.core.server import align_proc
from rest_rpc.training.core.utils import RPCFormatter
from synarchive.connection import ExperimentRecords, RegistrationRecords
from synarchive.training import AlignmentRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "alignments", 
    description='API to faciliate multiple feature alignment tracking in in a PySyft Grid.'
)

grid_idx = app.config['GRID']

db_path = app.config['DB_PATH']
expt_records = ExperimentRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("training/alignments.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling inputs
input_model = ns_api.model(
    name="alignment_input",
    model={
        'auto_align': fields.Boolean(default=True, required=True),
        'auto_fix': fields.Boolean(default=True, required=True)
    }
)

# Marshalling Outputs
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
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'participant_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(
    subject=alignment_records.subject, 
    namespace=ns_api, 
    model=alignment_output_model
)

#############
# Resources #
#############
"""
In connection, there was no need for both TTP and participants to interact with
alignments. However, during the training phase, preprocessing is of utmost 
priority. Hence, alignment routes are given to the TTP to allow manual 
triggering of alignment processes, mainly dataset alignment via Multiple 
Feature Alignment (MFA), model alignments via input/output detection, and other
state alignment mechanisms.
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
    def get(self, collab_id, project_id):
        """ Retrieves all alignments for all registered data under a project """
        retrieved_alignments = alignment_records.read_all(
            filter={
                'collab_id': collab_id,
                'project_id': project_id
            }
        )

        if retrieved_alignments:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="alignments.get",
                params=request.view_args,
                data=retrieved_alignments
            )
            
            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Alignments: Bulk record retrieval successful!",
                code=200, 
                description=f"Alignments under project '{project_id}' were successfully retrieved!", 
                ID_path=SOURCE_FILE,
                ID_class=Alignments.__name__, 
                ID_function=Alignments.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Alignments: Bulk record retrieval failed!",
                code=404, 
                description=f"MFA has not been performed for Project '{project_id}'!",
                ID_path=SOURCE_FILE,
                ID_class=Alignments.__name__, 
                ID_function=Alignments.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"MFA has not been performed for Project '{project_id}'!"
            )


    @ns_api.doc("trigger_alignments")
    @ns_api.expect(input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    @ns_api.response(201, "New alignments have been created!")
    def post(self, collab_id, project_id):
        """ Searches for all registered participant under project, and uses
            their registered data tags to trigger the RPC for polling 
            participant metadata for alignment
        """
        # Populate grid-initialising parameters
        init_params = request.json

        try:
            all_relevant_registrations = registration_records.read_all(
                filter={'collab_id': collab_id, 'project_id': project_id}
            )
            usable_grids = rpc_formatter.extract_grids(all_relevant_registrations)
            selected_grid = usable_grids[grid_idx]

            all_expts = expt_records.read_all(
                filter={'collab_id': collab_id, 'project_id': project_id}
            )

            spacer_collection, aligned_experiments, _ = align_proc(
                grid=selected_grid,
                kwargs={'experiments': all_expts, **init_params}
            )

            # Store generated alignment indexes for subsequent use
            retrieved_alignments = []
            for p_id, spacer_idxs in spacer_collection.items():

                alignment_records.create(
                    collab_id=collab_id,
                    project_id=project_id,
                    participant_id=p_id,
                    details=spacer_idxs
                ) 
                retrieved_alignment = alignment_records.read(
                    collab_id=collab_id,
                    project_id=project_id,
                    participant_id=p_id
                )
                retrieved_alignments.append(retrieved_alignment)

            # Apply alignment updates to existing model architecture
            for expt_updates in aligned_experiments:
                expt_records.update(**expt_updates)

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="alignments.post",
                params=request.view_args,
                data=retrieved_alignments
            )
            
            logging.info(
                f"Collaboration '{collab_id}' > Project '{project_id}' > Alignments: Record creation successful!", 
                description=f"Alignment procedure for project '{project_id}' was completed successfully!",
                code=201, 
                ID_path=SOURCE_FILE,
                ID_class=Alignments.__name__, 
                ID_function=Alignments.post.__name__,
                **request.view_args
            )

            return success_payload, 201

        except RuntimeError as e:
            logging.error(
                f"Error creating alignments for project_id: {project_id}",
                code=417,
                description="Inappropriate conditions available for multiple feature alignment!",
                ID_path=SOURCE_FILE,
                ID_class=Alignments.__name__, 
                ID_function=Alignments.post.__name__,
                **request.view_args
            )  
            ns_api.abort(
                code=417,
                message="Inappropriate conditions available for multiple feature alignment!"
            )
