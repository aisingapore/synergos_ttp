#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
import shutil
from pathlib import Path

# Libs
import jsonschema
import mlflow
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload, 
    ExperimentRecords
)
from rest_rpc.connection.runs import run_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "experiments", 
    description='API to faciliate experiment management in a PySyft Grid.'
)

SUBJECT = "Experiment" # table name

db_path = app.config['DB_PATH']
expt_records = ExperimentRecords(db_path=db_path)

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

structure_model = ns_api.model(
    name='structure',
    model={
        'in_features': fields.Integer(),
        'out_features': fields.Integer(),
        'bias': fields.Boolean(required=True)
    }
)

layer_model = ns_api.model(
    name="layer",
    model={
        'is_input': fields.Boolean(required=True),
        'structure': fields.Nested(
            model=structure_model, 
            skip_none=True,
            required=True
        ),
        'l_type': fields.String(required=True),
        'activation': fields.String(required=True)
    }
)

expt_model = ns_api.model(
    name="experiment",
    model={
        'model': fields.List(
            fields.Nested(layer_model, required=True, skip_none=True)
        )
    }
)

expt_input_model = ns_api.inherit(
    "experiment_input",
    expt_model,
    {'expt_id': fields.String()}
)

expt_output_model = ns_api.inherit(
    "experiment_output",
    expt_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String(),
                    'expt_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='expt_relations',
                model={
                    'Run': fields.List(
                        fields.Nested(run_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, expt_output_model)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Experiments(Resource):
    """ Handles the entire collection of experiments as a catalogue """

    @ns_api.doc("get_experiments")
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, project_id):
        """ Retrieve all run configurations queued for training """
        all_relevant_expts = expt_records.read_all(
            filter={'project_id': project_id}
        )
        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="experiments.get",
            params={'project_id': project_id},
            data=all_relevant_expts
        )
        return success_payload, 200

    @ns_api.doc("register_experiment")
    @ns_api.expect(expt_input_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New experiment created!")
    @ns_api.response(417, "Inappropriate experiment configurations passed!")
    def post(self, project_id):
        """ Takes a model configuration to be queued for training and stores it
        """
        try:
            new_expt_details = request.json
            expt_id = new_expt_details.pop('expt_id')

            new_expt = expt_records.create(
                project_id=project_id, 
                expt_id=expt_id,
                details=new_expt_details
            )
            retrieved_expt = expt_records.read(
                project_id=project_id, 
                expt_id=expt_id
            )
            assert new_expt.doc_id == retrieved_expt.doc_id

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="experiments.post",
                params={'project_id': project_id},
                data=retrieved_expt
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate experimental configurations passed!"
            )


@ns_api.route('/<expt_id>')
@ns_api.response(404, 'Experiment not found')
@ns_api.response(500, 'Internal failure')
class Experiment(Resource):
    """ Handles all TTP interactions for managing experimental configuration.
        Such interactions involve listing, specifying, updating and cancelling 
        experiments.
    """

    @ns_api.doc("get_experiment")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, project_id, expt_id):
        """ Retrieves all experimental parameters corresponding to a specified
            project
        """
        retrieved_expt = expt_records.read(
            project_id=project_id, 
            expt_id=expt_id
        )

        if retrieved_expt:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.get",
                params={'project_id': project_id, 'expt_id': expt_id},
                data=retrieved_expt
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!"
            )
            
    @ns_api.doc("update_experiment")
    @ns_api.expect(expt_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, project_id, expt_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            expt_updates = request.json

            updated_expt = expt_records.update(
                project_id=project_id, 
                expt_id=expt_id,
                updates=expt_updates
            )
            retrieved_expt = expt_records.read(
                project_id=project_id, 
                expt_id=expt_id
            )
            assert updated_expt.doc_id == retrieved_expt.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.put",
                params={'project_id': project_id, 'expt_id': expt_id},
                data=retrieved_expt
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate experimental configurations passed!"
            )

    @ns_api.doc("delete_experiment")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, project_id, expt_id):
        """ De-registers previously registered experiment, and clears out all 
            metadata
        """
        retrieved_expt = expt_records.read(
            project_id=project_id, 
            expt_id=expt_id
        )
        deleted_expt = expt_records.delete(
            project_id=project_id,
            expt_id=expt_id
        )

        if deleted_expt:
            assert deleted_expt.doc_id == retrieved_expt.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.delete",
                params=request.view_args,
                data=retrieved_expt
            )
            return success_payload

        else:
            ns_api.abort(
                code=404, 
                message=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!"
            )
