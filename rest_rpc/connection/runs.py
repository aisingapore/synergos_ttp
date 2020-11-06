#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload, 
    RunRecords
)
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "runs", 
    description='API to faciliate run management in in a PySyft Grid.'
)

SUBJECT = "Run" # table name

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

config_model = ns_api.model(
    name="configurations",
    model={
        "input_size": fields.Integer(),
        "output_size": fields.Integer(),
        "batch_size": fields.Integer(),
        "rounds": fields.Integer(),
        "epochs": fields.Integer(),
        "lr": fields.Float(),
        "lr_decay": fields.Float(),
        "weight_decay": fields.Float(),
        "seed": fields.Integer(),
        "is_condensed": fields.Boolean(),
        "precision_fractional": fields.Integer(),
        "use_CLR": fields.Boolean(),
        "mu": fields.Float(),
        "reduction": fields.String(),
        "l1_lambda": fields.Float(),
        "l2_lambda": fields.Float(),
        "dampening": fields.Float(),
        "base_lr": fields.Float(),
        "max_lr": fields.Float(),
        "step_size_up": fields.Integer(),
        "step_size_down": fields.Integer(),
        "mode": fields.String(),
        "gamma": fields.Float(),
        "scale_mode": fields.String(),
        "cycle_momentum": fields.Boolean(),
        "base_momentum": fields.Float(),
        "max_momentum": fields.Float(),
        "last_epoch": fields.Integer(),
        "patience": fields.Integer(),
        "delta": fields.Float(),
        "cumulative_delta": fields.Boolean()
    }
)

run_input_model = ns_api.inherit(
    "run_input", 
    config_model, 
    {"run_id": fields.String()}
)

run_output_model = ns_api.inherit(
    "run_output",
    config_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='run_relations',
                model={
                    'Model': fields.List(
                        fields.Nested(model_output_model, skip_none=True)
                    ),
                    'Validation': fields.List(
                        fields.Nested(val_output_model, skip_none=True)
                    ),
                    'Prediction': fields.List(
                        fields.Nested(pred_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, run_output_model)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Runs(Resource):
    """ Handles the entire collection of runs as a catalogue """

    @ns_api.doc("get_runs")
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, project_id, expt_id):
        """ Retrieve all run configurations queued for training """
        all_relevant_runs = run_records.read_all(
            filter={'project_id': project_id, 'expt_id': expt_id}
        )
        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="runs.get",
            params={'project_id': project_id, 'expt_id': expt_id},
            data=all_relevant_runs
        )
        return success_payload, 200        

    @ns_api.doc("register_run")
    @ns_api.expect(run_input_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New run created!")
    @ns_api.response(417, "Inappropriate run configurations passed!")
    def post(self, project_id, expt_id):
        """ Takes in a set of FL training run configurations and stores it """
        try:
            new_run_details = request.json
            run_id = new_run_details.pop('run_id')

            new_run = run_records.create(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id,
                details=new_run_details
            )
            retrieved_run = run_records.read(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )
            assert new_run.doc_id == retrieved_run.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="runs.post",
                params={'project_id': project_id, 'expt_id': expt_id},
                data=retrieved_run
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate run configurations passed!"
            )


@ns_api.route('/<run_id>')
@ns_api.response(404, 'Run not found')
@ns_api.response(500, 'Internal failure')
class Run(Resource):
    """ Handles all TTP interactions for managing run registration """
    
    @ns_api.doc("get_run")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, project_id, expt_id, run_id):
        """ Retrieves all runs registered for an experiment under a project """
        retrieved_run = run_records.read(
            project_id=project_id, 
            expt_id=expt_id,
            run_id=run_id
        )

        if retrieved_run:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.get",
                params={
                    'project_id': project_id, 
                    'expt_id': expt_id,
                    'run_id': run_id    
                },
                data=retrieved_run
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Run '{run_id}' does not exist for Experiment {expt_id} under Project '{project_id}'!"
            )

    @ns_api.doc("update_run")
    @ns_api.expect(config_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, project_id, expt_id, run_id):
        """ Updates a run's specified configurations IF & ONLY IF the run has
            yet to begin
        """
        try:
            run_updates = request.json
            updated_run = run_records.update(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id,
                updates=run_updates
            )
            retrieved_run = run_records.read(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )
            assert updated_run.doc_id == retrieved_run.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.put",
                params={
                    'project_id': project_id, 
                    'expt_id': expt_id,
                    "run_id": run_id
                },
                data=retrieved_run
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate experimental configurations passed!"
            )

    @ns_api.doc("delete_run")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, project_id, expt_id, run_id):
        """ De-registers a previously registered run and deletes it """
        deleted_run = run_records.delete(
            project_id=project_id,
            expt_id=expt_id,
            run_id=run_id
        )

        if deleted_run:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.delete",
                params=request.view_args,
                data=deleted_run
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Run '{run_id}' does not exist in for Experiment {expt_id} under Project '{project_id}'!"
            )
            