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
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model

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

expt_schema = app.config['SCHEMAS']['experiment_schema']

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Note: In Flask-restx==0.2.0, 
# Creating a marshallable model from a specified JSON schema is bugged. While it
# is possible to use a schema model for formatting expectations, it cannot be
# used for marshalling outputs.
# Error thrown -> AttributeError: 'SchemaModel' object has no attribute 'items'
# Mitigation   -> Manually implement schema model until bug is fixed
""" 
[REDACTED in Flask-restx==0.2.0]
structure_model = ns_api.schema_model(name='structure', schema=expt_schema)
"""
class ListableInteger():
    def format(self, value):
        return value

structure_model = ns_api.model(
    name='structure',
    model={
        "activation": fields.String(),
        "add_bias_kv": fields.Boolean(),
        "add_zero_attn": fields.Boolean(),
        "affine": fields.Boolean(),
        "align_corners": fields.Boolean(),
        "alpha": fields.Float(),
        "batch_first": fields.Boolean(),
        "beta": fields.Float(),
        "bias": fields.Boolean(),
        "bidirectional": fields.Boolean(),
        "blank": fields.Integer(),
        "ceil_mode": fields.Boolean(),
        "count_include_pad": fields.Boolean(),
        "cutoffs": fields.List(fields.String()),
        "d_model": fields.Integer(),
        "device_ids": fields.List(fields.Integer()),
        "dilation": fields.Integer(),
        "dim": fields.Integer(),
        "dim_feedforward": fields.Integer(),
        "div_value": fields.Float(),
        # "divisor_override": {
        #     "description": "if specified, it will be used as divisor in place of kernel_size"
        # },
        "dropout": fields.Float(),
        "elementwise_affine": fields.Boolean(),
        "embed_dim": fields.Integer(),
        "embedding_dim": fields.Integer(),
        "end_dim": fields.Integer(),
        "eps": fields.Float(),
        "full": fields.Boolean(),
        "groups": fields.Integer(),
        "head_bias": fields.Boolean(),
        "hidden_size": fields.Integer(),        # Flagged for arrayable value,
        "ignore_index": fields.Integer(),
        "in1_features": fields.Integer(),
        "in2_features": fields.Integer(),
        "in_channels": fields.Integer(),
        "in_features": fields.Integer(),
        "init": fields.Float(),
        "inplace": fields.Boolean(),
        "input_size": fields.Integer(),
        "k": fields.Float(),
        "kdim": fields.Integer(),
        "keepdim": fields.Boolean(),
        "kernel_size": fields.Integer(),        # Flagged for arrayable value
        "lambd": fields.Float(),
        "log_input": fields.Boolean(),
        "lower": fields.Float(),
        "margin": fields.Float(),
        "max_norm": fields.Float(),
        "max_val": fields.Float(),
        "min_val": fields.Float(),
        "mode": fields.String(),
        "momentum": fields.Float(),
        "n_classes": fields.Integer(),
        "negative_slope": fields.Float(),
        "nhead": fields.Integer(),
        "nonlinearity": fields.String(),
        "norm_type": fields.Float(),
        "normalized_shape": fields.Integer(),   # Flagged for arrayable value
        "num_channels": fields.Integer(),
        "num_chunks": fields.Integer(),
        "num_decoder_layers": fields.Integer(),
        "num_embeddings": fields.Integer(),
        "num_encoder_layers": fields.Integer(),
        "num_features": fields.Integer(),
        "num_groups": fields.Integer(),
        "num_heads": fields.Integer(),
        "num_layers": fields.Integer(),
        "num_parameters": fields.Integer(),
        "out_channels": fields.Integer(),
        "out_features": fields.Integer(),
        "output_device": fields.Integer(),
        "output_padding": fields.Integer(),
        "output_ratio": fields.Float(),         # Flagged for arrayable value
        "output_size": fields.Integer(),        # Flagged for arrayable value
        "p": fields.Float(),
        "padding": fields.Integer(),
        "padding_idx": fields.Integer(),
        "padding_mode": fields.String(),
        "pos_weight": fields.List(fields.Float()),
        "reduction": fields.String(),
        "requires_grad": fields.Boolean(),
        "return_indices": fields.Boolean(),
        "scale_factor": fields.Float(),         # Flagged for arrayable value
        "scale_grad_by_freq": fields.Boolean(),
        "size": fields.Integer(),               # Flagged for arrayable value
        "size_average": fields.Boolean(),
        "sparse": fields.Boolean(),
        "start_dim": fields.Integer(),
        "stride": fields.Integer(),             # Flagged for arrayable value
        "swap": fields.Boolean(),
        "threshold": fields.Float(),
        "track_running_stats": fields.Boolean(),
        "upper": fields.Float(),
        "upscale_factor": fields.Integer(),
        "value": fields.Float(),
        "vdim": fields.Integer(),
        "zero_infinity": fields.Boolean()
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
                    ),
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
    # @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New experiment created!")
    @ns_api.response(417, "Inappropriate experiment configurations passed!")
    def post(self, project_id):
        """ Takes a model configuration to be queued for training and stores it
        """
        # try:
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
        logging.debug(f"Retrieved experiment: {retrieved_expt}")
        assert new_expt.doc_id == retrieved_expt.doc_id

        success_payload = payload_formatter.construct_success_payload(
            status=201, 
            method="experiments.post",
            params={'project_id': project_id},
            data=retrieved_expt
        )
        return success_payload, 201

        # except jsonschema.exceptions.ValidationError:
        #     ns_api.abort(
        #         code=417,
        #         message="Inappropriate experimental configurations passed!"
        #     )


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
