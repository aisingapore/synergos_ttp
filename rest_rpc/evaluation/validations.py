#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
import mlflow
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload,
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    ParticipantRecords,
    RegistrationRecords,
    TagRecords
)
from rest_rpc.training.core.utils import (
    AlignmentRecords, 
    ModelRecords,
    Poller,
    RPCFormatter
)
from rest_rpc.evaluation.core.server import start_proc
from rest_rpc.evaluation.core.utils import ValidationRecords

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "validations", 
    description='API to faciliate model validation in a REST-RPC Grid.'
)

SUBJECT = "Validation"

db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
participant_records = ParticipantRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
tag_records = TagRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

################################################################
# Validations - Used for marshalling (i.e. moulding responses) #
################################################################

# Marshalling inputs
val_input_model = ns_api.model(
    name="prediction_input",
    model={
        'dockerised': fields.Boolean(default=False, required=True),
    }
)

# Marshalling Outputs
stats_model = ns_api.model(
    name="statistics",
    model={
        'accuracy': fields.Float(),
        'roc_auc_score': fields.Float(),
        'pr_auc_score': fields.Float(),
        'f_score': fields.Float(),
        'TPR': fields.Float(),
        'TNR': fields.Float(),
        'PPV': fields.Float(),
        'NPV': fields.Float(),
        'FPR': fields.Float(),
        'FNR': fields.Float(),
        'FDR': fields.Float(),
        'TP': fields.Integer(),
        'TN': fields.Integer(),
        'FP': fields.Integer(),
        'FN': fields.Integer()
    }
)

meta_stats_model = ns_api.model(
    name="meta_statistics",
    model={
        'statistics': fields.Nested(stats_model, skip_none=True),
        'res_path': fields.String(skip_none=True)
    }
)

val_inferences_model = ns_api.model(
    name="validation_inferences",
    model={
        'evaluate': fields.Nested(meta_stats_model, required=True)
    }
)

val_output_model = ns_api.inherit(
    "validation_output",
    val_inferences_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'participant_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, val_output_model)
