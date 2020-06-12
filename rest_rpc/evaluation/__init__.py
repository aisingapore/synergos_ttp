#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from rest_rpc.evaluation.predictions import ns_api as prediction_ns

##################
# Configurations #
##################

blueprint = Blueprint('evaluation', __name__)

api = Api(
    app=blueprint,
    version="1.0",
    title="PySyft TTP REST-RPC Evaluation API", 
    description="API to facilitate model inference between TTP & participants"
)

#############################
# Validation management API #
#############################

# api.add_namespace(
#     prediction_ns,
#     path="/participants/<participant_id>/validations"
# )

#############################
# Prediction management API #
#############################

# api.add_namespace(
#     prediction_ns,
#     path="/projects/<project_id>/predictions"
# )

api.add_namespace(
    prediction_ns,
    path="/participants/<participant_id>/predictions"
)