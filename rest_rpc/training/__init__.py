#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from rest_rpc.training.alignments import ns_api as alignment_ns
from rest_rpc.training.models import ns_api as model_ns
from rest_rpc.training.optimizations import ns_api as optim_ns

##################
# Configurations #
##################

blueprint = Blueprint('training', __name__)

api = Api(
    app=blueprint,
    version="1.0",
    title="PySyft TTP REST-RPC Training API", 
    description="API to facilitate Grid setup & training kickoff between TTP & participants"
)

############################
# Alignment management API #
############################
"""
Supported routes:
1) "/projects/<project_id>/alignments"
"""

api.add_namespace(
    alignment_ns,
    path="/projects/<project_id>/alignments"
)

########################
# Model management API #
########################
"""
Supported routes:
1) "/projects/<project_id>/models"
2) "/projects/<project_id>/models/<expt_id>"
3) "/projects/<project_id>/models/<expt_id>/<run_id>"
"""

api.add_namespace(
    model_ns,
    path="/projects/<project_id>/models"
)

###############################
# Optimization management API #
###############################
"""
Supported routes:
1) "/projects/<project_id>/models/<expt_id>/optimizations/"
"""

api.add_namespace(
    optim_ns,
    path="/projects/<project_id>/models/<expt_id>/optimizations"
)