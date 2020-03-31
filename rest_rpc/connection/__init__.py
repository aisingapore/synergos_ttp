#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from .projects import ns_api as project_ns
from .participants import ns_api as participant_ns
from .experiments import ns_api as experiment_ns
from .runs import ns_api as run_ns

##################
# Configurations #
##################

blueprint = Blueprint('connections', __name__)

api = Api(
    app=blueprint,
    version="1.0",
    title="PySyft TTP REST-RPC Connection API", 
    description="API to facilitate metadata collection between TTP & participants for WS connection"
)

api.add_namespace(participant_ns, path="/participants")
api.add_namespace(project_ns, path="/projects")
api.add_namespace(experiment_ns, path="/projects/<project_id>/experiments")
api.add_namespace(run_ns, path="/projects/<project_id>/experiments/<expt_id>/runs")
