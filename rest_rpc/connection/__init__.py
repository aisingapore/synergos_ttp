#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from rest_rpc.connection.projects import ns_api as project_ns
from rest_rpc.connection.experiments import ns_api as experiment_ns
from rest_rpc.connection.runs import ns_api as run_ns
from rest_rpc.connection.participants import ns_api as participant_ns
from rest_rpc.connection.registration import ns_api as registration_ns
from rest_rpc.connection.tags import ns_api as tag_ns

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

###########################
# Project management APIs #
###########################
"""
Supported routes:
1) "/projects"
2) "/projects/<project_id>"
3) "/projects/<project_id>/participants"
4) "/projects/<project_id>/participants/<participant_id>"
5) "/projects/<project_id>/participants/<participant_id>/registration       ***
6) "/projects/<project_id>/participants/<participant_id>/registration/tags" ***

***: imported
"""
api.add_namespace(project_ns, path="/projects")

##############################
# Experiment management APIs #
##############################
"""
Supported routes:
1) "/projects/<project_id>/experiments"
2) "/projects/<project_id>/experiments/<expt_id>"
"""
api.add_namespace(experiment_ns, path="/projects/<project_id>/experiments")

#######################
# Run management APIs #
#######################
"""
Supported routes:
1) "/projects/<project_id>/experiments/<expt_id>/run"
2) "/projects/<project_id>/experiments/<expt_id>/run/<run_id>"
"""
api.add_namespace(run_ns, path="/projects/<project_id>/experiments/<expt_id>/runs")

###############################
# Participant management APIs #
###############################
"""
Supported routes:
1) "/participants"
2) "/participants/<participant_id>"
3) "/participants/<participant_id>/projects"
4) "/participants/<participant_id>/projects/<project_id>"
5) "/participants/<participant_id>/projects/<project_id>/registration"      ***
6) "/participants/<participant_id>/projects/<project_id>/registration/tags" ***

***: imported
"""
api.add_namespace(participant_ns, path="/participants")

################################
# Registration management APIs #
################################
"""
The resources defined under this section are used as imports into other
namespaces (i.e. projects & participants). Hence they do not support direct
routing. Instead, all supported routes can be found in their respective hosts. 
"""
api.add_namespace(registration_ns)
api.add_namespace(tag_ns)