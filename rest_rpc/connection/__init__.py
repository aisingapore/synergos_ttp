#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from rest_rpc.connection.collaborations import ns_api as collab_ns
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
    title="Synergos Orchestrator REST-RPC Connection API", 
    description="API to facilitate metadata collection between TTP & participants for WS connection"
)

#################################
# Collaboration management APIs #
#################################
"""
Supported routes:
1) "/collaborations"
2) "/collaborations/<collab_id>"
3) "/collaborations/<collab_id>/registrations"                              ***
4) "/collaborations/<collab_id>/participants/<participant_id>/registrations ***

***: Imported
"""
api.add_namespace(collab_ns, path="/collaborations")

###########################
# Project management APIs #
###########################
"""
Supported routes:
1) "/collaborations/<collab_id>/projects"
2) "/collaborations/<collab_id>/projects/<project_id>"
3) "/collaborations/<collab_id>/projects/<project_id>/registrations"                                   ***
4) "/collaborations/<collab_id>/projects/<project_id>/participants/<participant_id>/registration       ***
5) "/collaborations/<collab_id>/projects/<project_id>/participants/<participant_id>/registration/tags" ***

***: imported
"""
api.add_namespace(project_ns, path="/collaborations/<collab_id>/projects")

##############################
# Experiment management APIs #
##############################
"""
Supported routes:
1) "/collaborations/<collab_id>/projects/<project_id>/experiments"
2) "/collaborations/<collab_id>/projects/<project_id>/experiments/<expt_id>"
"""
api.add_namespace(experiment_ns, path="/collaborations/<collab_id>/projects/<project_id>/experiments")

#######################
# Run management APIs #
#######################
"""
Supported routes:
1) "/collaborations/<collab_id>/projects/<project_id>/experiments/<expt_id>/run"
2) "/collaborations/<collab_id>/projects/<project_id>/experiments/<expt_id>/run/<run_id>"
"""
api.add_namespace(run_ns, path="/collaborations/<collab_id>/projects/<project_id>/experiments/<expt_id>/runs")

###############################
# Participant management APIs #
###############################
"""
Supported routes:
1) "/participants"
2) "/participants/<participant_id>"
3) "/participants/<participant_id>/registrations"                                                      ***
4) "/participants/<participant_id>/collaborations/<collab_id>/registrations"                           ***
5) "/participants/<participant_id>/collaborations/<collab_id>/projects/<project_id>/registration"      ***
6) "/participants/<participant_id>/collaborations/<collab_id>/projects/<project_id>/registration/tags" ***

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