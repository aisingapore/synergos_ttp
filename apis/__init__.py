#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask_restx import Api

# Custom
from .connection import project_ns, experiment_ns, run_ns, participant_ns
#from .training import ns_api as ns2
#from .prediction import ns_api as ns3

##################
# Configurations #
##################

api = Api(
    version="1.0",
    title="PySyft WSSW Controller API", 
    description="Controller API to facilitate model training in a PySyft grid",
)

api.add_namespace(participant_ns, path="/participants")
api.add_namespace(project_ns, path="/projects")
api.add_namespace(experiment_ns, path="/projects/<project_id>/experiments")
api.add_namespace(run_ns, path="/projects/<project_id>/experiments/<expt_id>/runs")
