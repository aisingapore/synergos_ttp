#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
from flask import jsonify
from flask_restx import Namespace, Resource, fields, reqparse

# Custom
from config import schemas, database, server_params

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "runs", 
    description='API to faciliate run management in in a PySyft Grid.'
)

##########
# Models #
##########

# Models are used for marshalling (i.e. moulding responses)
run_model = ns_api.model(
    name="run",
    model={
        'run_id': fields.String(required=True),
        'arguments': fields.List(fields.String, required=True),
        'created_at': fields.String()
    }
)

#############
# Resources #
#############

@ns_api.route('/')
class Runs(Resource):
    """
    """

    @ns_api.doc("get_runs")
    @ns_api.marshal_list_with(run_model)
    def get(self, project_id, expt_id):
        """ Retrieve all run configurations queued for training
        """
        pass

    @ns_api.doc("register_run")
    def post(self, project_id, expt_id):
        """ Takes in a participant's host machine configuration, dataset 
            offerings & choice of project participation, and stores it
        """
        pass

@ns_api.route('/<run_id>')
class Run(Resource):
    """ Handles all TTP interactions for managing project registration & logging
        incentive schemes
    """

    @ns_api.doc("get_run")
    def get(self, project_id, expt_id):
        """ Retrieves all experiments registered under a project
        """
        pass

    @ns_api.doc("update_run")
    def put(self, project_id, expt_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        pass

    @ns_api.doc("delete_run")
    def delete(self, project_id, expt_id):
        """ De-registers participant from previously registered experiment(s),
            and clears out all their data
        """
        pass
