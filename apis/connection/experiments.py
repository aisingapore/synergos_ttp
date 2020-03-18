#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
from flask import jsonify, request
from flask_restx import Namespace, Resource, fields
from tinydb import Query

# Custom
from config import schemas, database, server_params

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "experiments", 
    description='API to faciliate experiment management in a PySyft Grid.'
)

SUBJECT_TABLE = "Experiment"

##########
# Models #
##########

# Models are used for marshalling (i.e. moulding responses)
expt_model = ns_api.model(
    name="experiment",
    model={
        'expt_id': fields.String(required=True),
        'model': fields.List(fields.String, required=True),
        'created_at': fields.String()
    }
)

#############
# Resources #
#############

@ns_api.route('/')
class Experiments(Resource):
    """
    """

    @ns_api.doc("get_experiments")
    @ns_api.marshal_list_with(expt_model)
    def get(self, project_id):
        """ Retrieve all run configurations queued for training
        """
        return project_id

    @ns_api.doc("register_experiment")
    @ns_api.expect(expt_model)
    @ns_api.marshal_with(expt_model)
    @ns_api.response(201, "New project created!")
    @ns_api.response(417, "Insufficient project configurations passed!")
    @ns_api.response(500, 'Internal failure')
    def post(self, project_id):
        """ Takes in a participant's host machine configuration, dataset 
            offerings & choice of project participation, and stores it
        """
        try:
            new_experiment = request.json
            jsonschema.validate(new_experiment, schemas['project_schema'])

            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_project['created_at'] = date_created

            with database as db:
                project_table = db.table(SUBJECT_TABLE)
                project_table.insert(new_project)

            # Append experiment id in project's experiment field

            payload['data'] = new_project
            payload['success'] = 1
            
            return payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(417)


@ns_api.route('/<expt_id>')
class Experiment(Resource):
    """ Handles all TTP interactions for managing experimental configuration.
        Such interactions involve listing, specifying, updating and cancelling 
        experiments.
    """

    @ns_api.doc("get_experiment")
    @ns_api.marshal_with(expt_model)
    def get(self, project_id, expt_id):
        """ Retrieves all experimental parameters corresponding to a specified
            project
        """
        return f"{project_id} - {expt_id}"

    @ns_api.doc("update_experiment")
    @ns_api.expect(expt_model)
    @ns_api.marshal_with(expt_model)
    def put(self, project_id, expt_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        pass

    @ns_api.doc("delete_experiment")
    @ns_api.marshal_with(expt_model)
    def delete(self, project_id, expt_id):
        """ De-registers participant from previously registered experiment(s),
            and clears out all their data
        """
        pass
