#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
from datetime import datetime

# Libs
import jsonschema
from flask import jsonify, request
from flask_restx import Namespace, Resource, fields
from tinydb import Query, where
from tinyrecord import transaction

# Custom
#from config import schemas, database, server_params
from rest_rpc import app

schemas = app.config['SCHEMAS']

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "participants", 
    description='API to faciliate participant management in a PySyft Grid.'
)

SUBJECT_TABLE = "Participant"

PAYLOAD_TEMPLATE = {'success': 0, 'id': 0, 'type': 'participants', 'data': None}

##########
# Models #
##########

# Models are used for marshalling (i.e. moulding responses)
"""
participant_model = ns_api.model(
    name="participant",
    model={
        'participant_id': fields.String(required=True),
        'configurations': fields.Nested(
            ns_api.model(
                name="",
                model={}
            ),
            required=True
        ),
        'created_at': fields.String()
    }
)
"""
participant_model = ns_api.schema_model(
    name='participant',
    schema=schemas['participant_schema']
)

# Marshalling success payload
payload_model = ns_api.model(
    name='payload',
    model={
        'success': fields.Integer(required=True),
        'id': fields.Integer(required=True),
        'type': fields.String(required=True),
        'data': fields.Nested(participant_model)
    }
)

#############
# Resources #
#############

@ns_api.route("/")
class Participants(Resource):
    """ Handles the entire collection of participants as a catalogue """

    @ns_api.doc("list_participants")
    #@ns_api.marshal_list_with(payload_model)
    def get(self):
        """ Retrieve all general metadata for each registered participant.
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            with database as db:
                participant_table = db.table(SUBJECT_TABLE)
                data = [participant for participant in iter(participant_table)]

            payload['data'] = data
            payload['success'] = 1

            return payload, 200
        
        except Exception:
            ns_api.abort(500)        

    @ns_api.doc("register_participant")
    @ns_api.expect(participant_model)
    #@ns_api.marshal_with(payload_model)
    @ns_api.response(201, "New project created!")
    @ns_api.response(417, "Insufficient participant configurations passed!")
    def post(self):
        """ Takes in a participant's host machine configuration and stores it
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            new_participant = request.json
            jsonschema.validate(new_participant, schemas['participant_schema'])

            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_participant['created_at'] = date_created

            new_participant['projects'] = []

            with database as db:
                participant_table = db.table(SUBJECT_TABLE)

                doc_id = participant_table.upsert(
                    new_participant, 
                    where('participant_id') == new_participant['participant_id']
                )[0]

                participant = participant_table.get(doc_id=doc_id)
            
            payload['id'] = doc_id
            payload['data'] = participant
            payload['success'] = 1
            
            return payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(417)

@ns_api.route("/<participant_id>")
@ns_api.param('participant_id', 'The participant identifier')
@ns_api.response(404, 'Project not found')
@ns_api.response(500, 'Internal failure')
class Participant(Resource):
    """ Handles all participant interactions with the TTP. Such interactions
        involve viewing registry, registering for, updating particulars and 
        de-eregistering from available projects.
    """

    @ns_api.doc("get_participant")
    #@ns_api.marshal_with(payload_model)
    def get(self, participant_id):
        """ Retrieves all metadata registered by a participant
        """
        payload = PAYLOAD_TEMPLATE.copy()

        with database as db:
            participant_table = db.table(SUBJECT_TABLE)

            Participant = Query()
            participant = participant_table.get(
                where('participant_id') == participant_id
            )

        if participant:
            payload['id'] = participant.doc_id
            payload['data'] = participant
            payload['success'] = 1
            return payload, 200

        else:
            ns_api.abort(404)

    @ns_api.doc("update_participant")
    @ns_api.expect(participant_model)
    #@ns_api.marshal_with(participant_model)
    def put(self, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            participant_updates = request.json
            jsonschema.validate(
                participant_updates, 
                schemas['participant_schema']
            )

            with database as db:
                participant_table = db.table(SUBJECT_TABLE)

                doc_id = participant_table.update(
                    participant_updates, 
                    where('participant_id') == participant_id
                )[0]
                participant = participant_table.get(doc_id=doc_id)

            if participant:
                payload['id'] = doc_id
                payload['data'] = participant
                payload['success'] = 1
                return payload, 200

            else:
                ns_api.abort(404)

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(417)

    @ns_api.doc("delete_participant")
    #@ns_api.marshal_with(participant_model)
    def delete(self, participant_id):
        """ De-registers participant from previously registered experiment(s),
            and clears out all their data
        """
        payload = PAYLOAD_TEMPLATE.copy()

        with database as db:
            participant_table = db.table(SUBJECT_TABLE)
            project_table = db.table('Project')

            participant = participant_table.get(
                where('participant_id') == participant_id
            )

            if participant:
        
                registered_projects = participant['projects']

                # Remove participant from all registered projects
                for project_id in registered_projects:
                    project = project_table.get(where('project_id') == project_id)
                    project['participants'] = [
                        p_id 
                        for p_id in project['participants']
                        if p_id != participant_id
                    ]
                    project_table.update(project, where('project_id') == project_id)

                payload['id'] = participant.doc_id
                payload['data'] = participant
                payload['success'] = 1
                return payload, 200

            else:
                ns_api.abort(404)


@ns_api.route("/<participant_id>/projects")
class RegisteredProjects(Resource):
    
    def get(self, participant_id):
        pass

@ns_api.route("/<participant_id>/projects/<project_id>")
class RegisteredProject(Resource):

    def get(self, participant_id, project_id):
        pass

    def put(self, participant_id, project_id):
        pass

    def delete(self, participant_id, project_id):
        pass


@ns_api.route("/<participant_id>/projects/<project_id>/experiments")
class RegisteredExperiments(Resource):

    def get(self, participant_id, project_id):
        pass


@ns_api.route("/<participant_id>/projects/<project_id>/experiments/<expt_id>")
class RegisteredExperiment(Resource):

    def get(self, participant_id, project_id, expt_id):
        pass


@ns_api.route("/<participant_id>/projects/<project_id>/experiments/<expt_id>/run")
class RegisteredRuns(Resource):

    def get(self, participant_id, project_id, expt_id):
        pass


@ns_api.route("/<participant_id>/projects/<project_id>/experiments/<expt_id>/run/<run_id>")
class RegisteredRun(Resource):

    def get(self, participant_id, project_id, expt_id, run_id):
        pass