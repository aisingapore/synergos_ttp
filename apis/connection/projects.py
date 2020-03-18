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
from config import schemas, database, server_params

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "projects", 
    description='API to faciliate project management in a PySyft Grid.'
)

SUBJECT_TABLE = "Project"

PAYLOAD_TEMPLATE = {'success': 0, 'id': 0, 'type': 'projects', 'data': None}

##########
# Models #
##########

# Marshalling project data
project_model = ns_api.model(
    name="project",
    model={
        'project_id': fields.String(required=True),
        'participants': fields.List(fields.String, required=True),
        'experiments': fields.List(fields.String, required=True),
        #'incentives': fields.Nested(),
        'created_at': fields.String(),
        #'start_at': fields.String()
    }
)

# Marshalling participant data
participant_model = ns_api.schema_model(
    name="participant",
    schema=schemas['participant_schema']
)

# Marshalling success payload
payload_model = ns_api.model(
    name='payload',
    model={
        'success': fields.Integer(required=True),
        'id': fields.Integer(required=True),
        'type': fields.String(required=True),
        'data': fields.Nested(project_model)
    }
)

# Let Flask-restx handle the errors

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Projects(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_projects')
    @ns_api.marshal_list_with(payload_model)
    def get(self):
        """ Retrieve all metadata for each registered project.
            Metadata here includes:
            1) List of participant_ids
            2) List of experiments queued
            3) Parameters for incentive mechanism (pending)
            4) Date created
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            with database as db:
                project_table = db.table(SUBJECT_TABLE)
                data = [project for project in iter(project_table)]

            payload['data'] = data
            payload['success'] = 1

            return payload, 200
        
        except Exception:
            ns_api.abort(500)

    @ns_api.doc('register_project')
    @ns_api.expect(project_model) # for guiding payload
    @ns_api.marshal_with(payload_model)
    @ns_api.response(201, "New project created!")
    @ns_api.response(417, "Insufficient project configurations passed!")
    def post(self):
        """ Takes in a participant's host machine configuration, dataset 
            offerings & choice of project participation, and stores it
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            new_project = request.json
            jsonschema.validate(new_project, schemas['project_schema'])

            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_project['created_at'] = date_created

            with database as db:
                project_table = db.table(SUBJECT_TABLE)
                Project = Query()
                #with transaction(project_table) as tr:
                #    doc_id = tr.insert(new_project)
                doc_id = project_table.upsert(
                    new_project, 
                    Project.project_id == new_project['project_id']
                )[0]

                project = project_table.get(doc_id=doc_id)
            
            payload['id'] = doc_id
            payload['data'] = project
            payload['success'] = 1
            
            return payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(417)


@ns_api.route('/<project_id>')
@ns_api.param('project_id', 'The project identifier')
@ns_api.response(404, 'Project not found')
@ns_api.response(500, 'Internal failure')
class Project(Resource):
    """ 
    Handles all TTP interactions for managing project registration & logging
    incentive schemes
    """

    @ns_api.doc('get_project')
    @ns_api.marshal_with(payload_model)
    def get(self, project_id):
        """ Retrieves all metadata describing specified project 
        """
        payload = PAYLOAD_TEMPLATE.copy()

        with database as db:
            project_table = db.table(SUBJECT_TABLE)

            Project = Query()
            project = project_table.get(Project.project_id == project_id)

        if project:
            payload['id'] = project.doc_id
            payload['data'] = project
            payload['success'] = 1
            return payload, 200

        else:
            ns_api.abort(404)

    @ns_api.doc('update_project')
    @ns_api.expect(project_model)
    @ns_api.marshal_with(payload_model)
    def put(self, project_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            project_updates = request.json
            jsonschema.validate(project_updates, schemas['project_schema'])

            with database as db:
                project_table = db.table(SUBJECT_TABLE)

                #with transaction(project_table) as tr:
                #    tr.update(project_updates, where('project_id') == project_id)
                doc_id = project_table.update(
                    project_updates, where('project_id') == project_id
                )[0]
                project = project_table.get(doc_id=doc_id)

            if project:
                payload['id'] = doc_id
                payload['data'] = project
                payload['success'] = 1
                return payload, 200

            else:
                ns_api.abort(404)

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(417)
            
    @ns_api.doc('delete_project')
    #@ns_api.marshal_with(payload_model)
    def delete(self, project_id):
        """ De-registers participant from previously registered experiment(s),
            and clears out all their data
        """
        payload = PAYLOAD_TEMPLATE.copy()

        with database as db:
            project_table = db.table(SUBJECT_TABLE)
            experiment_table = db.table("Experiment")
            run_table = db.table("Run")

            #with transaction(project_table) as tr_proj:
            #    with transaction(experiment_table) as tr_expt:
            #        with transaction(run_table) as tr_run:

            project = project_table.get(where('project_id') == project_id)

            # Check if the project exists to be removed
            if project:

                removed_project_doc_ids = project_table.remove(
                    where('project_id') == project_id
                )

                # Retrieve metadata of experiments & runs to be removed
                related_expts = []
                related_runs = []
                for expt_id in project['experiments']:
                    experiment = experiment_table.get(
                        where('project_id') == project_id &
                        where('expt_id') == expt_id
                    )
                    related_expts.append(experiment)
                    for run_id in experiment['runs']:
                        run = run_table.get(
                            where('project_id') == project_id &
                            where('expt_id') == expt_id &
                            where('run_id') == run_id
                        )
                        related_runs.append(run)

                # Perform cascading deletion of experiments & runs 
                removed_runs_doc_ids = run_table.remove(
                    where('project_id') == project_id
                )
                removed_expts_doc_ids = experiment_table.remove(
                    where('project_id') == project_id
                )

                #################################################
                # Verify that removal operations are successful #
                #################################################
                assert (
                    len(removed_project_doc_ids) == 1 and 
                    removed_project_doc_ids[0] == project.doc_id
                )
                assert (
                    set(project['experiments']) == 
                    set([expt['expt_id'] for expt in related_expts]) and
                    set(removed_expts_doc_ids) ==
                    set([expt.doc_id for expt in related_expts])
                )

                related_run_ids = [run['run_id'] for run in related_runs]
                for expt in related_expts:
                    run_ids = expt['runs']
                    assert set(run_ids).issubset(set(related_run_ids))
                related_run_doc_ids = [run.doc_id for run in related_runs]
                assert set(removed_runs_doc_ids) == set(related_run_doc_ids)

                payload['id'] = project.doc_id
                payload['data'] = {
                    'project': project, 
                    'experiments': related_expts,
                    'runs': related_runs
                }
                payload['success'] = 1
                return payload, 200

            else:
                ns_api.abort(404)


@ns_api.route('/<project_id>/participants')
@ns_api.response(500, 'Internal failure')
class Participants(Resource):
    """ Handles all participants registered to specified project as a catalog
    """

    @ns_api.doc('list_project_participants')
    #@ns_api.marshal_list_with(payload_model)
    def get(self, project_id):
        """ Retrieves all metadata for all registered participants
        """
        try:
            payload = PAYLOAD_TEMPLATE.copy()

            with database as db:
                project_table = db.table(SUBJECT_TABLE)
                participant_table = db.table('Participant')

                project = project_table.get(where('project_id') == project_id)
                registered_participant_ids = project['participants']
                participants = [
                    participant_table.get(where('participant_id') == id)
                    for id in registered_participant_ids
                ]

            payload['type'] = 'participants'
            payload['data'] = participants
            payload['success'] = 1

            return payload, 200

        except Exception:
            ns_api.abort(500)

    @ns_api.doc('register_project_participant')
    #@ns_api.expect(participant_model) # for guiding payload
    #@ns_api.marshal_with(payload_model)
    @ns_api.response(201, "New participant registered!")
    @ns_api.response(417, "Insufficient participant configurations passed!")
    def post(self, project_id):
        """ Registers an EXISTING participant to an EXISTING project
        """
        payload = PAYLOAD_TEMPLATE.copy()




@ns_api.route('/<project_id>/participants/<participant_id>')
@ns_api.param('project_id', 'The project identifier')
@ns_api.param('participant_id', 'Identifier of a registered participant')
@ns_api.response(404, 'Participant not found. Check if exists or is registered')
@ns_api.response(500, 'Internal failure')
class Participant(Resource):


    @ns_api.doc('get_project')
    #@ns_api.marshal_with(payload_model)
    def get(self, project_id, participant_id):

        payload = PAYLOAD_TEMPLATE.copy()

    @ns_api.doc('update_project_participant')
    @ns_api.expect(participant_model)
    #@ns_api.marshal_with(payload_model)
    def put(self, project_id, participant_id):
        payload = PAYLOAD_TEMPLATE.copy()

    @ns_api.doc('delete_project')
    #@ns_api.marshal_with(payload_model)
    def delete(self, project_id, participant_id):
        payload = PAYLOAD_TEMPLATE.copy()