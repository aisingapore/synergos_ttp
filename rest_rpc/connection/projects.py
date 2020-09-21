#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
from flask import request, redirect, url_for
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload, ProjectRecords
from rest_rpc.connection.experiments import expt_output_model
from rest_rpc.connection.runs import run_output_model
from rest_rpc.connection.registration import (
    Registrations, 
    Registration, 
    registration_export_model
)
from rest_rpc.connection.tags import Tag, tag_output_model
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "projects", 
    description='API to faciliate project management in a PySyft Grid.'
)

SUBJECT = "Project" # table name

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

incentives_field = fields.Wildcard(fields.Raw()) # tentative
incentive_model = ns_api.model(
    name="incentives",
    model={"*": incentives_field}
)

project_model = ns_api.model(
    name="project",
    model={
        'action': fields.String(),
        'universe_alignment': fields.List(fields.String),
        'incentives': fields.Nested(
            model=incentive_model,
            #required=True,
            skip_none=True
        ),
        'start_at': fields.String()
    }
)

project_input_model = ns_api.inherit(
    "project_input",
    project_model,
    {'project_id': fields.String()}
)

project_output_model = ns_api.inherit(
    "project_output",
    project_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='project_relations',
                model={
                    'Experiment': fields.List(
                        fields.Nested(expt_output_model, skip_none=True)
                    ),
                    'Run': fields.List(
                        fields.Nested(run_output_model, skip_none=True)
                    ),
                    'Registration': fields.List(
                        fields.Nested(registration_export_model, skip_none=True)
                    ),
                    'Tag': fields.List(
                        fields.Nested(tag_output_model, skip_none=True)
                    ),
                    'Model': fields.List(
                        fields.Nested(model_output_model, skip_none=True)
                    ),
                    'Validation': fields.List(
                        fields.Nested(val_output_model, skip_none=True)
                    ),
                    'Prediction': fields.List(
                        fields.Nested(pred_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, project_output_model)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Projects(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_projects')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self):
        """ Retrieve all metadata for each registered project.
            Metadata here includes:
            1) List of participant_ids
            2) List of experiments queued
            3) Parameters for incentive mechanism (pending)
            4) Date created
        """
        project_records = ProjectRecords(db_path=db_path)
        all_relevant_projects = project_records.read_all()
        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="projects.get",
            params={},
            data=all_relevant_projects
        )
        return success_payload, 200

    @ns_api.doc('register_project')
    @ns_api.expect(project_input_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New project created!")
    @ns_api.response(417, "Insufficient project configurations passed!")
    def post(self):
        """ Takes in a project configuration, which includes its incentives,
            scheduled starting time, and stores it for use in orchestration
        """
        try:
            new_project_details = request.json
            project_id = new_project_details.pop('project_id')

            project_records = ProjectRecords(db_path=db_path)
            new_project = project_records.create(
                project_id=project_id, 
                details=new_project_details
            )
            retrieved_project = project_records.read(project_id=project_id)
            assert new_project.doc_id == retrieved_project.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="projects.post",
                params={},
                data=retrieved_project
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate project configurations passed!"
            )


@ns_api.route('/<project_id>')
@ns_api.param('project_id', 'The project identifier')
@ns_api.response(404, 'Project not found')
@ns_api.response(500, 'Internal failure')
class Project(Resource):
    """ Handles all TTP interactions for managing project registration & logging
        incentive schemes
    """

    @ns_api.doc('get_project')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, project_id):
        """ Retrieves all metadata describing specified project """
        project_records = ProjectRecords(db_path=db_path)
        retrieved_project = project_records.read(project_id=project_id)
        
        from pprint import pprint
        pprint(retrieved_project)
        
        if retrieved_project:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.get",
                params={'project_id': project_id},
                data=retrieved_project
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' does not exist!"
            )

    @ns_api.doc('update_project')
    @ns_api.expect(project_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, project_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            project_updates = request.json

            project_records = ProjectRecords(db_path=db_path)
            updated_project = project_records.update(
                project_id=project_id,
                updates=project_updates
            )
            retrieved_project = project_records.read(project_id=project_id)
            assert updated_project.doc_id == retrieved_project.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.put",
                params={'project_id': project_id},
                data=retrieved_project
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate project configurations passed!"
            )

    @ns_api.doc('delete_project')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, project_id):
        """ De-registers all participants from previously registered 
            experiment(s), and removes the project
        """
        project_records = ProjectRecords(db_path=db_path)
        retrieved_project = project_records.read(project_id=project_id)
        deleted_project = project_records.delete(project_id=project_id)
        
        if deleted_project:
            assert deleted_project.doc_id == retrieved_project.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.delete",
                params={'project_id': project_id},
                data=retrieved_project
            )
            return success_payload

        else:
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' does not exist!"
            )


# Registered Participants
ns_api.add_resource(
    Registrations,
    '/<project_id>/registrations'
)

# Registered Registration
ns_api.add_resource(
   Registration, 
   '/<project_id>/participants/<participant_id>/registration'
)

# Registered Tags
ns_api.add_resource(
   Tag, 
   '/<project_id>/participants/<participant_id>/registration/tags'
)
