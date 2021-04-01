#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os

# Libs
import jsonschema
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.connection.experiments import expt_output_model
from rest_rpc.connection.runs import run_output_model
from rest_rpc.connection.registration import (
    Registrations, 
    Registration, 
    registration_output_model#registration_export_model
)
from rest_rpc.connection.tags import Tag, tag_output_model
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model
from synarchive.connection import ProjectRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "projects", 
    description='API to faciliate project management in a PySyft Grid.'
)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/projects.py logged", Description="No Changes")

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
                    'collab_id': fields.String(),
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
                        fields.Nested(registration_output_model, skip_none=True)
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

payload_formatter = TopicalPayload(
    subject=project_records.subject, 
    namespace=ns_api, 
    model=project_output_model
)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Projects(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_projects')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, collab_id):
        """ Retrieve all metadata for each registered project.
            Metadata here includes:
            1) List of participant_ids
            2) List of experiments queued
            3) Parameters for incentive mechanism (pending)
            4) Date created
        """
        all_relevant_projects = project_records.read_all(
            filter={'collab_id': collab_id}
        )

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="projects.get",
            params=request.view_args,
            data=all_relevant_projects
        )

        logging.info(
            "Collaboration '{collab_id}' -> Projects: Bulk record retrieval successful!",
            code=200, 
            description="Successfully retrieved metadata for projects",
            ID_path=SOURCE_FILE,
            ID_class=Projects.__name__, 
            ID_function=Projects.get.__name__,
            **request.view_args
        )

        return success_payload, 200


    @ns_api.doc('register_project')
    @ns_api.expect(project_input_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New project created!")
    @ns_api.response(417, "Insufficient project configurations passed!")
    def post(self, collab_id):
        """ Takes in a project configuration, which includes its incentives,
            scheduled starting time, and stores it for use in orchestration
        """
        try:
            new_project_details = request.json
            project_id = new_project_details.pop('project_id')

            project_records.create(
                collab_id=collab_id,
                project_id=project_id, 
                details=new_project_details
            )
            retrieved_project = project_records.read(
                collab_id=collab_id,
                project_id=project_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="projects.post",
                params=request.view_args,
                data=retrieved_project
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record creation successful!",
                code=201,
                description=f"Projects '{project_id}' was successfully submitted!",
                ID_path=SOURCE_FILE,
                ID_class=Projects.__name__, 
                ID_function=Projects.post.__name__,
                **request.view_args
            )

            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record creation failed.",
                code=417,
                description="Inappropriate project configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Projects.__name__, 
                ID_function=Projects.post.__name__,
                **request.view_args
            )
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
    def get(self, collab_id, project_id):
        """ Retrieves all metadata describing specified project """
        retrieved_project = project_records.read(
            collab_id=collab_id,
            project_id=project_id
        )
                
        if retrieved_project:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.get",
                params=request.view_args,
                data=retrieved_project
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Single record retrieval successful!",
                code=200, 
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Single record retrieval failed!",
                code=404, 
                description=f"Project '{project_id}' does not exist!", 
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' does not exist!"
            )


    @ns_api.doc('update_project')
    @ns_api.expect(project_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id, project_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            project_updates = request.json

            project_records.update(
                collab_id=collab_id,
                project_id=project_id,
                updates=project_updates
            )
            retrieved_project = project_records.read(
                collab_id=collab_id,
                project_id=project_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.put",
                params=request.view_args,
                data=retrieved_project
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record update successful!",
                code=200,
                description=f"Project '{project_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.put.__name__,
                **request.view_args
            )

            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record update failed.",
                code=417, 
                description="Inappropriate participant configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate project configurations passed!"
            )


    @ns_api.doc('delete_project')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id, project_id):
        """ De-registers all participants from previously registered 
            experiment(s), and removes the project
        """
        retrieved_project = project_records.read(
            collab_id=collab_id,
            project_id=project_id
        )
        deleted_project = project_records.delete(
            collab_id=collab_id,
            project_id=project_id
        )
        
        if deleted_project:

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="project.delete",
                params=request.view_args,
                data=retrieved_project
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record deletion successful!",
                code=200, 
                description=f"Project '{project_id}' was successfully deleted!",
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.delete.__name__,
                **request.view_args
            )

            return success_payload

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}': Record deletion failed.", 
                code=404, 
                description=f"Project '{project_id}': does not exist!", 
                ID_path=SOURCE_FILE,
                ID_class=Project.__name__, 
                ID_function=Project.delete.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' does not exist!"
            )



#######################
# Inherited Resources #
#######################

### Registration Routing ###

# Accesses all registrations submitted for a single project under a specific collaboration
ns_api.add_resource(
    Registrations,
    '/<project_id>/registrations'
)

# Accesses a participant's registration for a single project under a specific collaboration
ns_api.add_resource(
   Registration, 
   '/<project_id>/participants/<participant_id>/registration'
)

### Tag Routing ###

# Accesses all participant's data tags submitted for a single project under a specific collaboration
ns_api.add_resource(
   Tag, 
   '/<project_id>/participants/<participant_id>/registration/tags'
)
