#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os

# Libs
import jsonschema
from flask import request, redirect, url_for
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from synarchive.connection import TagRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "tags", 
    description='API to faciliate tag registration in in a PySyft Grid.'
)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
tag_records = TagRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/tags.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

tag_model = ns_api.model(
    name="tags",
    model={
        'train': fields.List(fields.List(fields.String()), required=True),
        'evaluate': fields.List(fields.List(fields.String())),
        'predict': fields.List(fields.List(fields.String())),
        'model': fields.List(fields.String()),
        'hyperparameters': fields.List(fields.String())
    }
)

# No need for input model since no additional input from used required to
# generate a tag_id.

tag_output_model = ns_api.inherit(
    "tag_output",
    tag_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'participant_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(
    subject=tag_records.subject, 
    namespace=ns_api, 
    model=tag_output_model
)

#############
# Resources #
#############

@ns_api.response(404, 'Tag not found')
@ns_api.response(500, 'Internal failure')
class Tag(Resource):
    """ Handles all TTP/Participant interactions for registering data tags to be
        used for FL training.
    """
    
    @ns_api.doc("get_tag")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id, project_id, participant_id):
        """ Retrieves all tags registered for a participant under a project """
        retrieved_tag = tag_records.read(
            collab_id=collab_id,
            project_id=project_id, 
            participant_id=participant_id
        )

        if retrieved_tag:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="tag.get",
                params=request.view_args,
                data=retrieved_tag
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Single record retrieval successful!", 
                code=200, 
                description=f"Data tagging of '{participant_id}' under project '{project_id}' was successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Single record retrieval failed!",
                code=404, 
                description=f"Data tags do not exist for participant '{participant_id}'' under Project '{project_id}'!", 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Data tags does not exist for participant '{participant_id}'' under Project '{project_id}'!"
            )

    @ns_api.doc("register_tag")
    @ns_api.expect(tag_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New tag created!")
    @ns_api.response(417, "Inappropriate tag configurations passed!")
    def post(self, collab_id, project_id, participant_id):
        """ Takes in a set of data tags under a participant for a specified 
            project, and stores it 
        """
        try:
            new_tag_details = request.json

            tag_records.create(
                collab_id=collab_id,
                project_id=project_id, 
                participant_id=participant_id,
                details=new_tag_details
            )
            retrieved_tag = tag_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                participant_id=participant_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="tag.post",
                params=request.view_args,
                data=retrieved_tag
            )
            
            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record creation successful!", 
                description=f"Data tagging of '{participant_id}' under project '{project_id}' was successfully submitted!",
                code=201, 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.post.__name__,
                **request.view_args
            )
            
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record creation failed.",
                code=417,
                description="Inappropriate tag configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.post.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=417,
                message="Inappropriate run configurations passed!"
            )


    @ns_api.doc('update_tags')
    @ns_api.expect(tag_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id, project_id, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            tag_updates = request.json

            tag_records.update(
                collab_id=collab_id,
                project_id=project_id,
                participant_id=participant_id,
                updates=tag_updates
            )
            retrieved_tag = tag_records.read(
                collab_id=collab_id,
                project_id=project_id,
                participant_id=participant_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="tag.put",
                params=request.view_args,
                data=retrieved_tag
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record update successful!",
                code=200,
                description=f"Data tagging of '{participant_id}' under project '{project_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.put.__name__,
                **request.view_args
            )

            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record update failed.",
                code=417, 
                description="Inappropriate tag configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate tag configurations passed!"
            )


    @ns_api.doc('delete_tag')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id, project_id, participant_id):
        """ De-registers all tags registered 
        """
        retrieved_tag = tag_records.read(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )
        deleted_tag = tag_records.delete(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )
        
        if deleted_tag:

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="tag.delete",
                params=request.view_args,
                data=retrieved_tag
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record deletion successful!",
                code=200, 
                description=f"Data tagging of '{participant_id}' under project '{project_id}' was successfully deleted!", 
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.delete.__name__,
                **request.view_args
            )           
            
            return success_payload

        else:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Tag: Record deletion failed.", 
                code=404, 
                description=f"Data tags does not exist for participant '{participant_id}'' under Project '{project_id}'!",
                ID_path=SOURCE_FILE,
                ID_class=Tag.__name__, 
                ID_function=Tag.delete.__name__,
                **request.view_args
            )            
            ns_api.abort(
                code=404, 
                message=f"Data tags does not exist for participant '{participant_id}'' under Project '{project_id}'!"
            )

##############
# Deprecated #
##############      
"""
        'relations': fields.Nested(
            ns_api.model(
                name='tag_relations',
                model={
                    'Alignment': fields.List(
                        fields.Nested(alignment_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
"""