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
from rest_rpc.connection.core.utils import TopicalPayload, TagRecords

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "tags", 
    description='API to faciliate tag registration in in a PySyft Grid.'
)

SUBJECT = "Tag" # table name

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

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
                    'project_id': fields.String(),
                    'participant_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, tag_output_model)

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
    def get(self, project_id, participant_id):
        """ Retrieves all tags registered for a participant under a project """
        tag_records = TagRecords(db_path=db_path)
        retrieved_tag = tag_records.read(
            project_id=project_id, 
            participant_id=participant_id
        )

        if retrieved_tag:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="tag.get",
                params={
                    'project_id': project_id, 
                    'participant_id': participant_id 
                },
                data=retrieved_tag
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Tag does not exist for participant '{participant_id}'' under Project '{project_id}'!"
            )

    @ns_api.doc("register_tag")
    @ns_api.expect(tag_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New tag created!")
    @ns_api.response(417, "Inappropriate tag configurations passed!")
    def post(self, project_id, participant_id):
        """ Takes in a set of data tags under a participant for a specified 
            project, and stores it 
        """
        try:
            new_tag_details = request.json

            tag_records = TagRecords(db_path=db_path)
            new_tag = tag_records.create(
                project_id=project_id, 
                participant_id=participant_id,
                details=new_tag_details
            )
            retrieved_tag = tag_records.read(
                project_id=project_id, 
                participant_id=participant_id
            )
            assert new_tag.doc_id == retrieved_tag.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="tags.post",
                params={
                    'project_id': project_id, 
                    'participant_id': participant_id
                },
                data=retrieved_tag
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate run configurations passed!"
            )

    @ns_api.doc('update_tags')
    @ns_api.expect(tag_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, project_id, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            tag_updates = request.json

            tag_records = TagRecords(db_path=db_path)
            updated_tag= tag_records.update(
                project_id=project_id,
                participant_id=participant_id,
                updates=tag_updates
            )
            retrieved_tag = tag_records.read(
                project_id=project_id,
                participant_id=participant_id
            )
            assert updated_tag.doc_id == retrieved_tag.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="tag.put",
                params={
                    'project_id': project_id, 
                    'participant_id': participant_id
                },
                data=retrieved_tag
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate tag configurations passed!"
            )

    @ns_api.doc('delete_tag')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, project_id, participant_id):
        """ De-registers all tags registered 
        """
        tag_records = TagRecords(db_path=db_path)
        retrieved_tag = tag_records.read(
            project_id=project_id,
            participant_id=participant_id
        )
        deleted_tag = tag_records.delete(
            project_id=project_id,
            participant_id=participant_id
        )
        
        if deleted_tag:
            assert deleted_tag.doc_id == retrieved_tag.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.delete",
                params={
                    'project_id': project_id,                     
                    'participant_id': participant_id
                },
                data=retrieved_tag
            )
            return success_payload

        else:
            ns_api.abort(
                code=404, 
                message=f"Tag does not exist for participant '{participant_id}'' under Project '{project_id}'!"
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