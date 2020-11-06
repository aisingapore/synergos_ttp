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
from rest_rpc.connection.core.utils import TopicalPayload, RegistrationRecords
from rest_rpc.connection.tags import tag_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "registration", 
    description='API to faciliate registation management in a PySyft Grid.'
)

SUBJECT = "Registration" # table name

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

role_model = ns_api.model(
    name="role",
    model={
        'role': fields.String(required=True)
    }
)

incentives_field = fields.Wildcard(fields.Raw()) # tentative
incentive_model = ns_api.model(
    name="registered_incentives",
    model={"*": incentives_field}
)

registration_model = ns_api.inherit(
    "registration",
    role_model,
    {
        'project':fields.Nested(
            ns_api.model(
                name="registered_project",
                model={
                    'incentives': fields.Nested(
                        model=incentive_model,
                        skip_none=True
                    ),
                    'start_at': fields.String()
                }
            ),
            required=True
        ),
        'participant': fields.Nested(
            ns_api.model(
                name="registered_participant",
                model={
                    'id': fields.String(required=True),
                    'host': fields.String(required=True),
                    'port': fields.Integer(required=True),
                    'f_port': fields.Integer(required=True),
                    'log_msgs': fields.Boolean(),
                    'verbose': fields.Boolean()
                }
            ),
            required=True
        )
    }
)

# No need for input model since no additional input from used required to
# generate a registration_id.

registration_output_model = ns_api.inherit(
    "registration_input",
    registration_model,
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
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='registration_relations',
                model={
                    'Tag': fields.List(
                        fields.Nested(tag_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

registration_export_model = ns_api.inherit(
    "registration_export",
    role_model,
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

payload_formatter = TopicalPayload(SUBJECT, ns_api, registration_output_model)

#############
# Resources #
#############

@ns_api.response(500, 'Internal failure')
class Registrations(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_registrations')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, **kwargs):
        """ Retrieve all metadata for each set of registrations """
        registration_records = RegistrationRecords(db_path=db_path)
        all_relevant_registration = registration_records.read_all(filter=kwargs)
        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="registrations.get",
            params={},
            data=all_relevant_registration
        )
        return success_payload, 200


@ns_api.response(404, 'Registration not found')
@ns_api.response(500, 'Internal failure')
class Registration(Resource):
    """ Handles all TTP & participant interactions for activity registration and
        role definition
    """

    @ns_api.doc('get_registration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, project_id, participant_id):
        """ Retrieves all metadata describing specified project """
        registration_records = RegistrationRecords(db_path=db_path)
        retrieved_registration = registration_records.read(
            project_id=project_id,
            participant_id=participant_id
        )

        if retrieved_registration:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.get",
                params={
                    'project_id': project_id, 
                    'participant_id':participant_id
                },
                data=retrieved_registration
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' has not registered for Project '{project_id}'!"
            )

    @ns_api.doc('register_registration')
    @ns_api.expect(role_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New registration entry created!")
    @ns_api.response(417, "Insufficient registration configurations passed!")
    def post(self, project_id, participant_id):
        """ Takes in configurations for a participant to complete registration 
            for a project, which includes its supposed role in the training grid
            and stores it for use in orchestration
        """
        try:
            new_registration_details = request.json

            registration_records = RegistrationRecords(db_path=db_path)
            new_registration = registration_records.create(
                project_id=project_id, 
                participant_id=participant_id,
                details=new_registration_details
            )
            retrieved_registration = registration_records.read(
                project_id=project_id,
                participant_id=participant_id
            )
            assert new_registration.doc_id == retrieved_registration.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="registration.post",
                params={},
                data=retrieved_registration
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate registration configurations passed!"
            )

    @ns_api.doc('update_registration')
    @ns_api.expect(role_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, project_id, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            registration_updates = request.json

            registration_records = RegistrationRecords(db_path=db_path)
            updated_registration = registration_records.update(
                project_id=project_id,
                participant_id=participant_id,
                updates=registration_updates
            )
            retrieved_registration = registration_records.read(
                project_id=project_id,
                participant_id=participant_id
            )
            assert updated_registration.doc_id == retrieved_registration.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.put",
                params={
                    'project_id': project_id, 
                    'participant_id': participant_id
                },
                data=retrieved_registration
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate registration configurations passed!"
            )

    @ns_api.doc('delete_registration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, project_id, participant_id):
        """ De-registers participant from project and removes all their metadata
        """
        registration_records = RegistrationRecords(db_path=db_path)
        retrieved_registration = registration_records.read(
            project_id=project_id,
            participant_id=participant_id
        )
        deleted_registration = registration_records.delete(
            project_id=project_id,
            participant_id=participant_id
        )
        
        if deleted_registration:
            assert deleted_registration.doc_id == retrieved_registration.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.delete",
                params={'project_id': project_id},
                data=retrieved_registration
            )
            return success_payload

        else:
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' has not registered for Project '{project_id}'!"
            )
