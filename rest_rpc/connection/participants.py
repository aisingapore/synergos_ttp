#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload, ParticipantRecords
from rest_rpc.connection.registration import (
    Registrations,
    Registration,
    registration_export_model
) 
from rest_rpc.connection.tags import Tag, tag_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "participants", 
    description='API to faciliate participant management in a PySyft Grid.'
)

SUBJECT = "Participant"

PAYLOAD_TEMPLATE = {'success': 0, 'id': 0, 'type': 'participants', 'data': None}

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

participant_model = ns_api.model(
    name="participant",
    model={
        'id': fields.String(required=True),
        'host': fields.String(required=True),
        'port': fields.Integer(required=True),
        'log_msgs': fields.Boolean(),
        'verbose': fields.Boolean(),
        'f_port': fields.Integer(required=True)
    }
)

participant_input_model = ns_api.inherit(
    "participant_input",
    participant_model,
    {'participant_id': fields.String()}
)

participant_output_model = ns_api.inherit(
    "participant_output",
    participant_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'participant_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='participant_relations',
                model={
                    'Registration': fields.List(
                        fields.Nested(registration_export_model, skip_none=True)
                    ),
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


payload_formatter = TopicalPayload(SUBJECT, ns_api, participant_output_model)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Participants(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_participants')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self):
        """ Retrieve all metadata for each registered participant.
            Metadata here includes:
            1) Worker ID
            2) IP
            3) port
            4) Log_msgs (boolean switch to toggle logs)
            5) verbose  (boolean switch to toggle verbosity)
        """
        participant_records = ParticipantRecords(db_path=db_path)
        all_relevant_participants = participant_records.read_all()
        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="participants.get",
            params={},
            data=all_relevant_participants
        )
        return success_payload, 200

    @ns_api.doc('register_participant')
    @ns_api.expect(participant_input_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New participant created!")
    @ns_api.response(417, "Insufficient participant configurations passed!")
    def post(self):
        """ Takes in a participant's server configurations, and stores it for 
            subsequent use in orchestration
        """
        try:
            new_participant_details = request.json
            participant_id = new_participant_details['id']

            participant_records = ParticipantRecords(db_path=db_path)
            new_participant = participant_records.create(
                participant_id=participant_id, 
                details=new_participant_details
            )
            retrieved_participant = participant_records.read(
                participant_id=participant_id
            )
            assert new_participant.doc_id == retrieved_participant.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="participants.post",
                params={},
                data=retrieved_participant
            )
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(
                code=417,
                message="Inappropriate participant configurations passed!"
            )

@ns_api.route('/<participant_id>')
@ns_api.param('participant_id', 'The participant identifier')
@ns_api.response(404, 'Participant not found')
@ns_api.response(500, 'Internal failure')
class Participant(Resource):
    """ Handles all participant interactions for logging individual server 
        configurations
    """

    @ns_api.doc('get_participant')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, participant_id):
        """ Retrieves all metadata describing specified project """
        participant_records = ParticipantRecords(db_path=db_path)
        retrieved_participant = participant_records.read(
            participant_id=participant_id
        )

        if retrieved_participant:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="participant.get",
                params={'participant_id': participant_id},
                data=retrieved_participant
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' does not exist!"
            )

    @ns_api.doc('update_participant')
    @ns_api.expect(participant_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            participant_updates = request.json

            participant_records = ParticipantRecords(db_path=db_path)
            updated_participant = participant_records.update(
                participant_id=participant_id,
                updates=participant_updates
            )
            retrieved_participant = participant_records.read(
                participant_id=participant_id
            )
            assert updated_participant.doc_id == retrieved_participant.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="participant.put",
                params={'participant_id': participant_id},
                data=retrieved_participant
            )
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            ns_api.abort(                
                code=417,
                message="Inappropriate participant configurations passed!"
            )

    @ns_api.doc('delete_participant')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, participant_id):
        """ Removes participant's account & registered interactions entirely
        """
        participant_records = ParticipantRecords(db_path=db_path)
        retrieved_participant = participant_records.read(
            participant_id=participant_id
        )
        deleted_participant = participant_records.delete(
            participant_id=participant_id
        )
        
        if deleted_participant:
            assert deleted_participant.doc_id == retrieved_participant.doc_id
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="participant.delete",
                params={'participant_id': participant_id},
                data=retrieved_participant
            )
            return success_payload

        else:
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' does not exist!"
            )

# Registered Participants
ns_api.add_resource(
    Registrations,
    '/<participant_id>/registrations'
)

# Registered Participants
ns_api.add_resource(
   Registration, 
   '/<participant_id>/projects/<project_id>/registration'
)

# Registered Tags
ns_api.add_resource(
   Tag, 
   '/<participant_id>/projects/<project_id>/registration/tags'
)
