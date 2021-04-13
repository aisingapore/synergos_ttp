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
from rest_rpc.connection.projects import project_output_model
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
from synarchive.connection import CollaborationRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "collaborations", 
    description='API to faciliate collaboration management in a Synergos Grid.'
)

db_path = app.config['DB_PATH']
collab_records = CollaborationRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/collaborations.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

logger_ports_model = ns_api.model(
    name="logger_ports",
    model={
        'sysmetrics': fields.Integer(),
        'director': fields.Integer(),
        'ttp': fields.Integer(),
        'worker': fields.Integer(),
    }
)

collab_model = ns_api.model(
    name="collaboration",
    model={
        # Catalogue Connection
        'catalogue_host': fields.String(),
        'catalogue_port': fields.Integer(),
        # Logger Connection
        'logger_host': fields.String(),
        'logger_ports': fields.Nested(
            model=logger_ports_model,
            skip_none=True
        ),
        # Meter Connection
        'meter_host': fields.String(),
        'meter_port': fields.Integer(),
        # MLOps Connection
        'mlops_host': fields.String(),
        'mlops_port': fields.Integer(),
        # MQ Connection
        'mq_host': fields.String(),
        'mq_port': fields.Integer(),
        # UI Connection
        'ui_host': fields.String(),
        'ui_port': fields.Integer()
    }
)

collab_input_model = ns_api.inherit(
    "collab_input",
    collab_model,
    {'collab_id': fields.String()}
)

collab_output_model = ns_api.inherit(
    "collab_output",
    collab_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='collab_relations',
                model={
                    'Project': fields.List(
                        fields.Nested(project_output_model, skip_none=True)
                    ),
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
    subject=collab_records.subject, 
    namespace=ns_api, 
    model=collab_output_model
)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Collaborations(Resource):
    """ Handles the entire collection of collaborations as a catalogue """

    @ns_api.doc('list_collaborations')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self):
        """ Retrieve all metadata for each registered collaboration.
            Metadata here includes:
            1) Catalogue Host + Port
            2) Logger Host + Ports
            3) Meter Host + Port
            4) MLOps Host + Port
            5) MQ Host + Port
            6) UI Host + Port
            7) Date created
        """
        all_relevant_collabs = collab_records.read_all()

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="collaborations.get",
            params={},
            data=all_relevant_collabs
        )

        logging.info(
            "Collaborations: Bulk record retrieval successful!",
            code=200, 
            description="Successfully retrieved metadata for collaborations!",
            ID_path=SOURCE_FILE,
            ID_class=Collaborations.__name__, 
            ID_function=Collaborations.get.__name__,
            **request.view_args
        )

        return success_payload, 200


    @ns_api.doc('register_collaboration')
    @ns_api.expect(collab_input_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New collaboration created!")
    @ns_api.response(417, "Insufficient collaboration configurations passed!")
    def post(self):
        """ Takes in a collaboration configuration, which includes all
            component IPs & Port allocations and commencement date, and stores
            them for use in orchestration
        """
        # try:
        new_collab_details = request.json
        collab_id = new_collab_details.pop('collab_id')

        collab_records.create(
            collab_id=collab_id, 
            details=new_collab_details
        )
        retrieved_collab = collab_records.read(collab_id=collab_id)

        success_payload = payload_formatter.construct_success_payload(
            status=201, 
            method="collaborations.post",
            params={},
            data=retrieved_collab
        )

        logging.info(
            f"Collaboration '{collab_id}': Record creation successful!",
            code=201,
            description=f"Collaboration '{collab_id}' was successfully submitted!",
            ID_path=SOURCE_FILE,
            ID_class=Collaborations.__name__, 
            ID_function=Collaborations.post.__name__,
            **request.view_args
        )

        return success_payload, 201

        # except jsonschema.exceptions.ValidationError:
        #     logging.error(
        #         f"Collaboration '{collab_id}': Record creation failed.",
        #         code=417,
        #         description="Inappropriate collaboration configurations passed!", 
        #         ID_path=SOURCE_FILE,
        #         ID_class=Collaborations.__name__, 
        #         ID_function=Collaborations.post.__name__,
        #         **request.view_args
        #     )
        #     ns_api.abort(
        #         code=417,
        #         message="Inappropriate collaboration configurations passed!"
        #     )



@ns_api.route('/<collab_id>')
@ns_api.param('collab_id', 'The collaboration identifier')
@ns_api.response(404, 'Collaboration not found')
@ns_api.response(500, 'Internal failure')
class Collaboration(Resource):
    """ Handles all TTP interactions for managing collaboration registration &
        logging component connections
    """

    @ns_api.doc('get_collaboration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id):
        """ Retrieves all metadata describing specified project """
        retrieved_collab = collab_records.read(collab_id=collab_id)
                
        if retrieved_collab:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="collaboration.get",
                params={'collab_id': collab_id},
                data=retrieved_collab
            )

            logging.info(
                f"Collaboration '{collab_id}': Single record retrieval successful!",
                code=200, 
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}': Single record retrieval failed!",
                code=404, 
                description=f"Collaboration '{collab_id}' does not exist!", 
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Collaboration '{collab_id}' does not exist!"
            )


    @ns_api.doc('update_collaboration')
    @ns_api.expect(collab_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id):
        """ Updates a participant's specified collaboration choices IF & ONLY 
            IF his/her registered experiments have not yet commenced
        """
        try:
            collab_updates = request.json

            collab_records.update(
                collab_id=collab_id,
                updates=collab_updates
            )
            retrieved_collab = collab_records.read(collab_id=collab_id)

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="collaboration.put",
                params={'collab_id': collab_id},
                data=retrieved_collab
            )

            logging.info(
                f"Collaboration '{collab_id}': Record update successful!",
                code=200,
                description=f"Collaboration '{collab_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.put.__name__,
                **request.view_args
            )

            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}': Record update failed.",
                code=417, 
                description="Inappropriate collaboration configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate collaboration configurations passed!"
            )


    @ns_api.doc('delete_collaboration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id):
        """ De-registers all participants from previously registered 
            project(s), and removes the collaboration
        """
        retrieved_collab = collab_records.read(collab_id=collab_id)
        deleted_collab = collab_records.delete(collab_id=collab_id)
        
        if deleted_collab:

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="collaboration.delete",
                params={'collab_id': collab_id},
                data=retrieved_collab
            )

            logging.info(
                f"Collaboration '{collab_id}': Record deletion successful!",
                code=200, 
                description=f"Collaboration '{collab_id}' was successfully deleted!",
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.delete.__name__,
                **request.view_args
            )

            return success_payload

        else:
            logging.error(
                f"Collaboration '{collab_id}': Record deletion failed.", 
                code=404, 
                description=f"Collaboration '{collab_id}': does not exist!", 
                ID_path=SOURCE_FILE,
                ID_class=Collaboration.__name__, 
                ID_function=Collaboration.delete.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Collaboration '{collab_id}' does not exist!"
            )


#######################
# Inherited Resources #
#######################

### Registration Routing ###

# Accesses all registrations submitted under a single collaboration
ns_api.add_resource(
    Registrations,
    '/<collab_id>/registrations'
)

# Accesses all registrations for a single participant submitted under a single collaboration
ns_api.add_resource(
    Registrations,
    '/<collab_id>/participants/<participant_id>/registrations'
)
