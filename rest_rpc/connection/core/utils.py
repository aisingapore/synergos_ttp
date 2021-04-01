#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os

# Libs
import jsonschema
from flask_restx import fields

# Custom
from rest_rpc import app
from synarchive.datetime_serialization import DateTimeSerializer

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
payload_template = app.config['PAYLOAD_TEMPLATE']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/core/utils.py logged", Description="No Changes")

###################################################
# REST Response Formatting Class - TopicalPayload #
###################################################

class TopicalPayload:
    """ Helper class to standardise response formatting for the REST-RPC service
        in order to ensure compatibility between the TTP's & Workers' Flask
        interfaces

    Attributes:
        # Private Attributes
        __template (dict): Configured payload template
        # Public Attributes
        subject (str): Topic of data in payload (i.e. name of table accessed)
    
    Args:
        subject (str): Topic of data in payload (i.e. name of table accessed)
        namespace (flask_restx.Namespace): Namespace API to construct models in
        model (flask_restx.Model): Seeding model to propagate
    """
    def __init__(self, subject, namespace, model):
        self.__template = payload_template.copy()
        self.subject = subject

        payload_model = namespace.model(
            name="payload",
            model={
                'apiVersion': fields.String(required=True),
                'success': fields.Integer(required=True),
                'status': fields.Integer(required=True),
                'method': fields.String(),
                'params': fields.Nested(
                    namespace.model(
                        name="route_parameters",
                        model={
                            'collab_id': fields.String(),
                            'project_id': fields.String(),
                            'expt_id': fields.String(),
                            'run_id': fields.String(),
                            'participant_id': fields.String(),
                            'tag_id': fields.String(),
                            'alignment_id': fields.String()
                        }
                    ),
                    skip_none=True
                )
            }
        )
        self.singular_model = namespace.inherit(
            "payload_single",
            payload_model,
            {'data': fields.Nested(model, required=True, skip_none=True)}
        )
        self.plural_model = namespace.inherit(
            "payload_plural",
            payload_model,
            {
                'data': fields.List(
                    fields.Nested(model, skip_none=True), 
                    required=True
                )
            }
        )

    def construct_success_payload(self, status, method, params, data):
        """ Automates the construction & formatting of a payload for a
            successful endpoint operation 
        Args:
            status (int): Status code of method of operation
            method (str): Endpoint operation invoked
            params (dict): Identifiers required to start endpoint operation
            data (list or dict): Data to be moulded into a response
        Returns:
            Formatted payload (dict)
        """
        
        def format_document(document, kind):

            def encode_datetime_objects(document):
                datetime_serialiser = DateTimeSerializer()
                document['created_at'] = datetime_serialiser.encode(document['created_at'])
                return document
            
            def annotate_document(document, kind):
                document['doc_id'] = document.doc_id
                document['kind'] = kind
                return document

            def annotate_relations(document):
                for subject, records in document['relations'].items():
                    annotated_records = [
                        annotate_document(
                            encode_datetime_objects(document), 
                            subject
                        )
                        for document in records
                    ]
                    document['relations'][subject] = annotated_records
                return document

            encoded_document = encode_datetime_objects(document)
            annotated_document = annotate_document(encoded_document, kind)
            annotated_doc_and_relations = annotate_relations(annotated_document)
            
            logging.debug(
                "Annotated docs while constructing success payload tracked.", 
                annotated_docs=annotated_doc_and_relations, 
                ID_path=SOURCE_FILE,
                ID_class=TopicalPayload.__name__,
                ID_function=TopicalPayload.construct_success_payload.__name__
            )

            return annotated_doc_and_relations

        self.__template['success'] = 1
        self.__template['status'] = status
        self.__template['method'] = method
        self.__template['params'] = params
        
        if isinstance(data, list):
            formatted_data = []
            for record in data:
                formatted_record = format_document(record, kind=self.subject)
                formatted_data.append(formatted_record)
        else:
            formatted_data = format_document(data, kind=self.subject)
                
        self.__template['data'] = formatted_data

        jsonschema.validate(self.__template, schemas['payload_schema'])

        return self.__template
