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
from flask_restx import fields
from tinydb import TinyDB, Query, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tinyrecord import transaction
from tinydb_serialization import SerializationMiddleware
from tinydb_smartcache import SmartCacheTable

# Custom
from rest_rpc import app
from .datetime_serialization import DateTimeSerializer, TimeDeltaSerializer

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
payload_template = app.config['PAYLOAD_TEMPLATE']

############################################
# REST Response Formatting Class - Payload #
############################################

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
                        annotate_document(document, subject)
                        for document in records
                    ]
                    document['relations'][subject] = annotated_records
                return document

            encoded_document = encode_datetime_objects(document)
            annotated_document = annotate_document(encoded_document, kind)
            annotated_doc_and_relations = annotate_relations(annotated_document)
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

#####################################
# Base Data Storage Class - Records #s
#####################################

class Records:
    """ 
    Automates CRUD operations on a structured TinyDB database. Operations are
    atomicised using TinyRecord transactions, queries are smart cahced

    Attributes:
        db_path (str): Path to json source
    
    Args:
        db_path (str): Path to json source
        *subjects: All subject types pertaining to records
    """
    def __init__(self, db_path=db_path):
        self.db_path = db_path

    ###########
    # Helpers #
    ###########

    def load_database(self):
        """ Loads json source as a TinyDB database, configured to cache queries,
            I/O operations, as well as serialise datetimes objects if necessary.
            Subjects are initialised as tables of the database

        Returns:
            database (TinyDB)
        """
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
        serialization.register_serializer(TimeDeltaSerializer(), 'TinyDelta')

        database = TinyDB(
            path=self.db_path, 
            sort_keys=True,
            indent=4,
            separators=(',', ': '),
            storage=CachingMiddleware(serialization)
        )

        database.table_class = SmartCacheTable

        return database

    ##################
    # Core Functions #
    ##################

    def create(self, subject, key, new_record):
        """ Creates a new record in a specified subject table within database

        Args:  
            subject (str): Table to be operated on
            new_record (dict): Information for creating a new record
            key (str): Primary key of the current table
        Returns:
            New record added (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            with transaction(subject_table) as tr:

                # Remove additional digits (eg. microseconds)
                date_created = datetime.strptime(
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "%Y-%m-%d %H:%M:%S"
                )
                new_record['created_at'] = date_created

                if subject_table.contains(where(key) == new_record[key]):
                    tr.update(new_record, where(key) == new_record[key])

                else:
                    tr.insert(new_record)

            record = subject_table.get(where(key) == new_record[key])

        return record

    def read_all(self, subject):
        """ Retrieves all records in a specified table of the database

        Args:
            subject (str): Table to be operated on
        Returns:
            Records (list(tinydb.database.Document))
        """
        database = self.load_database()

        with database as db:
            subject_table = db.table(subject)
            records = [record for record in iter(subject_table)]

        return records

    def read(self, subject, key, r_id):
        """ Retrieves a single record from a specified table in the database

        Args:  
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (str): Identifier of specified records
        Returns:
            Specified record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:
            subject_table = db.table(subject)
            record = subject_table.get(where(key) == r_id)

        return record

    def update(self, subject, key, r_id, updates):
        """ Updates an existing record with specified updates

        Args:  
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (str): Identifier of specified records
            updates (dict): New key-value pairs to update existing record with
        Returns:
            Updated record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            with transaction(subject_table) as tr:

                tr.update(updates, where(key) == r_id)

            updated_record = subject_table.get(where(key) == r_id)

        return updated_record
        
    def delete(self, subject, key, r_id):
        """ Deletes a specified record from the specified table in the database

        Args:
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (str): Identifier of specified records
        Returns:
            Deleted record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            record = subject_table.get(where(key) == r_id)

            with transaction(subject_table) as tr:

                tr.remove(where(key) == r_id)
            
            assert not subject_table.get(where(key) == r_id)
        
        return record

#######################################
# Data Storage Class - TopicalRecords #
#######################################

class TopicalRecords(Records):
    """
    Args:
        subject (str): Main subject type of records
        identifier (str): Identifying key of record
        db_path (str): Path to json source
        *relations (list(str)): All subjects related to specified subject
    """

    def __init__(self, subject, identifier, db_path=db_path, *relations):
        self.subject = subject
        self.identifier = identifier
        self.relations = relations
        super().__init__(db_path=db_path)

    def __get_related_metadata(self, r_id):
        """ Retrieves all related records from specified relations
        
        Args:
            r_id (dict(str, str)): 
                Record Identifier implemented as a composite collection of keys
        Returns:
            Collection of all related records (dict(str,list(Document)))
        """
        database = self.load_database()

        with database as db:

            all_related_records = {}
            for subject in self.relations:
                related_table = db.table(subject)
                related_records = related_table.search(
                    where('key')[self.identifier] == r_id[self.identifier]
                )
                all_related_records[subject] = related_records

        return all_related_records

    def __expand_record(self, record):
        """ Adds additional metadata from related subjects to specified record

        Args:
            record (tinydb.database.Document): Record to be expanded
        Returns:
            Expanded record (tinydb.database.Document)
        """
        r_id = record['key']
        related_records = self.__get_related_metadata(r_id)
        record['relations'] = related_records
        return record

    def create(self, new_record):
        return super().create(self.subject, "key", new_record)

    def read_all(self, filter=None):
        """ Retrieves entire collection of records, with an option to filter out
            ones with specific key-value pairs.

        Args:
            filter (dict(str,str)): Key-value pairs for filtering records
        Returns:
            Filtered records (list(tinydb.database.Document))
        """
        all_records = super().read_all(self.subject)
        expanded_records = []
        for record in all_records:
            #  expanded_records = [self.__expand_record(r) for r in all_records]
            if filter and not filter.items() <= record.items():
                pass
            exp_record = self.__expand_record(record)
            expanded_records.append(exp_record)
        return expanded_records

    def read(self, r_id):
        main_record = super().read(self.subject, "key", r_id)
        if main_record:
            return self.__expand_record(main_record)
        return main_record

    def update(self, r_id, updates):
        return super().update(self.subject, "key", r_id, updates)

    def delete(self, r_id):
        """ Uses composite keys for efficient cascading deletion of child 
            relations in related subjects

        Args:
            r_id (dict(str, str)): 
                Record Identifier implemented as a composite collection of keys
        Returns:
            Deleted record + related records deleted (dict)
        """
        database = self.load_database()

        with database as db:

            # Archive record targeted for deletion for output
            subject_table = db.table(self.subject)
            main_record = subject_table.get(where('key') == r_id)
            expanded_record = self.__expand_record(main_record)

            for subject in self.relations:
                related_table = db.table(subject)

                # Perform cascading deletion of all involved relations
                with transaction(related_table) as related_tr:
                    related_tr.remove(
                        where('key')[self.identifier] == r_id[self.identifier]
                    )

                ###################################################
                # Check that only the correct records are deleted #
                ###################################################

                related_records = expanded_record['relations'][subject]
                for r_record in related_records:
                    assert related_table.get(doc_id=r_record.doc_id) is None
                    assert r_record['key'][self.identifier] == r_id[self.identifier]

            # Finally, delete main entry itself
            with transaction(subject_table) as main_tr:
                main_tr.remove(where('key') == r_id)

            assert subject_table.get(doc_id=main_record.doc_id) is None
            assert main_record['key'] == r_id
        
        return main_record

#######################################
# Data Storage Class - ProjectRecords #
#######################################

class ProjectRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Project", 
            "project_id", 
            db_path,
            *["Tag", "Alignment", "Experiment", "Run"]
        )

    def __generate_key(self, project_id):
        return {"project_id": project_id}

    def create(self, project_id, details):
        # Check that new details specified conforms to project schema
        jsonschema.validate(details, schemas["project_schema"])
        project_key = self.__generate_key(project_id)
        new_project = {'key': project_key}
        new_project.update(details)
        return super().create(new_project)

    def read(self, project_id):
        project_key = self.__generate_key(project_id)
        return super().read(project_key)

    def update(self, project_id, updates):
        project_key = self.__generate_key(project_id)
        return super().update(project_key, updates)

    def delete(self, project_id):
        project_key = self.__generate_key(project_id)
        return super().delete(project_key)

###########################################
# Data Storage Class - ParticipantRecords #
###########################################

class ParticipantRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Participant", 
            "participant_id", 
            db_path,
            *["Tag", "Alignment"]
        )

    def __generate_key(self, participant_id):
        return {"participant_id": participant_id}

    def create(self, participant_id, details):
        # Check that new details specified conforms to project schema
        jsonschema.validate(details, schemas["participant_schema"])
        assert participant_id == details["id"] 
        participant_key = self.__generate_key(participant_id)
        new_participant = {'key': participant_key}
        new_participant.update(details)
        return super().create(new_participant)

    def read(self, participant_id):
        participant_key = self.__generate_key(participant_id)
        return super().read(participant_key)

    def update(self, participant_id, updates):
        participant_key = self.__generate_key(participant_id)
        return super().update(participant_key, updates)

    def delete(self, participant_id):
        participant_key = self.__generate_key(participant_id)
        return super().delete(participant_key)

##########################################
# Data Storage Class - ExperimentRecords #
##########################################

class ExperimentRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Experiment", 
            "expt_id", 
            db_path,
            *["Run"]
        )

    def __generate_key(self, project_id, expt_id):
        return {"project_id": project_id, "expt_id": expt_id}

    def create(self, project_id, expt_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["experiment_schema"])
        expt_key = self.__generate_key(project_id, expt_id)
        new_expt = {'key': expt_key}
        new_expt.update(details)
        return super().create(new_expt)

    def read(self, project_id, expt_id):
        expt_key = self.__generate_key(project_id, expt_id)
        return super().read(expt_key)

    def update(self, project_id, expt_id, updates):
        expt_key = self.__generate_key(project_id, expt_id)
        return super().update(expt_key, updates)

    def delete(self, project_id, expt_id):
        expt_key = self.__generate_key(project_id, expt_id)
        return super().delete(expt_key)

###################################
# Data Storage Class - RunRecords #
###################################

class RunRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            subject="Run",  
            identifier="run_id", 
            db_path=db_path
        )

    def __generate_key(self, project_id, expt_id, run_id):
        return {"project_id": project_id, "expt_id": expt_id, "run_id": run_id}

    def create(self, project_id, expt_id, run_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["run_schema"])
        run_key = self.__generate_key(project_id, expt_id, run_id)
        new_run = {'key': run_key}
        new_run.update(details)
        return super().create(new_run)

    def read(self, project_id, expt_id, run_id):
        run_key = self.__generate_key(project_id, expt_id, run_id)
        return super().read(run_key)

    def update(self, project_id, expt_id, run_id, updates):
        run_key = self.__generate_key(project_id, expt_id, run_id)
        return super().update(run_key, updates)

    def delete(self, project_id, expt_id, run_id):
        run_key = self.__generate_key(project_id, expt_id, run_id)
        return super().delete(run_key)

###############################################
# Data Storage Association class - TagRecords #
###############################################

class TagRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            "Tag",  
            "tag_id", 
            db_path,
            *["Alignment"]
        )

    def __generate_key(self, project_id, participant_id, tag_id):
        return {
            "project_id": project_id, 
            "participant_id": participant_id,
            "tag_id": tag_id
        }

    def create(self, project_id, participant_id, tag_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["tag_schema"])
        tag_key = self.__generate_key(project_id, participant_id, tag_id)
        new_tag = {'key': tag_key}
        new_tag.update(details)
        return super().create(new_tag)

    def read(self, project_id, participant_id, tag_id):
        run_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().read(run_key)

    def update(self, project_id, participant_id, tag_id, updates):
        run_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().update(run_key, updates)

    def delete(self, project_id, participant_id, tag_id):
        run_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().delete(run_key)

#####################################################
# Data Storage Association class - AlignmentRecords #
#####################################################

class AlignmentRecords(TopicalRecords):

    def __init__(self, db_path=db_path):
        super().__init__(
            subject="Alignment",  
            identifier="XXXX", # no primary identifier, just foreign keys
            db_path=db_path
        )

    def __generate_key(self, project_id, participant_id, tag_id):
        return {
            "project_id": project_id, 
            "participant_id": participant_id,
            "tag_id": tag_id
        }

    def create(self, project_id, participant_id, tag_id, details):
        # Check that new details specified conforms to experiment schema
        jsonschema.validate(details, schemas["alignment_schema"])
        alignment_key = self.__generate_key(project_id, participant_id, tag_id)
        new_alignment = {'key': alignment_key}
        new_alignment.update(details)
        return super().create(new_alignment)

    def read(self, project_id, participant_id, tag_id):
        alignment_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().read(alignment_key)

    def update(self, project_id, participant_id, tag_id, updates):
        alignment_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().update(alignment_key, updates)

    def delete(self, project_id, participant_id, tag_id):
        alignment_key = self.__generate_key(project_id, participant_id, tag_id)
        return super().delete(alignment_key)

