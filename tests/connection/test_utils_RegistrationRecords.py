#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from datetime import datetime, timedelta
from pathlib import Path

# Libs
from tinydb import where

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    ProjectRecords,
    ParticipantRecords,
    RegistrationRecords, 
    TagRecords
)
from rest_rpc.training.core.utils import AlignmentRecords

##################
# Configurations #
##################

registration_details = {
    "role": "guest"
}

registration_updates = {
    "role": "host"
}

project_details = {
    "universe_alignment": [],
    "incentives": {},
    "start_at": datetime.strptime(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        '%Y-%m-%d %H:%M:%S'
    )
}

participant_details = {
    "id": "NYP",
    "host": "0.0.0.0",
    "port": 8020,
    "log_msgs": False,
    "verbose": False
}

tag_details = {
    "train": [["2018","interpolated_eicu_worker_0"]],
    "evaluate": [["2019", "interpolated_eicu_validation"]],
    "predict": [],
    "model": ["2018", "interpolated_model"],
    "hyperparameters": ["2018", "hyperparameters"]
}

alignment_details = {
    "train": {
        "X": [0, 1, 2],
        "y": []
    },
    "evaluate": {
        "X": [1, 3],
        "y": []
    },
    "predict": {
        "X": [4],
        "y": []
    }
}

project_id = "poly_collab"
participant_id = "NYP"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

####################################
# Association Evaluation Functions #
####################################

def activate_env():
    project_records = ProjectRecords(db_path=test_db_path)
    project_records.create(
        project_id=project_id, 
        details=project_details
    )
    participant_records = ParticipantRecords(db_path=test_db_path)
    participant_records.create(
        participant_id=participant_id, 
        details=participant_details
    )

def deactivate_env():
    project_records = ProjectRecords(db_path=test_db_path)
    project_records.delete(project_id=project_id)
    participant_records = ParticipantRecords(db_path=test_db_path)
    participant_records.delete(participant_id)

def check_equivalence_and_format(record):
    assert 'created_at' in record.keys()
    record.pop('created_at')
    assert "key" in record.keys()
    key = record.pop('key')
    assert set([project_id, participant_id]) == set(key.values())
    assert "link" in record.keys()
    link = record.pop('link')
    assert not set(link.items()).issubset(set(key.items()))
    return record 

def check_relation_equivalence_and_format(record):
    assert 'relations' in record.keys()
    relations = record.pop('relations')
    assert (set(relations.keys()) == set(["Project", "Participant", "Tag", "Alignment"]))
    return record 

def check_detail_equivalence(details):
    from pprint import pprint
    expanded_registration_details = registration_details.copy()
    if 'project' in details.keys():
        details['project'].pop('created_at')
        pprint(details['project'])
        expanded_registration_details.update({'project': project_details})
        expanded_registration_details['project']['key'] = {'project_id': project_id}
    if 'participant' in details.keys():
        details['participant'].pop('created_at')
        expanded_registration_details.update({'participant': participant_details})
        expanded_registration_details['participant']['key'] = {'participant_id': participant_id}
    assert details == expanded_registration_details
    return details

###################################
# RegistrationRecords Class Tests #
###################################

def test_RegistrationRecords_create():
    registration_records = RegistrationRecords(db_path=test_db_path)
    created_registration = registration_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=registration_details
    )
    raw_details = check_equivalence_and_format(created_registration)
    check_detail_equivalence(raw_details)


def test_RegistrationRecords_read_all():
    activate_env()
    registration_records = RegistrationRecords(db_path=test_db_path)
    all_registrations = registration_records.read_all()
    assert len(all_registrations) == 1
    retrieved_registration = all_registrations[0]
    trimmed_details = check_equivalence_and_format(retrieved_registration)
    raw_details = check_relation_equivalence_and_format(trimmed_details)
    check_detail_equivalence(raw_details)


def test_RegistrationRecords_read():
    registration_records = RegistrationRecords(db_path=test_db_path)
    retrieved_registration = registration_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    assert retrieved_registration is not None
    trimmed_details = check_equivalence_and_format(retrieved_registration)
    raw_details = check_relation_equivalence_and_format(trimmed_details)
    check_detail_equivalence(raw_details)


def test_RegistrationRecords_update():
    registration_records = RegistrationRecords(db_path=test_db_path)
    targeted_registration = registration_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    updated_registration = registration_records.update(
        project_id=project_id,
        participant_id=participant_id,
        updates=registration_updates
    )
    assert targeted_registration.doc_id == updated_registration.doc_id
    for k,v in registration_updates.items():
        assert updated_registration[k] == v


def test_RegistrationRecords_delete():
    # Register an tag under the current registration
    tag_records = TagRecords(db_path=test_db_path)
    created_tag = tag_records.create(
        project_id=project_id,
        participant_id=participant_id, 
        details=tag_details 
    )
    # Register an alignment under tag
    alignment_records = AlignmentRecords(db_path=test_db_path)
    created_alignment = alignment_records.create(
        project_id=project_id,
        participant_id=participant_id, 
        details=alignment_details
    )
    # Now perform project deletion, checking for cascading deletion into Experiment & Run
    registration_records = RegistrationRecords(db_path=test_db_path)
    targeted_registration = registration_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_registration = registration_records.delete(
        project_id=project_id,
        participant_id=participant_id
    )
    assert targeted_registration.doc_id == deleted_registration.doc_id
    assert registration_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None
    assert (created_tag.doc_id == 
            deleted_registration['relations']['Tag'][0].doc_id)
    assert (created_alignment.doc_id == 
            deleted_registration['relations']['Alignment'][0].doc_id)
    assert tag_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None
    deactivate_env()
