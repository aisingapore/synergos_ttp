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
from rest_rpc.connection.core.utils import ParticipantRecords, TagRecords
from rest_rpc.training.core.utils import AlignmentRecords

##################
# Configurations #
##################

participant_details = {
    "id": "CGH",
    "host": "0.0.0.0",
    "port": 8020,
    "log_msgs": False,
    "verbose": False
}

participant_updates = {
    "host": "1.1.1.1",
    "port": 8040,
    "log_msgs": True,
}

tag_details = {
    "train": [["2018","interpolated_eicu_worker_0"]],
    "evaluate": [["2019", "interpolated_eicu_validation"]],
    "predict": []
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

project_id = "eicu_hospital_collab"
participant_id = "CGH"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

##################################
# ParticipantRecords Class Tests #
##################################

def test_ParticipantRecords_create():
    participant_records = ParticipantRecords(db_path=test_db_path)
    created_participant = participant_records.create(
        participant_id=participant_id,
        details=participant_details
    )
    assert 'created_at' in created_participant.keys()
    created_participant.pop('created_at')
    key = created_participant.pop('key')
    assert set([participant_id]) == set(key.values())
    assert created_participant == participant_details


def test_ParticipantRecords_read_all():
    participant_records = ParticipantRecords(db_path=test_db_path)
    all_participants = participant_records.read_all()
    assert len(all_participants) == 1
    retrieved_participant = all_participants[0]
    key = retrieved_participant.pop('key')
    assert set([participant_id]) == set(key.values())
    assert 'relations' in retrieved_participant.keys()
    retrieved_participant.pop('created_at')
    retrieved_participant.pop('relations')
    assert retrieved_participant == participant_details


def test_ParticipantRecords_read():
    participant_records = ParticipantRecords(db_path=test_db_path)
    retrieved_participant = participant_records.read(
        participant_id=participant_id
    )
    assert retrieved_participant is not None
    key = retrieved_participant.pop('key')
    assert set([participant_id]) == set(key.values())
    assert 'relations' in retrieved_participant.keys()
    retrieved_participant.pop('created_at')
    retrieved_participant.pop('relations')
    assert retrieved_participant == participant_details


def test_ParticipantRecords_update():
    participant_records = ParticipantRecords(db_path=test_db_path)
    targeted_participant = participant_records.read(
        participant_id=participant_id
    )
    updated_participant = participant_records.update(
        participant_id=participant_id,
        updates=participant_updates
    )
    assert targeted_participant.doc_id == updated_participant.doc_id
    for k,v in participant_updates.items():
        assert updated_participant[k] == v  


def test_ParticipantRecords_delete():
    # Register an tag under the current participant registered for a project
    tag_records = TagRecords(db_path=test_db_path)
    created_tag = tag_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=tag_details
    )
    # Register an alignment under created tag
    alignment_records = AlignmentRecords(db_path=test_db_path)
    created_alignment = alignment_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=alignment_details
    )
    # Now perform participant deletion, checking for cascading deletion into tag & alignment
    participant_records = ParticipantRecords(db_path=test_db_path)
    targeted_participant = participant_records.read(
        participant_id=participant_id
    )
    deleted_participant = participant_records.delete(
        participant_id=participant_id
    )
    assert targeted_participant.doc_id == deleted_participant.doc_id
    assert participant_records.read(participant_id=participant_id) is None
    assert created_tag.doc_id == deleted_participant['relations']['Tag'][0].doc_id
    assert created_alignment.doc_id == deleted_participant['relations']['Alignment'][0].doc_id
    assert tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
    ) is None
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
    ) is None