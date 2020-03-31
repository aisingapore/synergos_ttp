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
from rest_rpc.connection.core.utils import AlignmentRecords

##################
# Configurations #
##################

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

alignment_updates = {
    "predict": {
        "X": [3, 5],
        "y": []
    }
}

project_id = "eicu_hospital_collab"
participant_id = "CGH"
tag_id = "SS_2019"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

################################
# AlignmentRecords Class Tests #
################################

def test_AlignmentRecords_create():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    created_alignment = alignment_records.create(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id,
        details=alignment_details
    )
    assert 'created_at' in created_alignment.keys()
    created_alignment.pop('created_at')
    key = created_alignment.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert created_alignment == alignment_details


def test_AlignmentRecords_read_all():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    all_alignments = alignment_records.read_all()
    assert len(all_alignments) == 1
    retrieved_alignment = all_alignments[0]
    key = retrieved_alignment.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert 'relations' in retrieved_alignment.keys()
    retrieved_alignment.pop('created_at')
    retrieved_alignment.pop('relations')
    assert retrieved_alignment == alignment_details


def test_AlignmentRecords_read():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    retrieved_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    assert retrieved_alignment is not None
    key = retrieved_alignment.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert 'relations' in retrieved_alignment.keys()
    retrieved_alignment.pop('created_at')
    retrieved_alignment.pop('relations')
    assert retrieved_alignment == alignment_details


def test_AlignmentRecords_update():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    targeted_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    updated_alignment = alignment_records.update(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id,
        updates=alignment_updates
    )
    assert targeted_alignment.doc_id == updated_alignment.doc_id
    for k,v in alignment_updates.items():
        assert updated_alignment[k] == v   


def test_AlignmentRecords_delete():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    targeted_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    deleted_alignment = alignment_records.delete(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    assert targeted_alignment.doc_id == deleted_alignment.doc_id
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    ) is None