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
from rest_rpc.training.core.utils import AlignmentRecords

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


####################################
# Association Evaluation Functions #
####################################

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
    assert (set(relations.keys()) == set())
    return record 

def check_detail_equivalence(details):
    assert details == alignment_details
    return details

################################
# AlignmentRecords Class Tests #
################################

def test_AlignmentRecords_create():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    created_alignment = alignment_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=alignment_details
    )
    raw_details = check_equivalence_and_format(created_alignment)
    check_detail_equivalence(raw_details)


def test_AlignmentRecords_read_all():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    all_alignments = alignment_records.read_all()
    assert len(all_alignments) == 1
    retrieved_alignment = all_alignments[0]
    trimmed_details = check_equivalence_and_format(retrieved_alignment)
    raw_details = check_relation_equivalence_and_format(trimmed_details)
    check_detail_equivalence(raw_details)


def test_AlignmentRecords_read():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    retrieved_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    assert retrieved_alignment is not None
    trimmed_details = check_equivalence_and_format(retrieved_alignment)
    raw_details = check_relation_equivalence_and_format(trimmed_details)
    check_detail_equivalence(raw_details)


def test_AlignmentRecords_update():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    targeted_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    updated_alignment = alignment_records.update(
        project_id=project_id,
        participant_id=participant_id,
        updates=alignment_updates
    )
    assert targeted_alignment.doc_id == updated_alignment.doc_id
    for k,v in alignment_updates.items():
        assert updated_alignment[k] == v   


def test_AlignmentRecords_delete():
    alignment_records = AlignmentRecords(db_path=test_db_path)
    targeted_alignment = alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_alignment = alignment_records.delete(
        project_id=project_id,
        participant_id=participant_id
    )
    assert targeted_alignment.doc_id == deleted_alignment.doc_id
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None