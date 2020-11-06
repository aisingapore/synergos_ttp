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
from rest_rpc.connection.core.utils import TagRecords
from rest_rpc.training.core.utils import AlignmentRecords

##################
# Configurations #
##################

tag_details = {
    "train": [["2018","interpolated_eicu_worker_0"]],
    "evaluate": [["2019", "interpolated_eicu_validation"]],
    "predict": [],
    "model": ["2018", "interpolated_model"],
    "hyperparameters": ["2018", "hyperparameters"]
}

tag_updates ={
    "train": [["2019","interpolated_eicu_worker_0"]],
    "evaluate": [],
    "predict": [["2020","interpolated_eicu_prediction"]],
    "model": ["2019", "interpolated_model"],
    "hyperparameters": ["2019", "hyperparameters"]
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
    assert (set(relations.keys()) == set(["Alignment"]))
    return record 

def check_detail_equivalence(details):
    assert details == tag_details
    return details

##########################
# TagRecords Class Tests #
##########################

def test_TagRecords_create():
    tag_records = TagRecords(db_path=test_db_path)
    created_tag = tag_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=tag_details
    )
    raw_details = check_equivalence_and_format(created_tag)
    check_detail_equivalence(raw_details)


def test_TagRecords_read_all():
    tag_records = TagRecords(db_path=test_db_path)
    all_tags = tag_records.read_all()
    assert len(all_tags) == 1
    retrieved_tag = all_tags[0]
    trimmed_details = check_equivalence_and_format(retrieved_tag)
    raw_details = check_relation_equivalence_and_format(trimmed_details)
    check_detail_equivalence(raw_details)


def test_TagRecords_read():
    tag_records = TagRecords(db_path=test_db_path)
    retrieved_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
    )
    assert retrieved_tag is not None
    trimmed_details = check_equivalence_and_format(retrieved_tag)
    raw_details = check_relation_equivalence_and_format(retrieved_tag)
    check_detail_equivalence(raw_details)

def test_TagRecords_update():
    tag_records = TagRecords(db_path=test_db_path)
    targeted_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
    )
    updated_tag = tag_records.update(
        project_id=project_id,
        participant_id=participant_id,
        updates=tag_updates
    )
    assert targeted_tag.doc_id == updated_tag.doc_id
    for k,v in tag_updates.items():
        assert updated_tag[k] == v   


def test_TagRecords_delete():
    # Register an alignment under current experiment
    alignment_records = AlignmentRecords(db_path=test_db_path)
    created_alignment = alignment_records.create(
        project_id=project_id,
        participant_id=participant_id,
        details=alignment_details
    )
    # Now perform tag deletion, checking for cascading deletion into alignment
    tag_records = TagRecords(db_path=test_db_path)
    targeted_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_tag = tag_records.delete(
        project_id=project_id,
        participant_id=participant_id
    )
    assert targeted_tag.doc_id == deleted_tag.doc_id
    assert tag_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None
    assert created_alignment.doc_id == deleted_tag['relations']['Alignment'][0].doc_id
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id
    ) is None
