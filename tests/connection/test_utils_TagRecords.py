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
from rest_rpc.connection.core.utils import TagRecords, AlignmentRecords

##################
# Configurations #
##################

tag_details = {
    "train": [["2018","interpolated_eicu_worker_0"]],
    "evaluate": [["2019", "interpolated_eicu_validation"]],
    "predict": []
}

tag_updates ={
    "train": [["2019","interpolated_eicu_worker_0"]],
    "evaluate": [],
    "predict": [["2020","interpolated_eicu_prediction"]]
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
tag_id = "SS_2019"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

##########################
# TagRecords Class Tests #
##########################

def test_TagRecords_create():
    tag_records = TagRecords(db_path=test_db_path)
    created_tag = tag_records.create(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id,
        details=tag_details
    )
    assert 'created_at' in created_tag.keys()
    created_tag.pop('created_at')
    key = created_tag.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert created_tag == tag_details


def test_TagRecords_read_all():
    tag_records = TagRecords(db_path=test_db_path)
    all_tags = tag_records.read_all()
    assert len(all_tags) == 1
    retrieved_tag = all_tags[0]
    key = retrieved_tag.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert 'relations' in retrieved_tag.keys()
    retrieved_tag.pop('created_at')
    retrieved_tag.pop('relations')
    assert retrieved_tag == tag_details


def test_TagRecords_read():
    tag_records = TagRecords(db_path=test_db_path)
    retrieved_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    assert retrieved_tag is not None
    key = retrieved_tag.pop('key')
    assert set([project_id, participant_id, tag_id]) == set(key.values())
    assert 'relations' in retrieved_tag.keys()
    retrieved_tag.pop('created_at')
    retrieved_tag.pop('relations')
    assert retrieved_tag == tag_details


def test_TagRecords_update():
    tag_records = TagRecords(db_path=test_db_path)
    targeted_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    updated_tag = tag_records.update(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id,
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
        tag_id=tag_id,
        details=alignment_details
    )
    # Now perform tag deletion, checking for cascading deletion into alignment
    tag_records = TagRecords(db_path=test_db_path)
    targeted_tag = tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    deleted_tag = tag_records.delete(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    )
    assert targeted_tag.doc_id == deleted_tag.doc_id
    assert tag_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    ) is None
    assert created_alignment.doc_id == deleted_tag['relations']['Alignment'][0].doc_id
    assert alignment_records.read(
        project_id=project_id,
        participant_id=participant_id,
        tag_id=tag_id
    ) is None
