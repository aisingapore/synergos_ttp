#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from pathlib import Path

# Libs
from datetime import datetime, timedelta
from tinydb import where

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import Records

##################
# Configurations #
##################

participant = {
    "key": {
        'participant_id': "worker_0"
    },
    "id": "worker_0",
    "host": "0.0.0.0",
    "port": 8020,
    "log_msgs": False,
    "verbose": False
}

project = {
    "key": {
        "project_id": "eicu_hospital_collab"
    },
    "universe_alignment": [],
    "incentives": {},
    "start_at": datetime.strptime(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        '%Y-%m-%d %H:%M:%S'
    ) + timedelta(hours=10)
}

experiment = {
    "key": {
        "project_id": "eicu_hospital_collab",
        "expt_id": "logistic_regression"
    },
    "model": [
        {
            "is_input": True,
            "structure": {
                "in_features": 20,
                "out_features": 1,
                "bias": True
            },
            "type": "linear",
            "activation": "sigmoid"
        }
    ]
}

run = {
    "key": {
        "project_id": "eicu_hospital_collab",
        "expt_id": "logistic_regression",
        "run_id": "fate_params"
    },
    "input_size": 20,
    "output_size": 1,
    "batch_size": 32,
    "lr": 0.15,
    "weight_decay": 0.01,
    "rounds": 2,
    "epochs": 1
}

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

#######################
# Records Class Tests #
#######################

def test_records_create():

    def check_field_equivalence(records, subject, key, new_record):
        created_record = records.create(subject, key, new_record)
        assert 'created_at' in created_record.keys()
        created_record.pop('created_at')
        for k,v in created_record.items():
            assert k in new_record.keys()
            assert v == new_record[k]

    def check_insertion_or_update(records, subject, key, new_record):
        assert records.load_database().table(subject).get(
            where(key) == new_record[key]
        )
        subject_record_count = len(records.load_database().table(subject).all())
        duplicated_record = records.create(subject, key, new_record)
        assert (
            len(records.load_database().table(subject).all()) == 
            subject_record_count
        )
        assert (
            records.load_database().table(subject).get(
                where(key) == new_record[key]
            ).doc_id == duplicated_record.doc_id
        )

    records = Records(db_path=test_db_path)
    records.load_database().purge()
    check_field_equivalence(records, 'Participant', 'key', participant)
    check_field_equivalence(records, 'Project', 'key', project)
    check_field_equivalence(records, 'Experiment', 'key', experiment)
    check_field_equivalence(records, 'Run', 'key', run)

    check_insertion_or_update(records, 'Participant', 'key', participant)
    check_insertion_or_update(records, 'Project', 'key', project)
    check_insertion_or_update(records, 'Experiment', 'key', experiment)
    check_insertion_or_update(records, 'Run', 'key', run)


def test_records_read_all():
    records = Records(db_path=test_db_path)
    assert records.read_all('Participant')[0] == participant
    assert records.read_all('Project')[0] == project
    assert records.read_all('Experiment')[0] == experiment
    assert records.read_all('Run')[0] == run


def test_records_read():
    records = Records(db_path=test_db_path)
    assert records.read("Participant", "key", participant["key"]) == participant
    assert records.read("Project", "key", project["key"]) == project
    assert records.read("Experiment", "key", experiment["key"]) == experiment
    assert records.read("Run", "key", run["key"]) == run


def test_records_update():
    records = Records(db_path=test_db_path)
    targeted_participant = records.read("Participant", "key", participant["key"])

    new_configurations = {
        "id": "worker_1",
        "host": "1.1.1.1",
        "port": 9020,
        "log_msgs": True,
        "verbose": True
    }

    updated_record = records.update(
        "Participant", 
        "key", 
        targeted_participant["key"],
        new_configurations
    )

    assert updated_record.doc_id == targeted_participant.doc_id


def test_records_delete():
    records = Records(db_path=test_db_path)

    target_participant_id = {"participant_id": "worker_0"}
    target_project_id = {"project_id": "eicu_hospital_collab"}
    target_expt_id = {
        "project_id": "eicu_hospital_collab",
        "expt_id": "logistic_regression"
    }
    mappings = {
        "Participant": target_participant_id, 
        "Project": target_project_id,
        "Experiment": target_expt_id
    }

    for subject, key in mappings.items():
        targeted_record = records.read(subject, "key", key)
        deleted_record = records.delete(subject, "key", key)
        assert not records.read(subject, "key", key)
        assert targeted_record.doc_id == deleted_record.doc_id
