#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from pathlib import Path

# Libs
from tinydb import where

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import ExperimentRecords, RunRecords

##################
# Configurations #
##################

expt_details = {
    "model": [
        {
            "is_input": True,
            "structure": {
                "in_features": 20,
                "out_features": 1,
                "bias": True
            },
            "l_type": "linear",
            "activation": "sigmoid"
        }
    ]
}
expt_updates = {
    "model": [
        {
            "is_input": True,
            "structure": {
                "in_features": 20,
                "out_features": 100,
                "bias": True
            },
            "l_type": "linear",
            "activation": "sigmoid"
        },
        {
            "is_input": False,
            "structure": {
                "in_features": 100,
                "out_features": 1,
                "bias": True
            },
            "l_type": "linear",
            "activation": "sigmoid"
        } 
    ]
}

run_details = {
    "input_size": 20,
    "output_size": 1,
    "batch_size": 32,
    "lr": 0.15,
    "weight_decay": 0.01,
    "rounds": 2,
    "epochs": 1
}

project_id = "eicu_hospital_collab"
expt_id = "logistic_regression"
run_id = "fate_params"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

#################################
# ExperimentRecords Class Tests #
#################################

def test_ExperimentRecords_create():
    expt_records = ExperimentRecords(db_path=test_db_path)
    created_expt = expt_records.create(
        project_id=project_id,
        expt_id=expt_id,
        details=expt_details
    )
    assert 'created_at' in created_expt.keys()
    created_expt.pop('created_at')
    key = created_expt.pop('key')
    assert set([project_id, expt_id]) == set(key.values())
    assert created_expt == expt_details


def test_ExperimentRecords_read_all():
    expt_records = ExperimentRecords(db_path=test_db_path)
    all_expts = expt_records.read_all()
    assert len(all_expts) == 1
    retrieved_expt = all_expts[0]
    key = retrieved_expt.pop('key')
    assert set([project_id, expt_id]) == set(key.values())
    assert 'relations' in retrieved_expt.keys()
    retrieved_expt.pop('created_at')
    retrieved_expt.pop('relations')
    assert retrieved_expt == expt_details


def test_ExperimentRecords_read():
    expt_records = ExperimentRecords(db_path=test_db_path)
    retrieved_expt = expt_records.read(
        project_id=project_id, 
        expt_id=expt_id
    )
    assert retrieved_expt is not None
    key = retrieved_expt.pop('key')
    assert set([project_id, expt_id]) == set(key.values())
    assert 'relations' in retrieved_expt.keys()
    retrieved_expt.pop('created_at')
    retrieved_expt.pop('relations')
    assert retrieved_expt == expt_details


def test_ExperimentRecords_update():
    expt_records = ExperimentRecords(db_path=test_db_path)
    targeted_expt = expt_records.read(
        project_id=project_id,
        expt_id=expt_id
    )
    updated_expt = expt_records.update(
        project_id=project_id,
        expt_id=expt_id,
        updates=expt_updates
    )
    assert targeted_expt.doc_id == updated_expt.doc_id
    for k,v in expt_updates.items():
        assert updated_expt[k] == v   


def test_ExperimentRecords_delete():
    # Register a run under current experiment
    run_records = RunRecords(db_path=test_db_path)
    created_run = run_records.create(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        details=run_details
    )
    # Now perform experiment deletion, checking for cascading deletion into run
    expt_records = ExperimentRecords(db_path=test_db_path)
    targeted_expt = expt_records.read(
        project_id=project_id,
        expt_id=expt_id
    )
    deleted_expt = expt_records.delete(
        project_id=project_id,
        expt_id=expt_id
    )
    assert targeted_expt.doc_id == deleted_expt.doc_id
    assert expt_records.read(project_id=project_id, expt_id=expt_id) is None
    assert created_run.doc_id == deleted_expt['relations']['Run'][0].doc_id
    assert run_records.read(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
