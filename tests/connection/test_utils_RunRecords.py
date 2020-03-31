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
from rest_rpc.connection.core.utils import RunRecords

##################
# Configurations #
##################

run_details = {
    "input_size": 20,
    "output_size": 1,
    "batch_size": 32,
    "lr": 0.15,
    "weight_decay": 0.01,
    "rounds": 2,
    "epochs": 1
}

run_updates = {
    "weight_decay": 0.1,
    "rounds": 20,
    "epochs": 5  
}

project_id = "eicu_hospital_collab"
expt_id = "logistic_regression"
run_id = "fate_params"

test_db_path = os.path.join(app.config['TEST_DIR'], "test_database.json")

##########################
# RunRecords Class Tests #
##########################

def test_RunRecords_create():
    run_records = RunRecords(db_path=test_db_path)
    database = run_records.load_database()
    database.purge()
    created_run = run_records.create(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        details=run_details
    )
    assert 'created_at' in created_run.keys()
    created_run.pop('created_at')
    key = created_run.pop('key')
    assert set([project_id, expt_id, run_id]) == set(key.values())
    assert created_run == run_details


def test_RunRecords_read_all():
    run_records = RunRecords(db_path=test_db_path)
    all_runs = run_records.read_all()
    assert len(all_runs) == 1
    retrieved_run = all_runs[0]
    key = retrieved_run.pop('key')
    assert set([project_id, expt_id, run_id]) == set(key.values())
    assert 'relations' in retrieved_run.keys()
    retrieved_run.pop('created_at')
    retrieved_run.pop('relations')
    assert retrieved_run == run_details


def test_RunRecords_read():
    run_records = RunRecords(db_path=test_db_path)
    retrieved_run = run_records.read(
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id
    )
    assert retrieved_run is not None
    key = retrieved_run.pop('key')
    assert set([project_id, expt_id, run_id]) == set(key.values())
    assert 'relations' in retrieved_run.keys()
    retrieved_run.pop('created_at')
    retrieved_run.pop('relations')
    assert retrieved_run == run_details


def test_RunRecords_update():
    run_records = RunRecords(db_path=test_db_path)
    targeted_run = run_records.read(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    updated_run = run_records.update(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        updates=run_updates
    )
    assert targeted_run.doc_id == updated_run.doc_id
    for k,v in run_updates.items():
        assert updated_run[k] == v


def test_RunRecords_delete():
    run_records = RunRecords(db_path=test_db_path)
    targeted_run = run_records.read(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    deleted_run = run_records.delete(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    assert targeted_run.doc_id == deleted_run.doc_id
    assert run_records.read(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None

