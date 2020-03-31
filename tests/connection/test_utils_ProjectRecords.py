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
from rest_rpc.connection.core.utils import ProjectRecords, ExperimentRecords, RunRecords

##################
# Configurations #
##################

project_details = {
    "universe_alignment": [],
    "incentives": {},
    "start_at": datetime.strptime(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        '%Y-%m-%d %H:%M:%S'
    )
}

project_updates = {
    "start_at": datetime.now() + timedelta(hours=10)
}

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

##############################
# ProjectRecords Class Tests #
##############################

def test_ProjectRecords_create():
    project_records = ProjectRecords(db_path=test_db_path)
    created_project = project_records.create(
        project_id=project_id,
        details=project_details
    )
    assert 'created_at' in created_project.keys()
    created_project.pop('created_at')
    key = created_project.pop('key')
    assert set([project_id]) == set(key.values())
    assert created_project == project_details


def test_ProjectRecords_read_all():
    project_records = ProjectRecords(db_path=test_db_path)
    all_projects = project_records.read_all()
    assert len(all_projects) == 1
    retrieved_project = all_projects[0]
    key = retrieved_project.pop('key')
    assert set([project_id]) == set(key.values())
    assert 'relations' in retrieved_project.keys()
    retrieved_project.pop('created_at')
    retrieved_project.pop('relations')
    assert retrieved_project == project_details


def test_ProjectRecords_read():
    project_records = ProjectRecords(db_path=test_db_path)
    retrieved_project = project_records.read(
        project_id=project_id
    )
    assert retrieved_project is not None
    key = retrieved_project.pop('key')
    assert set([project_id]) == set(key.values())
    assert 'relations' in retrieved_project.keys()
    retrieved_project.pop('created_at')
    retrieved_project.pop('relations')
    assert retrieved_project == project_details


def test_ProjectRecords_update():
    project_records = ProjectRecords(db_path=test_db_path)
    targeted_project = project_records.read(
        project_id=project_id
    )
    updated_project = project_records.update(
        project_id=project_id,
        updates=project_updates
    )
    assert targeted_project.doc_id == updated_project.doc_id
    for k,v in project_updates.items():
        assert updated_project[k] == v   


def test_ProjectRecords_delete():
    # Register an experiment under the current project
    expt_records = ExperimentRecords(db_path=test_db_path)
    created_expt = expt_records.create(
        project_id=project_id,
        expt_id=expt_id,  
        details=expt_details 
    )
    # Register a run under experiment
    run_records = RunRecords(db_path=test_db_path)
    created_run = run_records.create(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        details=run_details
    )
    # Now perform project deletion, checking for cascading deletion into Experiment & Run
    project_records = ProjectRecords(db_path=test_db_path)
    targeted_project = project_records.read(
        project_id=project_id
    )
    deleted_project = project_records.delete(
        project_id=project_id
    )
    assert targeted_project.doc_id == deleted_project.doc_id
    assert project_records.read(project_id=project_id) is None
    assert created_expt.doc_id == deleted_project['relations']['Experiment'][0].doc_id
    assert created_run.doc_id == deleted_project['relations']['Run'][0].doc_id
    assert expt_records.read(
        project_id=project_id,
        expt_id=expt_id
    ) is None
    assert run_records.read(
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
