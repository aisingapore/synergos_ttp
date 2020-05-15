#!/usr/bin/env python

"""
This script is for generating a 2-party end-to-end run of the PySyft REST-RPC
service. Detailed here are all the necessary payload submissions that are
required to be submitted to `http://<ttp_host>:<ttp_port>/ttp/connect/...` in
order to train a model for a PySyft REST-RPC project.

Note: Ensure that a TTP container, and both worker nodes, are all already up.
      Run `full_rpc_connect_simulation.py` before running this script to ensure
      that the necessary prerequisites parameters have been loaded in.
"""

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import requests

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Relevant IDs
project_id = "mmm_project"
expt_id = "mmm_experiment_1"
run_id = "mmm_run_1_2"
participant_id_1 = "fedlearn_worker_1"
participant_id_2 = "fedlearn_worker_2"

# Relevant Training Endpoints
#ttp_host = "0.0.0.0"#"localhost"
ttp_host = "localhost"   #distrib
#ttp_port = 5000#15000
ttp_port = 15000  #distrib
base_ttp_train_url = f"http://{ttp_host}:{ttp_port}/ttp/train"
project_train_url = f"{base_ttp_train_url}/projects/{project_id}"

alignment_init_url = f"{project_train_url}/alignments"
model_init_url = f"{project_train_url}/experiments/{expt_id}/runs/{run_id}/models"

#wssw_init_url = f"http://172.19.152.152:5001/worker/initialise/{project_id}/{expt_id}/{run_id}"

# Model initialisation simulation
init_params = {
    "dockerised": True,
    "verbose": False,
    "log_msgs": False
}

###################
# Helper Function #
###################

def execute_post(url, payload):
    status = requests.post(url=url, json=payload)
    assert status.status_code in [200, 201]
    return status.json()

##########
# Script #
##########

if __name__ == "__main__":
    # Step 1: TTP intialises multiple feature alignment
    align_resp = execute_post(url=alignment_init_url, payload=None)
    logging.debug(f"New alignments: {align_resp}")


    # Step 2: TTP commences model training for specified experiment-run set
    model_resp = execute_post(url=model_init_url, payload=init_params)
    logging.debug(f"New model: {model_resp}")
