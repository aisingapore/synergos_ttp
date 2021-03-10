#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import uuid
import argparse

# Libs
import re
import ray
from ray import tune

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    RunRecords,
    ProjectRecords,
    ExperimentRecords,
    RegistrationRecords,
)
from rest_rpc.training.core.utils import (
    Poller
)

from manager.train_operations import TrainProducerOperator
from manager.evaluate_operations import EvaluateProducerOperator

##################
# Configurations #
##################

db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)

########################################
# HP Tuning Class - HPTuning #
########################################
def tune_trainable(config, checkpoint_dir=None):
    """
        trainable function for tune.run()
        This trainable function create records on the fly and store it in database.json f
        or every search_space configuration it receives from calling tune.run()
    """
    print("config: ", config)
    expt_id = config["expt_id"]
    project_id = config["project_id"]
    run_id = "optim_run_" + str(uuid.uuid4())
    search_space = config['search_space']

    ''':
        {'algorithm': 'FedProx', 
        'base_lr': 0.0005, 'criterion': 'MSELoss', 'delta': 0.0, 'epochs': 1, 'is_snn': False, 'l1_lambda': 0.0,
        'l2_lambda': 0.0, 'lr': 0.001, 'lr_decay': 0.1, 'lr_scheduler': 'CyclicLR', 'max_lr': 0.005, 'mu': 0.1,
        'optimizer': 'SGD', 'patience': 10, 'precision_fractional': 5, 'rounds': 'tune.choice([1,2,3,4,5])', 'seed': 42, 'weight_decay': 0.0}
    '''

    # Store records into database.json
    new_run = run_records.create(
        project_id=project_id, 
        expt_id=expt_id,
        run_id=run_id,
        details=config['search_space']
    )
    retrieved_run = run_records.read(
        project_id=project_id, 
        expt_id=expt_id,
        run_id=run_id
    )
    assert new_run.doc_id == retrieved_run.doc_id

def start_generate_hp(kwargs=None):
    """
        Start generate hyperparameters configurations given the following kwargs argument
        args:
            kwargs: {'expt_id': 'test_experiment_1', 'project_id': 'test_project_1', n_samples: 3, 
                    'search_space': {'algorithm': 'FedProx', 'base_lr': 0.0005, 'criterion': 'MSELoss',
                    'delta': 0.0, 'epochs': 1, 'is_snn': False, 'l1_lambda': 0.0, 'l2_lambda': 0.0, 'lr': 0.001,
                    'lr_decay': 0.1, 'lr_scheduler': 'CyclicLR', 'max_lr': 0.005, 'mu': 0.1, 'optimizer': 'SGD',
                    'patience': 10, 'precision_fractional': 5, 'rounds': {'_type': 'choice', '_value': [1,2,3,4,5]},
                    'seed': 42, 'weight_decay': 0.0}}

    """
    # ray.shutdown()
    print("start generate hp")
    # ray.init(local_mode=True, ignore_reinit_error=True)
    ray.init(local_mode=False, num_cpus=1, num_gpus=0)

    num_samples = kwargs['n_samples'] # num of federated experiments (diff experiments diff hyperparameter configurations)
    gpus_per_trial=0

    # Mapping custom search space config into tune config (TODO)
    search_space = kwargs['search_space']
    for hyperparameter_key in search_space.keys():
        try:
            if search_space[hyperparameter_key]['_type'] == 'choice':
                search_space[hyperparameter_key] = tune.choice(
                    search_space[hyperparameter_key]['_value']
                    )

            elif search_space[hyperparameter_key]['_type'] == 'uniform':
                search_space[hyperparameter_key] = tune.uniform(
                    search_space[hyperparameter_key]['_value'][0],
                    search_space[hyperparameter_key]['_value'][1]
                    )
            
            elif search_space[hyperparameter_key]['_type'] == 'loguniform':
                search_space[hyperparameter_key] = tune.loguniform(
                    search_space[hyperparameter_key]['_values'][0],
                    search_space[hyperparameter_key]['_values'][1]
                )

            elif search_space[hyperparameter_key]['_type'] == 'randint':
                search_space[hyperparameter_key] = tune.randint(
                    search_space[hyperparameter_key]['_values'][0],
                    search_space[hyperparameter_key]['_values'][1]
                )
        
        except:
            continue

    result = tune.run(
        tune_trainable,
        config=kwargs,
        num_samples=num_samples,
    )

    print("hp kwargs: ", kwargs)

    return kwargs

def start_hp_training(project_id, expt_id, run_id):
    """
        start hyperparameter training by sending generated hyperparemters config into the train queue
    """
    # The below train producer logic is extracted from the post function at rest_rpc/training/models.py

    # Populate grid-initialising parameters
    init_params = {'auto_align': True, 'dockerised': True, 'verbose': True, 'log_msgs': True}

    # Retrieves expt-run supersets (i.e. before filtering for relevancy)
    retrieved_project = project_records.read(project_id=project_id)
    project_action = retrieved_project['action']
    experiments = retrieved_project['relations']['Experiment']
    runs = retrieved_project['relations']['Run']

    # If specific experiment was declared, collapse training space
    if expt_id:
        retrieved_expt = expt_records.read(
            project_id=project_id, 
            expt_id=expt_id
        )
        runs = retrieved_expt.pop('relations')['Run']
        experiments = [retrieved_expt]

        # If specific run was declared, further collapse training space
        if run_id:

            retrieved_run = run_records.read(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )
            retrieved_run.pop('relations')
            runs = [retrieved_run]

    # Retrieve all participants' metadata
    registrations = registration_records.read_all(
        filter={'project_id': project_id}
    )

    auto_align = init_params['auto_align']
    if not auto_align:
        poller = Poller(project_id=project_id)
        poller.poll(registrations)

    # Template for starting FL grid and initialising training
    kwargs = {
        'action': project_action,
        'experiments': experiments,
        'runs': runs,
        'registrations': registrations
    }
    kwargs.update(init_params)

    # output_payload = None #NOTE: Just added

    if app.config['IS_CLUSTER_MODE']:
        train_operator = TrainProducerOperator(host=app.config["SYN_MQ_HOST"])
        result = train_operator.process(project_id, kwargs)

        #return IDs of runs submitted
        resp_data = {"run_ids": result}
        print("resp_data: ", resp_data)    

def send_evaluate_msg(project_id, expt_id, run_id, participant_id=None):
    """
        Sending an evaluate message to the evaluate queue given the following args
        args:
            project_id: "test_project"
            expt_id: "test_experiment"
            run_id: "test_run"
            participant_id: "test_participant_1"
    """
    # Populate grid-initialising parameters
    # init_params = {'auto_align': True, 'dockerised': True, 'verbose': True, 'log_msgs': True} # request.json
    
    # Retrieves expt-run supersets (i.e. before filtering for relevancy)
    retrieved_project = project_records.read(project_id=project_id)
    print("retrieved_project: ", retrieved_project)
    project_action = retrieved_project['action']
    experiments = retrieved_project['relations']['Experiment']
    runs = retrieved_project['relations']['Run']

    # If specific experiment was declared, collapse training space
    if expt_id:

        retrieved_expt = expt_records.read(
            project_id=project_id, 
            expt_id=expt_id
        )
        runs = retrieved_expt.pop('relations')['Run']
        experiments = [retrieved_expt]

        # If specific run was declared, further collapse training space
        if run_id:

            retrieved_run = run_records.read(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )
            retrieved_run.pop('relations')
            runs = [retrieved_run]

    # Retrieve all participants' metadata
    registrations = registration_records.read_all(
        filter={'project_id': project_id}
    )

    # Retrieve all relevant participant IDs, collapsing evaluation space if
    # a specific participant was declared
    participants = [
        record['participant']['id'] 
        for record in registrations
    ] if not participant_id else [participant_id]

    # Template for starting FL grid and initialising validation
    kwargs = {
        'action': project_action,
        'experiments': experiments,
        'runs': runs,
        'registrations': registrations,
        'participants': participants,
        'metas': ['evaluate'],
        'version': None # defaults to final state of federated grid
    }

    # kwargs.update(init_params)

    if app.config['IS_CLUSTER_MODE']:
        evaluate_operator = EvaluateProducerOperator(host=app.config["SYN_MQ_HOST"])
        result = evaluate_operator.process(project_id, kwargs)

        data = {"run_ids": result}

def start_hp_validations(payload, host):
    """
        Custom callback function for sending evaluate message after receiving 
        the payload from completed queue
        args:
            payload:  "TRAINING COMPLETE -  test_project_1/test_experiment_1/optim_run_5c68e185-c28f-4159-8df4-2504ce94f4c7"
            host: RabbitMQ Server
    """
    
    if re.search(r"TRAINING COMPLETE .+/optim_run_.*", payload):
        message_components = re.findall(r"[\w\-]+", payload)
        project_id = message_components[3]
        expt_id = message_components[4]
        run_id = message_components[5]
        send_evaluate_msg(project_id, expt_id, run_id)

    # check if the payload contains training complete before sending to evaluate queue
    # if message_components[0] == 'TRAINING' and message_components[1] == 'COMPLETE':
    #     print("STARTING hp validations")
    #     print(project_id, expt_id, run_id)
    #     send_evaluate_msg(project_id, expt_id, run_id)
    else:
        print("NOT TRAINING. pass..")

# def read_search_space_path(search_space_path):
#     '''
#     Parse search_space.json for project
#     '''
#     search_space = json.load(search_space_path)

#     return search_space

def str2none(v):
    '''
    Converts string None to NoneType for module compatibility
    in main.py
    '''
    if v == "None":
        return None
    else:
        return v
        
if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # receive arguments for synergos mq server host
    parser.add_argument(
        '--n_samples',
        dest='n_samples',
        help='Synergos HP Tuning',
        type=int,
        default=3
    )

    # reference where search_space json file is located
    parser.add_argument(
        '--search',
        dest='search_space_path',
        help='Search space path',
        type=str
    )
    
    args = parser.parse_args()

    '''
    search_space = {
        'algorithm': 'FedProx',
        'rounds': {"_type": "choice", "_value": [1,2,3,4,5]},
        'epochs': 1,
        'lr': 0.001,
        'weight_decay': 0.0,
        'lr_decay': 0.1,
        'mu': 0.1,
        'l1_lambda': 0.0,
        'l2_lambda': 0.0,
        'optimizer': 'SGD',
        'criterion': 'MSELoss',
        'lr_scheduler': 'CyclicLR',
        'delta': 0.0,
        'patience': 10,
        'seed': 42,
        'is_snn': False,
        'precision_fractional': 5,
        'base_lr': 0.0005,
        'max_lr': 0.005,
    }
    '''
    # search_space = read_search_space_path(args.search_space_path)

    '''
    kwargs = {
        "project_id": "test_project_1",
        "expt_id": "test_experiment_1",
        "n_samples": args.n_samples,
        "search_space": search_space,
    }

    start_generate_hp(kwargs)
    '''