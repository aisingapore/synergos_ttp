# Running Femnist dataset

The following are simple instructions on how to run a FL cycle on two worker nodes.

The following folders must be created or exists before running initializing the TTP and worker container
- `path_to_femnist_dataset/femnist`
```
femnist
└───data1
│   └───train
│       │   class_1
|               └───.png
│       │   class_2
│       │   class_3
│       |   metadata.jsonn
│   └───evaluate
│   └───predict
│───data2
│   └───train
│       │   class_1
|               └───.png
│       │   class_2
│       │   class_3
│       |   metadata.jsonn
│   └───evaluate
│   └───predict
```
- `mlflow_test` # The project metrics/params from ttp node
- `ttp_data` # The information of the experiment from ttp node
- `outputs_1` # The directory output from worker_1 node
- `outputs_2` # The directory output from worker_2 node

### Initialize TTP and worker container
Start two worker node in the directory `./pysyft_worker`
```
$ pysyft_worker > docker run -v path_to_femnist_dataset/femnist/data1:/worker/data -v path_to_outputs/outputs_1:/worker/outputs --name worker_1 worker:pysyft_demo

$ pysyft_worker > docker run -v path_to_femnist_dataset/femnist/data1:/worker/data -v path_to_outputs/outputs_2:/worker/outputs --name worker_2 worker:pysyft_demo
```

Start TTP node in the directory `./pysyft_ttp`
```
$ pysyft_ttp > docker run -p 5000:5000 -p 5678:5678 -p 8020:8020 -p 8080:8080 -v path_to_ttp_directory/ttp_data:/ttp/data -v path_to_ml_flow_directory/mlflow_test:/ttp/mlflow --name ttp --link worker_1 --link worker_2 ttp:pysyft_demo
```

### Running one FL cycle
Configure the metadata for the worker nodes such as 
1. Registering the participants information onto the grid for experiments (training/evaluation/prediction).
2. Setting up the model architecture to be used for experiments.
3. Registering the datasets tags provided from participants.
```
$ pysyft_ttp/admin/ > python configure_training.py
```

Start training
```
$ pysyft_ttp/admin/ > python launch_training.py
```

Start evaluating (Ensure mlflow_test directory is empty before running evaluations, mlflow will create the output after each run)
```
$ pysyft_ttp/admin/ > python evaluate_training.py
```

### Misc
When TTP container is shutdown/restarted, you should also clear the cache from `mlflow_test` and `ttp_data` before re-running the above operations again.
