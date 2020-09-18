# Running Heart_disease dataset

The following are simple instructions on how to run a FL cycle on two worker nodes.

Start two worker node in the directory `./pysyft_worker`
```
$ pysyft_worker > docker run -v path_to_heart_disease_dataset/heart_disease/data1:/worker/data -v path_to_outputs/outputs_1:/worker/outputs --name worker_1 worker:pysyft_demo

$ pysyft_worker > docker run -v path_to_heart_disease_dataset/heart_disease/data1:/worker/data -v path_to_outputs/outputs_2:/worker/outputs --name worker_2 worker:pysyft_demo
```

Start TTP node in the directory `./pysyft_ttp`
```
$ pysyft_ttp > docker run -p 5000:5000 -p 5678:5678 -p 8020:8020 -p 8080:8080 -v path_to_ml_flow_directory/mlflow_test:/ttp/mlflow --name ttp --link worker_1 --link worker_2 ttp:pysyft_demo
```

Configure the metadata for the worker nodes
```
$ pysyft_ttp/admin/ > python configure_training.py
```

Start training
```
$ pysyft_ttp/admin/ > python launch_training.py
```

Start evaluating
```
$ pysyft_ttp/admin/ > python evaluate_training.py
```