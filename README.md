# Running Dockerised PySyft TTP Node
## Method 1: Via running `run.py` directly
1) Clone this repository
2) Navigate into the repository
    > `cd /path/to/ttp`
3) Make sure that the worker containers have already started running (i.e. primed for WS handshake <insert link-to-worker-readme.md here>).
4) Manually change server configuration parameters to match those initialised on the worker nodes. 

    For example, if worker node 1 was initialised with this command:

    >`docker run -p <host>:<port>:8020 worker:pysyft_demo -H 0.0.0.0 -p 8020 -i <id> -t <train> -e <evaluate> -v`

    Then the parameter set in `"ttp/participants/worker_1.json"` should be:

    > `{"host": "127.0.0.1",
       "port": 8020,
       "id": "Alice",
       "log_msgs": true,
       "verbose": true}`

    (Note: THIS IS NOT HOW THIS SHOULD WORK! This is only a temporary fix. The real mechanism is that these parameter sets will be configured automatically via Flask API calls to the worker nodes. Until the Flask extension has been integrated, this process will suffice for a simple demo)

5) Initialise the training process using the following command(s):

    > `python ./run.py -m <models> -e <experiments> <-v>`

    * models - Filenames of model architecture to load.
    * experiments - Filenames of experiment parameters to load
    * -v - Verbosity switch. Usually for TTP, this is turned off. Specify to turn on.

    **Explanation:**
    
    Here, the script variables for `run.py` are specified in order to load the selected model architectures and experiment parameters.

---

## Method 2: Via Docker (In Progress)
1) Clone this repository
2) Navigate into the repository
    > `cd /path/to/ttp`
3) Build image using the following command(s): 
    > `docker build -t ttp:pysyft_demo --label "WebsocketClientWorker" .`
4) Make sure that the worker containers have already started running (i.e. primed for WS handshake <insert link-to-worker-readme.md here>).
5) Start up the worker node using the following command(s):

    > `docker run -p <host>:<port>:8020 ttp:pysyft_demo -m <models> -e <experiments> <-v>`
    
    * host - IP of host machine
    * port - Selected port to route incoming connections into the container
    * models - Filenames of model architecture to load.
    * experiments - Filenames of experiment parameters to load
    * -v - Verbosity switch. Usually for TTP, this is turned off. Specify to turn on

    **Explanation:**
    
    By default, all PySyft containers, be it TTP or worker, will be set up to run on internal port `8020`.

    Here, we are setting docker up to route any incoming connections/requests on a specified port of the docker host to the internal port `8020`. The container then takes in script variables for `run.py` to log model architectures and experiment parameters to load.

6) Connect to bridge network to allow communication with TTP (Required if TTP is dockerised as well - in progress)