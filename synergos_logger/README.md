# SynergosLogging
[SynergosLogging](https://gitlab.int.aisingapore.org/aims/federatedlearning/synergos_worker/-/tree/synergos_logger/synergos_logger) is a custom logging package created with [Structlog](https://www.structlog.org/en/0.4.0/api.html), [Graypy](https://github.com/conda-forge/graypy-feedstock) and [psutil](https://psutil.readthedocs.io/en/latest/) to make logging easier using a centralized log management system such as [Graylog](https://www.graylog.org/). 

**Architecture - SynergosLogging**

![Screenshot_2020-11-08_at_9.13.54_PM](https://gitlab.int.aisingapore.org/aims/federatedlearning/fedlearn-prototype/-/wikis/uploads/6c68ca42e9550ce06b9fdd45ff4c82f0/Screenshot_2020-11-08_at_9.13.54_PM.png)

## Configure SynergosLogging

To run graylog, it requires:

- Graylog: [graylog/graylog](https://hub.docker.com/r/graylog/graylog/)

- MongoDB: [mongo](https://hub.docker.com/_/mongo/)

- Elasticsearch: [docker.elastic.co/elasticsearch/elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/5.5/docker.html)


**Run the following commands:**


```bash
#Navigate into the repository
cd ./synergos_ttp/synergos_logger

#Install the necessary requirements
pip install -r requirements.txt
pip install .
```


# Configuration for SynergosLogging

Should there be a need to configure the networking ports, the config file can be edited through:
```bash
#Navigate into the syn_logger_config.py file
cd ./SynergosLogger/syn_logger_config.py
```
It allows all synergos component to configure it's logging profile such as modifying and adding new synergos component or port to dynamically perform logging to graylog inputs.

### SynergosLogging switch (SME and Cluster)
The configuration file <b>syn_logger_config.py</b> allows switching to either 1. Using basic logging (without a graylog server) or 2. Using graypy logging (with a graylog server).
1. <b>Graylog logging</b>
    ```
    LOGGING_VARIANT = "graylog" # use graylog server or basic logging
    ```
2. <b>Basic logging</b>
    ```
    LOGGING_VARIANT = "basic" # use graylog server or basic logging
    ```
### Configuring logging profile for Synergos component
Change the constant `SYN_COMPONENT` in the configuration file to either `"ttp"` if using the ttp container or `"worker"` if using the worker container. If `SYN_COMPONENT="ttp"`, the port 12201 will be used for ttp logging in the graylog server. (<b>Change the default logging port accordingly when required.</b>)
```
..
LOGGING_VARIANT = "graylog" # use graylog server or basic logging
SYN_COMPONENT = "ttp" # change to "ttp" if working on ttp container else "worker"

### default logging port 12201 to 12203 which is to be created in the graylog server
TTP_PORT = 12201
WORKER_PORT = 12202
SYSMETRICS_PORT = 12203
..
```


Once all the ports have been configured, we can proceed to launch Graylog from **synergos_logger** directory.

```bash
#Run the docker-compose command
docker-compose up
```
It should return **Graylog server up and running.**

![graylog_up](https://gitlab.int.aisingapore.org/aims/federatedlearning/fedlearn-prototype/-/wikis/uploads/31e95f992e90cc429514c4d91ae2c212/graylog_up.png)

For more information on how to use Graylog, please refer to https://gitlab.int.aisingapore.org/aims/federatedlearning/fedlearn-prototype/-/wikis/Centralized-logging-for-Synergos