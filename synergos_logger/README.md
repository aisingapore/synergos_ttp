######################################################## Testing locally ########################################################
In the graylog folder
> pip install -r requirements.txt
> pip install .
> docker-compose up

Go to browser "localhost:9000" and launch new GELF TCP Input with port 12201

Open another terminal and cd to "graylog/SynergosLogger" folder
> python test.py
The log should be reflected in localhost:9000


Open another terminal and cd to "graylog/psutil" folder
> python HardwareStatsLogger.py
The cpu_percent log should be reflect in localhost:9000

######################################################## Testing on Docker container (docker compose) ########################################################
Test locally (With docker-compose) in graylog folder
> go to SynergosLogger/config.py and set GRAYLOG_SERVER="127.0.0.1" for testing locally or GRAYLOG_SERVER="graylog" for testing on linux container
> docker-compose build
> docker-compose up
> docker ps # get the <CONTAINER ID> of graylog_linux_ctr
> docker exec -it <CONTAINER ID> bash
> cd SynergosLogger
> python test.py

> cd psutil
> python test.py

The log should be reflected in localhost:9000

######################################################## Testing on Docker container (docker run) ########################################################
Test locally (Without docker-compose)
https://stackoverflow.com/questions/36489696/cannot-link-to-a-running-container-started-by-docker-compose
linux container
> docker network ls
> docker build -t linux_ctr_1 .
> docker run -p 9001:9001 --link graylog:graylog --net graylog_default linux_ctr_1

The log should be reflected in localhost:9000


Stopping container manually
> docker ps # get the <CONTAINER ID>
> docker container stop <CONTAINER ID>



