##############
# Base Image #
##############

FROM python:3.7.4-slim-buster as base

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git\
    pciutils

COPY requirements.txt ./

RUN pip install --upgrade pip \
 && pip install --upgrade setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ADD . /ttp
WORKDIR /ttp

RUN pip install ./synergos_algorithm
RUN pip install ./synergos_archive
RUN pip install ./synergos_logger
RUN pip install ./synergos_manager

EXPOSE 5000
EXPOSE 8020
EXPOSE 8080

##############################
# Dev Image Layer - Debugger #
##############################

FROM base as debug
RUN pip install ptvsd

WORKDIR /ttp
EXPOSE 5678
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main.py

###########################################
# Production Image Layer - Synergos Basic #
###########################################

FROM base as basic_ttp

WORKDIR /ttp
ENTRYPOINT ["python", "./main_basic.py"]
CMD ["--help"]

################################################
# Production Image Layer - Synergos SynCluster #
################################################

FROM base as syncluster_ttp

WORKDIR /ttp
ENTRYPOINT ["python", "./main_syncluster.py"]
CMD ["--help"]