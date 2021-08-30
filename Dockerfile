##############
# Base Image #
##############

FROM python:3.7.4-slim-buster as base

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git\
    pciutils

RUN pip install --upgrade setuptools wheel

ADD ./synergos_algorithm /ttp/synergos_algorithm
RUN pip install /ttp/synergos_algorithm

ADD ./synergos_archive /ttp/synergos_archive
RUN pip install /ttp/synergos_archive

ADD ./synergos_logger /ttp/synergos_logger
RUN pip install /ttp/synergos_logger

ADD ./synergos_manager /ttp/synergos_manager
RUN pip install /ttp/synergos_manager

ADD ./synergos_rest /ttp/synergos_rest
RUN pip install /ttp/synergos_rest

WORKDIR /ttp
ADD . /ttp

EXPOSE 5000
EXPOSE 8020
EXPOSE 8080

####################################
# Dev Image Layer - Debugger Basic #
####################################

FROM base as debug_basic
RUN pip install ptvsd

WORKDIR /ttp
EXPOSE 5678
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main_basic.py

###########################################
# Production Image Layer - Synergos Basic #
###########################################

FROM base as basic_ttp

WORKDIR /ttp
ENTRYPOINT ["python", "./main_basic.py"]
CMD ["--help"]

#########################################
# Dev Image Layer - Debugger SynCluster #
#########################################

FROM base as debug_syncluster
RUN pip install ptvsd

WORKDIR /ttp
EXPOSE 5678
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main_syncluster.py

################################################
# Production Image Layer - Synergos SynCluster #
################################################

FROM base as syncluster_ttp

WORKDIR /ttp
ENTRYPOINT ["python", "./main_syncluster.py"]
CMD ["--help"]