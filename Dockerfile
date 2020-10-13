# FROM python:3.7.4-slim-buster

# ##################
# # Configurations #
# ##################

# ARG PYSYFT_CHECKPOINT=PySyft-v0.2.4_aisg
# ARG SYFT_PROTO_CHECKPOINT=syft-proto-v0.2.5.a1_aisg

# ##########################################################
# # Step 1: Install Linux dependencies for source-building #
# ##########################################################

# RUN apt-get update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         build-essential \
#         libz-dev \
#         libbz2-dev \
#         libc6-dev \
#         libdb-dev \
#         libgdbm-dev \
#         libffi-dev \
#         liblzma-dev \
#         libncursesw5-dev \
#         libreadline-gplv2-dev \
#         libsqlite3-dev \
#         libssl-dev \
#         tk-dev \
#         git \
#         unzip

# #########################################
# # Step 2: Clone, build & setup REST-RPC #
# #########################################

# ADD . /ttp
# WORKDIR /ttp

# RUN pip install --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir -r requirements.txt

# RUN unzip -q '/ttp/etc/*.zip' -d /ttp/etc/tmp \
#  && pip install /ttp/etc/tmp/${PYSYFT_CHECKPOINT} \
#  && pip install /ttp/etc/tmp/${SYFT_PROTO_CHECKPOINT}

# #######################################################
# # Step 3: Expose relevant connection points & run app #
# #######################################################

# EXPOSE 5000
# EXPOSE 8020

# ENTRYPOINT ["python", "./main.py"]

# CMD ["--help"]

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

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ADD . /ttp
WORKDIR /ttp

EXPOSE 5000
EXPOSE 8020
EXPOSE 8080

########################
# New Image - Debugger #
########################

FROM base as debug
RUN pip install ptvsd

WORKDIR /ttp
EXPOSE 5678
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main.py

##########################
# New Image - Production #
##########################

FROM base as prod

WORKDIR /ttp
ENTRYPOINT ["python", "./main.py"]
CMD ["--help"]
