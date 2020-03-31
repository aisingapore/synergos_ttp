#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import os
import logging
import random
from collections import OrderedDict
from pathlib import Path

# Libs


# Custom
from rest_rpc import app

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

###########
# Scripts #
###########

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")