#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import logging

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
    
    app.run(host="0.0.0.0", port=5000, debug=False)