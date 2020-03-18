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
from flask import Flask, Blueprint
from flask_restx import Api

# Custom
from apis import api

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

app = Flask(__name__)
api.init_app(app)
#blueprint = Blueprint('api', __name__, url_prefix='/ttp')
#api = Api(blueprint)
#app.register_blueprint(blueprint)

###########
# Scripts #
###########

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
