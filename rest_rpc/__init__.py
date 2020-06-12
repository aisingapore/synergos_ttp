#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Flask, Blueprint
from flask_restx import Api

# Custom

##################
# Configurations #
##################

app = Flask(__name__)

app.config.from_object('config')

from .connection import blueprint as connection_api
from .training import blueprint as training_api
from .evaluation import blueprint as evaluation_api

app.register_blueprint(connection_api, url_prefix='/ttp/connect')
app.register_blueprint(training_api, url_prefix='/ttp/train')
app.register_blueprint(evaluation_api, url_prefix='/ttp/evaluate')