#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import logging
import uuid
from datetime import datetime

# Libs
import aiohttp
import jsonschema

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload, AssociationRecords
from rest_rpc.connection.core.datetime_serialization import DateTimeSerializer


##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']

"""
These are the subject-id-class mappings for the main utility records in 
Prediction:
{
    'Statistic': {
        'id': 'stat_id',
        'class': StatsRecords
    }
}
"""

#################################################
# Data Storage Association class - StatsRecords #
#################################################

class StatsRecords(AssociationRecords):
    def __init__(self, subject, identifier, db_path=db_path, relations=[], *associations):
        super().__init__(subject, identifier, db_path=db_path, relations=relations, *associations)
