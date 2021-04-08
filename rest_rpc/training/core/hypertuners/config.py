#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
from string import Template

# Libs


# Custom

##################
# Configurations #
##################

# Define a prefix pattern that uniquely differentiates a optimisation run from
# a standard user-defined one
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")