#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from datetime import datetime, timedelta
from dateutil.tz import tzutc, tzlocal

# Libs
from tinydb_serialization import Serializer

############################################
# Serialization Class - DateTimeSerializer #
############################################

class DateTimeSerializer(Serializer):
    """
    Serialises both aware and naive datetime objects for storage into TinyDB.

    Encoding - If the datetime object is aware, it is first converted to UTC and
               then encoded with an 'A' appended to the serialization. Otherwise
               it is serialized without conversion and an 'N' is appended.
    Decoding - If the serialization ends with 'A', the datetime object is 
               treated as UTC and then converted to localtime. Otherwise, the 
               datetime object is treated as localtime and no conversion is 
               necessary.

    Note: Microseconds are discarded but hours, minutes & seconds are preserved.
    """
    OBJ_CLASS = datetime

    def encode(self, obj):
        """ Serialize naive datetimes objects without conversion but with 'N' 
            for 'Naive' appended. Convert aware datetime objects to UTC and then
            serialize them with 'A' for 'Aware' appended.
        """
        if obj.tzinfo is None:
            return obj.strftime('%Y-%m-%d %H:%M:%S N')
        else:
            return obj.astimezone(tzutc()).strftime('%Y-%m-%d %H:%M:%S A')

    def decode(self, s):
        """ Return the serialization as a datetime object. If the serializaton 
            ends with 'A',  first converting to localtime and returning an aware 
            datetime object. If the serialization ends with 'N', returning 
            without conversion as a naive datetime object. 
        """
        if s[-1] == 'A':
            return datetime.strptime(
                s[:-2], 
                '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=tzutc()).astimezone(tzlocal())
        else:
            return datetime.strptime(s[:-2], '%Y-%m-%d %H:%M:%S')


class TimeDeltaSerializer(Serializer):
    """
    Handles serialisations of timedeltas (for checking datelines)
    """
    OBJ_CLASS = timedelta  # The class handles timedelta objects

    def encode(self, obj):
        """ Serialize the timedelta object as days.seconds. """
        return "{0}.{1}".format(obj.days, obj.seconds)

    def decode(self, s):
        """ Return the serialization as a timedelta object """
        days_seconds = (int(x) for x in s.split('.')) 
        return timedelta(*days_seconds)