# FROM python:3.7-alpine
# WORKDIR graylog
# # ENV FLASK_APP=app.py
# # ENV FLASK_RUN_HOST=0.0.0.0
# # RUN apk add --no-cache gcc musl-dev linux-headers
# # COPY requirements.txt requirements.txt
# COPY graylog graylog
# # RUN pip install -r requirements.txt
# EXPOSE 9006
# # COPY . .
# # CMD ["flask", "run"]

FROM ubuntu:latest

CMD tail -f /dev/null

RUN apt-get update -y && apt-get install -y python3-pip python-dev

EXPOSE 9001

WORKDIR /graylog

COPY SynergosLogger SynergosLogger
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY psutil psutil
COPY README.md README.md

RUN pip3 install -r requirements.txt
RUN pip3 install .
