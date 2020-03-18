FROM python:3.6.9-buster

COPY requirements.txt ./
RUN pip install -r requirements.txt

ADD . /ttp
WORKDIR /ttp

EXPOSE 5000
EXPOSE 8020

ENTRYPOINT ["python", "./run.py"]

CMD ["--help"]