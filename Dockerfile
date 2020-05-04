FROM python:3.7.4-slim-buster

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . /ttp
WORKDIR /ttp

EXPOSE 5000
EXPOSE 8020

ENTRYPOINT ["python", "./main.py"]

CMD ["--help"]