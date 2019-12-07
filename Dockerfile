FROM python:3.7

LABEL image classification

COPY ./requirements.txt /api/requirements.txt

WORKDIR /api

RUN pip install -r requirements.txt

COPY . /api

COPY ./project3/api/imagenet_class_index.json /api/project3/api/imagenet_class_index.json

CMD python project3/api/api.py
