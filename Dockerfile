FROM python:3

# do not buffer output to console
ENV PYTHONUNBUFFERED 1

RUN apt update && apt install -y pandoc

RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/
RUN pip install -r requirements.txt

COPY . /code/
RUN python scripts/build-all.py

