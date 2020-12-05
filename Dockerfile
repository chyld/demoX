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

ARG CADDY_FILE=caddy_2.3.0-beta.1_linux_amd64.deb
ARG CADDY_DIR=v2.3.0-beta.1

RUN wget "https://github.com/caddyserver/caddy/releases/download/$CADDY_DIR/$CADDY_FILE"
RUN dpkg -i $CADDY_FILE
RUN rm $CADDY_FILE

EXPOSE 80
CMD caddy file-server --root ./static --browse

