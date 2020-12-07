# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #

FROM python:3 as build

# do not buffer python output to console
ENV PYTHONUNBUFFERED 1

RUN apt update && apt install -y pandoc

RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/
RUN pip install -r requirements.txt

COPY . /code/
RUN python scripts/build-all.py

# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #

FROM caddy:latest as production

WORKDIR /app

COPY --from=build /code/static .

CMD caddy file-server --root . --browse

