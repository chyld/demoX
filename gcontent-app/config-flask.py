import os
__basedir__ = os.path.realpath(os.path.dirname(__file__))
WTF_CSRF_ENABLED = True
SECRET_KEY = 'rebbadiptoal@bblelabeltiecabletoatr$$'
SQLALCHEMY_DATABASE_URI = os.path.join("sqlite:///" + os.path.join(__basedir__,"database.db"))
SQLALCHEMY_TRACK_MODIFICATIONS = False
