from collections import defaultdict
import csv
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from myforms import LoginForm, RegisterForm
from model import get_user_table

app = Flask(__name__)
app.config.from_object('config-flask')

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# load the user lessons
user_lessons = defaultdict(list)
with open("./data/user-lessons.csv", mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    for row in reader:
        user_lessons[row[0]].append(row[1])

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():

    table = get_user_table()
    return render_template('dashboard.html', name=current_user.username, table=table)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

## contents view
@app.route('/static/<lesson>/index.html')
@login_required
def docs(lesson='hpc'):

    print("check {} for access to lesson".format(current_user.username, lesson))

    return app.send_static_file("{}/index.html".format(lesson))

@app.route('/content')
def content(path="hpc/index.html"):
    """
    sphinx project that resides in /static
    """

    return app.send_static_file("hpc/index.html")

if __name__ == '__main__':
    app.run(debug=True)
