from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

@app.route('/',methods = ['POST'])
def func():




if __name__ == "__main__":
    app.run(debug=True)