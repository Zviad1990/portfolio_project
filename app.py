from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
import pandas as pd


# Пробный запуск Flask

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/a")
def hello():
    return "Hello Wo rld!"

if __name__ == '__main__':
    app.run()