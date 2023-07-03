# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import pickle
import openai
from chat_functions import answer_query_with_context  

app = Flask(__name__)

# Load 'new_df' from a pickle file
with open('new_df.pkl', 'rb') as f:
    new_df = pickle.load(f)

# Load 'document_embeddings' from a pickle file
with open('document_embeddings.pkl', 'rb') as f:
    document_embeddings = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return answer_query_with_context(userText, new_df, document_embeddings)

if __name__ == "__main__":
    app.run(port=5000)
