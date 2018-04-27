from flask import Flask, render_template, request
from model_training import getIssueFeatures, getSimilarIssues
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'adjajoo'


@app.route('/')
def home():
   return render_template('index.html')

@app.route('/issuesimilarity', methods=["GET", "POST"])
def show_similar_issues():
   issue_text = request.args.get('issue', default=None)
   if issue_text != None:
       print('Issue:', issue_text)
       issue_vector = getIssueFeatures(issue_text)
       similar_issues = getSimilarIssues(issue_vector)
   df = pd.DataFrame(np.random.randint(0, 100, size=(5, 5)), columns=["Work Order", "Text", "Cost", "Building", "Days to Complete"])
   return render_template('similar_issues_ajax.html', data=similar_issues.to_html())

if __name__ == '__main__':
   app.run(debug = True)