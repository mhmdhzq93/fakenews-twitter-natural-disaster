## Natural Disaster Fake Tweets Detection Web Application
This is a repository of research project (WQD7002) consisting of datasets, EDA notebook, classifier module and application source codes.
The model is deployed as web application using Flask

### Objective
1) To implement supervised learning approach and Natural Language Processing (NLP) techniques on the natural disaster tweets.
2) To predict the authenticity of natural disaster tweets using models as built in (i) and evaluate the accuracy of the models.

### Dataset
Source of dataset is obtained and extracted from https://www.kaggle.com/datasets/vstepanenko/disaster-tweets

### How to deploy the model as web app (Flask)
1) Create a conda virtual environment with Python 3
2) Install the required packages from requirements.txt
3) Run the model.ipynb jupyter notebook to generate the model in pickle file format (i.e. model.pkl)
3) Run the app.py pyscript file
4) Copy and paste the given Flask IP adress in your browser
5) Done!

#### Disclaimer
Unfortunately, the pickle model file is too large to be stored in GitHub (i.e. 130MB)
