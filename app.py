import pickle
from flask import Flask, render_template, request, jsonify
from normalization import text_cleaning, text_preprocessing

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open(r'model\softvote_model.pkl', 'rb'))
vectorizer = pickle.load(open(r'model\tfidf.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Predict input text and generate result of prediction
def predict(text):
    cleaned_text = text_cleaning(text)
    processed_text = text_preprocessing(cleaned_text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = 'Fake' if model.predict(text_tfidf.toarray()) == 0 else 'Real'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()