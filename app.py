from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load Model
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Load Dataset
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fake News Detection Function
def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    vectorized_input_data = tfvect.transform([news])
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Page Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('predict.html', prediction=pred)
    return render_template('predict.html', prediction=None)

# Run App
if __name__ == '__main__':
    app.run(debug=True)
