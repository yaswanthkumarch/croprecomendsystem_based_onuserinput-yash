from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

df = pd.read_csv('crop_recommendation.csv')

encode_label = LabelEncoder()
df['label'] = encode_label.fit_transform(df['label'])

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=1000, random_state=123)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    pH = float(request.form['pH'])
    rainfall = float(request.form['rainfall'])

    input_features = [N, P, K, temperature, humidity, pH, rainfall]
    data = scaler.transform([input_features])

    predicted_label = model.predict(data)
    predicted_crop = encode_label.inverse_transform(predicted_label)

    return render_template('index.html', prediction_text='U Are Sutiable To Grow : {}'.format(predicted_crop))

if __name__ == "__main__":
    app.run(debug=True)
