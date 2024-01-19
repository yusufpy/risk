from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained models
model_amounts = RandomForestClassifier(n_estimators=100, random_state=42)
model_behavior = KNeighborsClassifier(n_neighbors=5)
model_history = LogisticRegression(random_state=42)
ensemble_model = VotingClassifier(estimators=[
   ('amounts', model_amounts),
   ('behavior', model_behavior),
   ('history', model_history)
], voting='soft')

# Load the label encoder and fit it on the training data
label_encoder = LabelEncoder()
df = pd.read_csv('risk_assessment.csv')

df['Time of Day']=label_encoder.fit_transform(df['Time of Day'])
df['Location']=label_encoder.fit_transform(df['Location'])
df['Service Type']=label_encoder.fit_transform(df['Service Type'])
df['Provider']=label_encoder.fit_transform(df['Provider'])
#df['Historical Transaction Amount']=label_encoder.fit_transform(df['Historical Transaction Amount'])
df['Risk']=label_encoder.fit_transform(df['Risk'])

ensemble_model.fit(df[['Time of Day','Location','Frequency','Service Type','Historical Transaction Amount','Transaction Amount']], df['Risk'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Transform input data using the fitted label encoder
        data['Time of Day'] = label_encoder.fit_transform([data['Time of Day']])[0]
        data['Location'] = label_encoder.fit_transform([data['Location']])[0]
        data['Service Type'] = label_encoder.fit_transform([data['Service Type']])[0]
        #data['Provider'] = label_encoder.fit_transform([data['Provider']])[0]
        #data['Historical Transaction Amount'] = label_encoder.fit_transform([data['Historical Transaction Amount']])[0]

        # Make predictions using the ensemble model
        risk_score = ensemble_model.predict_proba([[
            data['Transaction Amount'],
            data['Historical Transaction Amount'],
            data['Frequency'],
            data['Time of Day'],
            data['Location'],
            data['Service Type']
        ]])[:, 1].item()

        return jsonify({'risk_score': risk_score})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
