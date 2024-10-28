import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load data (replace with real dataset)
df = pd.DataFrame({
    'lead_time': [5, 10, 15, 7, 20, 3, 14, 6],
    'demand_variability': [0.1, 0.3, 0.5, 0.2, 0.7, 0.1, 0.4, 0.3],
    'transport_cost': [100, 200, 150, 120, 300, 90, 210, 130],
    'inventory_cost': [300, 400, 500, 320, 600, 310, 450, 330],
    'cost_effective': [1, 0, 0, 1, 0, 1, 0, 1],  # 1 for cost-effective, 0 for not
    'total_cost': [400, 600, 500,300,700,200,600,900]

})

X = df[['lead_time', 'demand_variability', 'transport_cost', 'inventory_cost']]
y = df['total_cost']



# Train a simple model
model = RandomForestRegressor()
model.fit(X, y)



# Save the model
file_name='model/supply_chain_model.pkl'
pickle.dump(model, open(file_name,'wb'))


from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open(file_name,'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    lead_time = float(request.form.get('lead_time'))
    demand_variability = float(request.form.get('demand_variability'))
    transport_cost = float(request.form.get('transport_cost'))
    inventory_cost = float(request.form.get('inventory_cost'))

    # Create DataFrame
    data = pd.DataFrame({
        'lead_time': [lead_time],
        'demand_variability': [demand_variability],
        'transport_cost': [transport_cost],
        'inventory_cost': [inventory_cost]
    })

    # Predict total cost
    prediction = model.predict(data)[0]
    


    return jsonify({'total_cost': prediction})

if __name__ == '__main__':
    app.run(debug=True)

