from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import kagglehub
import os

app = Flask(__name__)

# Global variables to store model and data
iris_df = None
knn = None
accuracy = None

def load_data():
    # Load and prepare the Iris dataset, then find the best n_neighbors
    global iris_df, knn, accuracy

    # Load the CSV file
    url = kagglehub.dataset_download("samybaladram/iris-dataset-extended")
    iris_df = pd.read_csv(os.path.join(url, 'iris_extended.csv'))
    iris_df = iris_df[['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    # Prepare the data
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['species']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Search for the best number of neighbors (k)
    param_grid = {'n_neighbors': list(range(1, 31))}
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    # Assign the best estimator and compute accuracy on the test set
    best_k = grid_search.best_params_['n_neighbors']
    knn = grid_search.best_estimator_
    print(f"Best number of neighbors: {best_k}")

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

# Load data when the app starts
load_data()

@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle prediction requests
    data = request.json
    
    # Extract features from the request
    features = pd.DataFrame({
        'sepal_length': [data['sepal_length']],
        'sepal_width': [data['sepal_width']],
        'petal_length': [data['petal_length']],
        'petal_width': [data['petal_width']]
    })
    
    # Make prediction
    prediction = knn.predict(features)
    prediction_proba = knn.predict_proba(features)
    
    # Map prediction to readable name with emoji
    species_map = {
        'setosa': 'Setosa 🌸',
        'versicolor': 'Versicolor 🌼',
        'virginica': 'Virginica 🌺'
    }
    predicted_species = species_map[prediction[0]]
    
    # Prepare probability data
    probabilities = {}
    for i, species in enumerate(knn.classes_):
        probabilities[species] = round(prediction_proba[0][i], 4)
    
    return jsonify({
        'prediction': predicted_species,
        'probabilities': probabilities,
        'accuracy': round(accuracy, 2)
    })

@app.route('/data')
def get_data():
    # Return the raw iris data
    return jsonify({
        'data': iris_df.to_dict('records'),
        'columns': iris_df.columns.tolist(),
        'total_rows': len(iris_df),
        'total_columns': len(iris_df.columns)
    })

if __name__ == '__main__':
    load_data()  # Ensure data is loaded before starting the app
    app.run(debug=True)