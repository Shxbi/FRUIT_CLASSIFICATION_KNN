import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_csv('fruits_2.csv')

fruits = pd.get_dummies(fruits, columns=['color', 'shape', 'skin_texture',
                                        'firmness', 'aroma_type', 'fleshiness',
                                        'growing_season', 'country_of_origin'])

X = fruits.drop('label', axis=1)
y = fruits['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

def preprocess_input(weight, color, size, shape, skin_texture, firmness, aroma_type, fleshiness, growing_season, country_of_origin):
    input_data = pd.DataFrame({
        'weight(gms)': [weight],
        'color': [color],
        'size(cms)': [size],
        'shape': [shape],
        'skin_texture': [skin_texture],
        'firmness': [firmness],
        'aroma_type': [aroma_type],
        'fleshiness': [fleshiness],
        'growing_season': [growing_season],
        'country_of_origin': [country_of_origin]
    })
    input_data = pd.get_dummies(input_data, columns=['color', 'shape', 'skin_texture',
                                                    'firmness', 'aroma_type', 'fleshiness',
                                                    'growing_season', 'country_of_origin'])
    
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    return input_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        color = request.form['color']
        size = float(request.form['size'])
        shape = request.form['shape']
        skin_texture = request.form['skin_texture']
        firmness = request.form['firmness']
        aroma_type = request.form['aroma_type']
        fleshiness = request.form['fleshiness']
        growing_season = request.form['growing_season']
        country_of_origin = request.form['country_of_origin']

        input_data = preprocess_input(weight, color, size, shape, skin_texture,
                                      firmness, aroma_type, fleshiness,
                                      growing_season, country_of_origin)

        knn_prediction = knn.predict(input_data)[0]
        #dt_prediction = dt.predict(input_data)[0]

        return render_template('index.html', knn_prediction=knn_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    