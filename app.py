from flask import Flask, request, render_template
import pandas as pd
from model.recommendation_system import search, find_similar_movies

app = Flask(__name__)

# Load datasets
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    results = search(title)
    if not results.empty:
        movie_id = results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
        recommendations = recommendations.to_dict(orient='records')
    else:
        recommendations = []

    return render_template('index.html', title=title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
