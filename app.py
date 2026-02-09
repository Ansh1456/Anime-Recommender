from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Rename relevant columns if needed
df.rename(columns={
    'name': 'title',
    'genres': 'genre',
    'premiered': 'year',
    'score': 'rating'
}, inplace=True)

# Fill missing genres
df['genre'] = df['genre'].fillna('')

# TF-IDF on genres
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(df['genre'])
cos_sim = cosine_similarity(genre_matrix)

# List of genres and years
all_genres = sorted(set(g.strip() for g_list in df['genre'] for g in g_list.split(',') if g))
all_years = sorted(df['year'].dropna().unique())

def recommend(title):
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cos_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        return df.iloc[[i[0] for i in sim_scores]]
    except:
        return pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        title = request.form.get('title', '')
        genre = request.form.get('genre', '')
        year = request.form.get('year', '')
        rating = float(request.form.get('rating', 0))

        filtered = df.copy()

        if title:
            filtered = filtered[filtered['title'].str.lower().str.contains(title.lower())]
        if genre:
            filtered = filtered[filtered['genre'].str.contains(genre)]
        if year:
            filtered = filtered[filtered['year'].astype(str).str.contains(year)]
        filtered = filtered[filtered['rating'] >= rating]

        if not filtered.empty:
            base_anime = filtered.iloc[0]
            recs = recommend(base_anime['title'])
            results = recs[['title', 'genre', 'rating', 'year']].to_dict(orient='records')

    return render_template('index.html', genres=all_genres, years=all_years, results=results)

if __name__ == '__main__':
    app.run(debug=True)
    
