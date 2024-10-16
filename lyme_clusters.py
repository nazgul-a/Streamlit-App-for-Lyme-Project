import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score, calinski_harabasz_score


data_path = "https://raw.githubusercontent.com/nazgul-a/lyme_analysis_app/refs/heads/main/synthetic_abstracts.csv"
raw_abstracts = pd.read_csv(data_path)

# Rest of your original code follows...


# Download stopwords from NLTK if not already downloaded
nltk.download('stopwords')

# Define function for generating word cloud
def generate_wordcloud(threshold):
    custom_stop_words = stopwords.words('english') + ['burgdorferi', 'sensu', 'lato', 'b', 's', 'l', 'ixodes', 'scapularis']

    # N-gram extraction
    vectorizer = CountVectorizer(
        ngram_range=(2, 3), 
        stop_words=custom_stop_words, 
        max_features=5000, 
        token_pattern=r'\b[A-Za-z]+\b', 
        max_df=0.90,
        min_df=5
    )    
    X_ngrams = vectorizer.fit_transform(raw_abstracts['abstract'])
    
    # Apply TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_ngrams)
    
    # Get feature names and scores
    ngrams = vectorizer.get_feature_names_out()
    tfidf_scores = X_tfidf.toarray()
    
    # Filter n-grams based on threshold
    important_terms = np.where(np.max(tfidf_scores, axis=0) > threshold)[0]
    important_ngrams = [ngrams[i] for i in important_terms]
    important_scores = np.max(tfidf_scores[:, important_terms], axis=0)
    
    # Create frequency dictionary
    ngram_freq_dict = dict(zip(important_ngrams, important_scores))

    # Generate and display word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_freq_dict)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Define function for LDA and clustering visualization
def lda_and_clustering(num_topics, num_clusters):
    # N-gram extraction
    custom_stop_words = stopwords.words('english') + ['burgdorferi', 'sensu', 'lato', 'b', 's', 'l', 'ixodes', 'scapularis']

    vectorizer = CountVectorizer(
        ngram_range=(2, 3), 
        stop_words=custom_stop_words, 
        max_features=5000, 
        token_pattern=r'\b[A-Za-z]+\b', 
        max_df=0.90,
        min_df=5
    )    
    X_ngrams = vectorizer.fit_transform(raw_abstracts['abstract'])
    
    # Apply TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_ngrams)

    # Apply LDA for topic modeling
    lda_model = LatentDirichletAllocation(n_components=num_topics, 
        random_state=42, learning_method='online')
    lda_topics = lda_model.fit_transform(X_tfidf)

    # Apply UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(lda_topics)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=100, 
        max_iter=1000)
    clusters = kmeans.fit_predict(umap_embeddings)
    
    # Calculate metrics
    silhouette = silhouette_score(umap_embeddings, clusters)
    calinski_harabasz = calinski_harabasz_score(umap_embeddings, clusters)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=clusters, cmap='Spectral', s=10)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title("Clustering Based on UMAP Embeddings")
    st.pyplot(fig)

    # Display metrics
    st.write(f"**Silhouette Score:** {silhouette:.2f}")
    st.write(f"**Calinski-Harabasz Index:** {calinski_harabasz:.2f}")

# Streamlit layout
st.title("On the way to find new features (with clustering)")

# Word Cloud section
st.header("N-Grams Word Cloud")
threshold = st.slider("Select TF-IDF Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
if st.button("Generate Word Cloud"):
    generate_wordcloud(threshold)

# LDA and Clustering section
st.header("LDA Topic Modeling and Clustering")
num_topics = st.slider("Select Number of Topics", min_value=2, max_value=30, value=15)
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
if st.button("Generate Clustering Visualization"):
    lda_and_clustering(num_topics, num_clusters)
