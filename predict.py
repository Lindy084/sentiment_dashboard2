from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model once
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_text(text):
    result = sentiment_model(text)
    return result[0]['label'], round(result[0]['score'], 2)

def batch_analyze(texts):
    results = sentiment_model(texts)
    output = []
    for i, res in enumerate(results):
        output.append({
            "text": texts[i],
            "sentiment": res["label"],
            "confidence": round(res["score"], 2)
        })
    return output

def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])  # ✅ FIXED: wrap in list
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_indices]
    return ", ".join(keywords)
