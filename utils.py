from transformers import pipeline

# Load the model once
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Basic keyword extraction (hardcoded for now)
def extract_keywords(text):
    keywords = []
    for word in text.lower().split():
        if word in ["love", "hate", "terrible", "great", "bad", "amazing", "happy", "sad", "disappointed", "okay"]:
            keywords.append(word)
    return ", ".join(set(keywords)) if keywords else "None"

# Main function that returns a list of results
def batch_analyze(texts):
    if not texts:
        return []

    results = sentiment_model(texts)

    output = []
    for i, res in enumerate(results):
        output.append({
            "text": texts[i],
            "sentiment": res["label"].upper(),   # Force upper
            "confidence": round(res["score"], 2),
            "keywords": extract_keywords(texts[i])
        })

    return output
