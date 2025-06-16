from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "I love this app!",
    "The service was terrible.",
    "It was okay, nothing special.",
    "I’m not happy with the results."
]

results = sentiment_model(texts)

for i, res in enumerate(results):
    print(f"Text: {texts[i]}")
    print(f"Sentiment: {res['label']}, Confidence: {res['score']:.2f}\n")
