import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
from fpdf import FPDF

# Load sentiment pipeline once
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

# Extract keywords using TF-IDF
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_indices]
    return ", ".join(keywords)

# Batch analyze texts
def batch_analyze(texts):
    results = sentiment_model(texts)
    output = []
    for i, res in enumerate(results):
        output.append({
            "text": texts[i],
            "sentiment": res["label"],
            "confidence": round(res["score"], 2),
            "keywords": extract_keywords(texts[i])
        })
    return pd.DataFrame(output)

# PDF export helper
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Sentiment Analysis Report", ln=True, align="C")
        self.ln(10)

def convert_df_to_pdf(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for _, row in df.iterrows():
        text_block = f"Text: {row['text']}\nSentiment: {row['sentiment']}\nConfidence: {row['confidence']}\nKeywords: {row['keywords']}\n"
        # handle encoding for special chars
        text_block = text_block.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, text_block)
        pdf.ln()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name

# Streamlit UI
st.title("🧠 Sentiment Analysis Dashboard")
st.write("Analyze the sentiment of your texts with confidence scores and keywords.")

# Choose input method
input_method = st.radio("Choose Input Method:", ("Direct Text", "Upload File"))

texts = []
if input_method == "Direct Text":
    raw_text = st.text_area("Enter one or more texts (each on a new line):")
    if raw_text:
        texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
else:
    uploaded_file = st.file_uploader("Upload a .txt or .csv file with texts (one text per line or a 'text' column)", type=["txt", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                if "text" in df.columns:
                    texts = df["text"].dropna().astype(str).tolist()
                else:
                    st.error("CSV file must contain a 'text' column.")
            else:  # txt file
                content = uploaded_file.read().decode("utf-8")
                texts = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

if texts:
    with st.spinner("Analyzing..."):
        result_df = batch_analyze(texts)
    st.success("✅ Analysis Complete!")

    # Show results table
    st.subheader("📋 Results Table")
    st.dataframe(result_df)

    # Export section
    st.subheader("📥 Export Results")

    # CSV export
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Download as CSV", data=csv, file_name="sentiment_results.csv", mime='text/csv')

    # JSON export
    json_data = result_df.to_json(orient="records", indent=2)
    st.download_button("🧾 Download as JSON", data=json_data, file_name="sentiment_results.json", mime="application/json")

    # PDF export
    pdf_path = convert_df_to_pdf(result_df)
    with open(pdf_path, "rb") as f:
        st.download_button("📘 Download as PDF", data=f, file_name="sentiment_report.pdf", mime="application/pdf")

    # Sentiment distribution plot
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = result_df["sentiment"].value_counts()
    if not sentiment_counts.empty:
        fig, ax = plt.subplots()
        colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
        sentiment_counts.plot(kind="bar", ax=ax, color=[colors.get(x, "blue") for x in sentiment_counts.index])
        ax.set_ylabel("Number of Texts")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)
    else:
        st.info("No sentiment data to display.")

    # Explanation feature
    st.subheader("🔍 Confidence Score Explanation")
    st.write(
        """
        The confidence score indicates how sure the AI model is about its sentiment prediction.
        Scores closer to 1.0 mean higher certainty. For example, a confidence of 0.95 means the model is
        95% confident this text expresses the predicted sentiment.
        """
    )

else:
    st.info("Please enter or upload texts to analyze.")

