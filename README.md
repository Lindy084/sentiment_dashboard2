## 🚀 Live Demo

[Click here to use the Sentiment Analysis Dashboard](https://sentimentdashboard2-48pmkpdk6zxouxnqyfsttg.streamlit.app/)
Here’s a complete and professional `README.md` for your **Sentiment Dashboard** project hosted at [Lindy084/sentiment\_dashboard2](https://github.com/Lindy084/sentiment_dashboard2.git):

---

```markdown
# 📊 Sentiment Analysis Dashboard

This project is a web-based **Sentiment Analysis Dashboard** built with **Python**, **Streamlit**, and **transformers (Hugging Face)**. It allows users to input multiple text samples, analyze their sentiment (positive, negative, or neutral), view confidence scores, extract keywords, and visualize the results with intuitive charts.

---

## 🚀 Features

- 🔍 Multi-class Sentiment Classification** (Positive, Negative, Neutral)
- 📈 Confidence Score Visualization**
- 🧠 Keyword Extraction** for each input text
- 📊 Bar Chart of Sentiment Distribution**
- 📋 Batch Text Input**
- 💾 Export Results** to CSV
- 🤖 Powered by Transformers (Hugging Face)**

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- Pandas
- Matplotlib
- Seaborn
- NLTK

---

## 📂 Project Structure

```

sentiment\_dashboard2/
│
├── app.py                  # Main Streamlit application
├── utils.py                # Helper functions for sentiment and keyword extraction
├── requirements.txt        # Required Python packages
└── sample\_data.csv         # Optional: Sample input data for testing

````

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Lindy084/sentiment_dashboard2.git
cd sentiment_dashboard2
````

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 5. Run the App

```bash
streamlit run app.py
```

---


## 🧪 How to Use

1. **Input multiple texts** (one per line) in the provided text box.
2. Click **Analyze** to view:

   * Sentiment label for each sentence
   * Confidence score
   * Extracted keywords
   * Bar chart showing sentiment counts
3. Use the **Download CSV** button to export results.


---

## 📌 Example Input

```
I love this app!
The service was terrible.
It was okay, nothing special.
I'm not happy with the results.
```

---

## ✅ Example Output

| Text                            | Sentiment | Confidence | Keywords               |
| ------------------------------- | --------- | ---------- | ---------------------- |
| I love this app!                | Positive  | 98.7%      | love, app              |
| The service was terrible.       | Negative  | 95.2%      | service, terrible      |
| It was okay, nothing special.   | Neutral   | 76.4%      | okay, nothing, special |
| I'm not happy with the results. | Negative  | 88.1%      | happy, results         |

---

## 🧠 Model Info

* **Model Used:** `nlptown/bert-base-multilingual-uncased-sentiment`
* **Library:** `transformers` by Hugging Face
* **Keyword Extraction:** `nltk` + simple filtering

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author

**Lindy Ndlazi**
🌍 Soweto, South Africa
💬 Passionate about AI, Data, and Technology
🐙 GitHub: [@Lindy084](https://github.com/Lindy084)

---

## ✨ Acknowledgements

* [Streamlit](https://streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [NLTK](https://www.nltk.org/)

```

