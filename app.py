# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import string
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression  # تم التعديل هنا
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import gradio as ui
# import matplotlib.pyplot as plt
# from thefuzz import process

# # 1. تنزيل الحزم الأساسية من nltk للسيرفر السحابي
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # إعداد أدوات تنظيف النص
# stop_words = set(stopwords.words('english'))
# negation_words = {'not', 'no', 'nor', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "never"}
# stop_words = stop_words.difference(negation_words)
# lemmatizer = WordNetLemmatizer()

# # 2. تحميل البيانات (تأكد أن ملف Reviews.csv مرفوع في المساحة)
# print("جاري تحميل البيانات...")
# df = pd.read_csv('Reviews.csv', nrows=100000)
# df = df[['Text', 'Summary', 'Score']].dropna()

# # 3. دوال المعالجة
# def map_score_to_sentiment(score):
#     if score in [1, 2]: return 'negative'
#     elif score == 3: return 'neutral'
#     else: return 'positive'

# df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)

# def clean_html_tags(text):
#     cleaned_text = re.sub(r'<.*?>', ' ', str(text))
#     return ' '.join(cleaned_text.split())

# def preprocess_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     tokens = word_tokenize(text)
#     cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return " ".join(cleaned_tokens)

# print("جاري تنظيف ومعالجة النصوص...")
# df['Cleaned_Text'] = df['Text'].apply(clean_html_tags)
# df['Processed_Text'] = df['Cleaned_Text'].apply(preprocess_text)

# # 4. بناء وتدريب نموذج Logistic Regression
# print("جاري تدريب نموذج تحليل المشاعر...")
# X = df['Processed_Text']
# y = df['Sentiment']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# X_train_tfidf = tfidf.fit_transform(X_train)

# # تم التعديل هنا لتدريب الموديل الجديد
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
# model.fit(X_train_tfidf, y_train)

# # 5. تحميل نموذج التلخيص (Transformers)
# print("جاري تحميل نموذج التلخيص...")
# model_name = "sshleifer/distilbart-cnn-12-6"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # def analyze_and_summarize_review(raw_text):
# #     cleaned = clean_html_tags(raw_text)
# #     processed = preprocess_text(cleaned)
    
# #     text_tfidf = tfidf.transform([processed])
# #     sentiment_prediction = model.predict(text_tfidf)[0]
    
# #     if len(raw_text.split()) > 10:
# #         inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
# #         summary_ids = sum_model.generate(inputs["input_ids"], max_length=30, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
# #         summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# #     else:
# #         summary_text = "النص قصير جداً ولا يحتاج إلى تلخيص."
        
# #     return {
# #         "الملخص الذكي (Summary)": summary_text,
# #         "تحليل المشاعر (Sentiment)": sentiment_prediction
# #     }
# def analyze_and_summarize_review(raw_text):
#     cleaned = clean_html_tags(raw_text)
#     processed = preprocess_text(cleaned)
    
#     # 1. تحليل المشاعر
#     text_tfidf = tfidf.transform([processed])
#     sentiment_prediction = model.predict(text_tfidf)[0]
    
#     # 2. حساب عدد كلمات النص المدخل
#     word_count = len(raw_text.split())
    
#     # الشرط الحاسم الذي طلبته: إذا كان أكبر من 15 كلمة يلخص، غير ذلك يطبع الرسالة
#     if word_count > 10:
#         inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
        
#         # بارامترات صارمة تمنع النموذج من زيادة أي كلام أو تأليف
#         summary_ids = sum_model.generate(
#             inputs["input_ids"], 
#             max_new_tokens=20,           # حد أقصى صغير جداً للملخص ليجبره على الاختصار الشديد
#             no_repeat_ngram_size=2,      # يمنع تكرار العبارات نهائياً
#             num_beams=2,                 # تقليل عدد الاحتمالات لمنع التخريف
#             early_stopping=True
#         )
#         summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
#         # حماية إضافية: لو الموديل هلوَس برضه وطلع التلخيص أطول أو مساوي للنص الأصلي، يعرض النص الأصلي فقط
#         if len(summary_text.split()) >= word_count:
#             summary_text = raw_text.strip()
#     else:
#         # إذا كان 15 كلمة أو أقل، يطبع هذه الرسالة مباشرة ولا يشغل نموذج التلخيص
#         summary_text = "The text is too short to require summarization. / النص قصير جداً ولا يحتاج إلى تلخيص."
        
#     return {
#         "الملخص الذكي (Summary)": summary_text,
#         "تحليل المشاعر (Sentiment)": sentiment_prediction
#     }

# # 6. إعداد دوال واجهة Gradio
# search_history = []

# def generate_dashboard(query):
#     if query and query.strip():
#         filtered_df = df[df['Processed_Text'].str.contains(query.lower(), case=False, na=False)]
#         title = f"Sentiment Distribution for: {query.capitalize()}"
#     else:
#         filtered_df = df
#         title = "General Sentiment Distribution"
        
#     if filtered_df.empty:
#         fig, ax = plt.subplots()
#         ax.text(0.5, 0.5, 'No data found for this feature', ha='center', fontsize=12)
#         ax.axis('off')
#         return fig
        
#     sentiment_counts = filtered_df['Sentiment'].value_counts()
#     fig, ax = plt.subplots(figsize=(6, 6))
#     colors = ['#50fa7b', '#ff5555', '#8be9fd']
    
#     sentiment_counts.plot(
#         kind='pie', 
#         autopct='%1.1f%%', 
#         startangle=140, 
#         colors=colors, 
#         ax=ax,
#         textprops={'fontsize': 12, 'weight': 'bold'}
#     )
    
#     ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
#     ax.set_ylabel('') 
#     plt.tight_layout()
#     return fig

# def web_interface_fn(review_text):
#     if not review_text.strip():
#         return "Please enter a review.", "", "", "", "", ""
    
#     tokens_raw = word_tokenize(review_text.lower())
#     text_no_punct = review_text.lower().translate(str.maketrans('', '', string.punctuation))
#     tokens_no_punct = word_tokenize(text_no_punct)
#     no_stop_words = [w for w in tokens_no_punct if w not in stop_words]
#     lemmatized = [lemmatizer.lemmatize(w) for w in no_stop_words]
    
#     results = analyze_and_summarize_review(review_text) 
    
#     return (
#         results["الملخص الذكي (Summary)"], 
#         results["تحليل المشاعر (Sentiment)"].upper() + (" 😊" if results["تحليل المشاعر (Sentiment)"] == 'positive' else " 😢"),
#         " | ".join(tokens_raw),
#         " | ".join(tokens_no_punct),
#         " | ".join(no_stop_words),
#         " | ".join(lemmatized)
#     )

# def search_in_dataset_with_history(query):
#     global search_history
#     if not query.strip(): return "Please enter a search term.", "\n".join(search_history)
    
#     common_keywords = ['battery', 'screen', 'camera', 'price', 'performance', 'design', 'software', 'quality', 'money']
#     match = process.extractOne(query.lower(), common_keywords)
#     actual_query = match[0] if match and match[1] > 80 else query.lower().strip()

#     if query not in search_history:
#         search_history.insert(0, query)
#         search_history = search_history[:5]
    
#     results = df[df['Text'].str.contains(actual_query, case=False, na=False)]
    
#     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"
#     for idx, row in results.head(5).iterrows():
#         output += f"📝: {str(row['Text'])[:120]}...\n📊: {str(row['Sentiment']).upper()}\n\n"
        
#     return output, "\n".join(search_history)

# # 7. بناء وتشغيل واجهة Gradio
# with ui.Blocks(title="NLP Project", theme=ui.themes.Soft()) as app:
#     ui.Markdown("""
#     <div style="text-align: center; background-color: #1e1e2e; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
#         <h1 style="color: #ff79c6;">💻 NLP Project</h1>
#         <div style="display: flex; justify-content: space-around; color: white; font-size: 16px;">
#             <div>👤 SAIF KHATTAB <br> 🆔 202312556</div>
#             <div>👤 YAZEED AMEEN <br> 🆔 202310058</div>
#         </div>
#     </div>
#     """)
    
#     with ui.Tabs():
#         with ui.TabItem("Analyze Text 📝"):
#             input_text = ui.Textbox(lines=3, label="Enter Review")
#             btn = ui.Button("Analyze 🚀", variant="primary")
            
#             with ui.Row():
#                 out1 = ui.Textbox(label="Summary")
#                 out2 = ui.Textbox(label="Sentiment")
            
#             with ui.Accordion("Text Processing Stages (Transparency Mode)", open=False):
#                 out_tokens = ui.Textbox(label="1. Raw Tokens")
#                 out_no_punct = ui.Textbox(label="1.5 No Punctuation")
#                 out_no_stop = ui.Textbox(label="2. No Stop Words")
#                 out_lemmatized = ui.Textbox(label="3. Lemmatized")
            
#             btn.click(web_interface_fn, input_text, [out1, out2, out_tokens, out_no_punct, out_no_stop, out_lemmatized])
            
#         with ui.TabItem("Search Dataset 🔍"):
#             with ui.Row():
#                 with ui.Column():
#                     query_input = ui.Textbox(label="Search Query")
#                     search_btn = ui.Button("Search Data 🔎")
#                 history_output = ui.Textbox(label="Last 5 Searches", interactive=False)
#             results_out = ui.Textbox(label="Results", lines=8)
#             search_btn.click(search_in_dataset_with_history, query_input, [results_out, history_output])
            
#         with ui.TabItem("Analytics Dashboard 📊"):
#             dash_query = ui.Textbox(label="Filter stats by feature (Optional)", placeholder="e.g., food,tea")
#             dashboard_btn = ui.Button("Show Chart 🚀")
#             plot_output = ui.Plot(label="Sentiment Distribution")
#             dashboard_btn.click(generate_dashboard, inputs=dash_query, outputs=plot_output)

# # السطر الأخير بدون أي إضافات ليعمل على السحابة
# app.launch()
import os
import re
import csv
import string
import pickle
import warnings

import gradio as gr
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from thefuzz import process


# =========================================================
# 1. General settings
# =========================================================

warnings.filterwarnings("ignore")

LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
REVIEWS_FILE_PATH = "Reviews.csv"

SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
SUMMARY_WORD_THRESHOLD = 15
DEFAULT_MAX_LENGTH = 111


# =========================================================
# 2. Download NLTK packages
# =========================================================

for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(package, quiet=True)

stop_words = set(stopwords.words("english"))

# Keep negation words because they affect sentiment strongly.
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words.difference(negation_words)

lemmatizer = WordNetLemmatizer()


# =========================================================
# 3. Text preprocessing
# Must match the preprocessing used during training
# =========================================================

def clean_html_tags(text):
    text = str(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_negations(text):
    text = str(text).replace("’", "'")

    contractions = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "mustn't": "must not",
        "needn't": "need not",
        "mightn't": "might not",
        "shan't": "shall not",
    }

    for contraction, expanded_form in contractions.items():
        text = re.sub(
            rf"\b{re.escape(contraction)}\b",
            expanded_form,
            text,
            flags=re.IGNORECASE,
        )

    return text


def preprocess_text(text):
    text = str(text).lower()
    text = normalize_negations(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:
        if not word.isalpha():
            continue

        if word in stop_words:
            continue

        cleaned_tokens.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned_tokens)


# =========================================================
# 4. Load trained LSTM files
# =========================================================

def require_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file '{file_path}' was not found. "
            "Upload it to the root of the Hugging Face Space."
        )


for required_file in [
    LSTM_MODEL_PATH,
    TEXT_TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
]:
    require_file(required_file)

print("Loading trained LSTM sentiment model...")

sentiment_model = load_model(LSTM_MODEL_PATH)

with open(TEXT_TOKENIZER_PATH, "rb") as tokenizer_file:
    text_tokenizer = pickle.load(tokenizer_file)

with open(LABEL_ENCODER_PATH, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

try:
    model_input_length = sentiment_model.input_shape[1]
    MAX_LENGTH = int(model_input_length) if model_input_length else DEFAULT_MAX_LENGTH
except (TypeError, ValueError, IndexError):
    MAX_LENGTH = DEFAULT_MAX_LENGTH

print(f"LSTM model loaded successfully. MAX_LENGTH = {MAX_LENGTH}")


# =========================================================
# 5. Load pretrained summarization model
# =========================================================

print("Loading DistilBART summarization model...")

summary_tokenizer = AutoTokenizer.from_pretrained(
    SUMMARIZATION_MODEL_NAME
)

summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZATION_MODEL_NAME
)

print("Summarization model loaded successfully.")


# =========================================================
# 6. Optional dataset loading
# =========================================================

def map_score_to_sentiment(score):
    if score in [1, 2]:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def read_csv_safely(file_path, max_valid_rows=100000):
    valid_rows = []
    bad_rows = []

    with open(
        file_path,
        mode="r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(
            file,
            delimiter=",",
            quotechar='"',
            strict=True,
        )

        try:
            columns = next(reader)
        except StopIteration:
            raise ValueError("Reviews.csv is empty.")

        columns = [
            column.strip().replace("\ufeff", "")
            for column in columns
        ]

        expected_columns = len(columns)

        while len(valid_rows) < max_valid_rows:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as error:
                bad_rows.append(
                    {"line": reader.line_num, "error": str(error)}
                )
                continue

            if len(row) != expected_columns:
                bad_rows.append(
                    {
                        "line": reader.line_num,
                        "error": (
                            f"Expected {expected_columns} fields, "
                            f"found {len(row)}."
                        ),
                    }
                )
                continue

            valid_rows.append(row)

    return pd.DataFrame(valid_rows, columns=columns), bad_rows


def load_reviews_dataset():
    if not os.path.exists(REVIEWS_FILE_PATH):
        print(
            "Reviews.csv was not found. "
            "Search and dashboard tabs will be unavailable."
        )
        return None

    try:
        dataset, bad_rows = read_csv_safely(
            REVIEWS_FILE_PATH,
            max_valid_rows=100000,
        )

        required_columns = {"Text", "Summary", "Score"}

        if not required_columns.issubset(dataset.columns):
            print(
                "Reviews.csv is missing required columns. "
                "Search and dashboard tabs will be unavailable."
            )
            return None

        dataset = dataset[["Text", "Summary", "Score"]].copy()

        dataset["Score"] = pd.to_numeric(
            dataset["Score"],
            errors="coerce",
        )

        dataset = dataset.dropna(
            subset=["Text", "Summary", "Score"]
        )

        dataset = dataset[
            dataset["Score"].isin([1, 2, 3, 4, 5])
        ].copy()

        dataset["Score"] = dataset["Score"].astype(int)
        dataset["Sentiment"] = dataset["Score"].apply(
            map_score_to_sentiment
        )

        dataset["Cleaned_Text"] = dataset["Text"].apply(
            clean_html_tags
        )

        dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(
            preprocess_text
        )

        dataset = dataset[
            dataset["Processed_Text"].str.strip() != ""
        ].reset_index(drop=True)

        print(
            f"Reviews.csv loaded: {len(dataset)} valid rows, "
            f"{len(bad_rows)} corrupted rows skipped."
        )

        return dataset

    except Exception as error:
        print(f"Could not load Reviews.csv: {error}")
        return None


df = load_reviews_dataset()


# =========================================================
# 7. Sentiment prediction
# =========================================================

def predict_sentiment(raw_text):
    raw_text = str(raw_text).strip()

    if not raw_text:
        raise ValueError("Please enter a review.")

    cleaned_text = clean_html_tags(raw_text)
    processed_text = preprocess_text(cleaned_text)

    if not processed_text:
        raise ValueError(
            "The review became empty after preprocessing."
        )

    sequence = text_tokenizer.texts_to_sequences(
        [processed_text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = sentiment_model.predict(
        padded_sequence,
        verbose=0,
    )[0]

    predicted_class_index = int(np.argmax(probabilities))

    predicted_sentiment = label_encoder.inverse_transform(
        [predicted_class_index]
    )[0]

    confidence = float(
        probabilities[predicted_class_index]
    )

    probability_dictionary = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(
            label_encoder.classes_
        )
    }

    return (
        predicted_sentiment,
        confidence,
        probability_dictionary,
        processed_text,
    )


# =========================================================
# 8. Summarization
# =========================================================

def summarize_review(raw_text):
    raw_text = str(raw_text).strip()

    if not raw_text:
        raise ValueError("Please enter a review.")

    word_count = len(raw_text.split())

    if word_count <= SUMMARY_WORD_THRESHOLD:
        return (
            "The text is too short to require summarization. "
            "/ النص قصير جداً ولا يحتاج إلى تلخيص."
        )

    inputs = summary_tokenizer(
        raw_text,
        max_length=1024,
        return_tensors="pt",
        truncation=True,
    )

    summary_ids = summarization_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30,
        no_repeat_ngram_size=2,
        num_beams=4,
        early_stopping=True,
    )

    summary_text = summary_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
    ).strip()

    if not summary_text:
        return raw_text

    if len(summary_text.split()) >= word_count:
        return raw_text

    return summary_text


# =========================================================
# 9. Combined pipeline
# Both models use the original review independently
# =========================================================

def analyze_and_summarize_review(raw_text):
    (
        sentiment,
        confidence,
        probabilities,
        processed_text,
    ) = predict_sentiment(raw_text)

    summary_text = summarize_review(raw_text)

    return {
        "summary": summary_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_text": processed_text,
    }


# =========================================================
# 10. Gradio helper functions
# =========================================================

def sentiment_emoji(sentiment):
    emoji_map = {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😢",
    }

    return emoji_map.get(sentiment.lower(), "🔎")


def web_interface_fn(review_text):
    if not review_text or not review_text.strip():
        return (
            "Please enter a review.",
            "",
            {},
            "",
            "",
            "",
            "",
        )

    try:
        tokens_raw = word_tokenize(review_text.lower())

        normalized_text = normalize_negations(
            review_text.lower()
        )

        text_no_punct = normalized_text.translate(
            str.maketrans("", "", string.punctuation)
        )

        text_no_punct = re.sub(
            r"\d+",
            " ",
            text_no_punct,
        )

        tokens_no_punct = word_tokenize(
            text_no_punct
        )

        no_stop_words = [
            word
            for word in tokens_no_punct
            if word.isalpha() and word not in stop_words
        ]

        lemmatized = [
            lemmatizer.lemmatize(word)
            for word in no_stop_words
        ]

        results = analyze_and_summarize_review(
            review_text
        )

        sentiment_text = (
            f"{results['sentiment'].upper()} "
            f"{sentiment_emoji(results['sentiment'])}\n"
            f"Confidence: {results['confidence'] * 100:.2f}%"
        )

        probability_labels = {
            class_name.capitalize(): probability
            for class_name, probability in results[
                "probabilities"
            ].items()
        }

        return (
            results["summary"],
            sentiment_text,
            probability_labels,
            " | ".join(tokens_raw),
            " | ".join(tokens_no_punct),
            " | ".join(no_stop_words),
            " | ".join(lemmatized),
        )

    except Exception as error:
        return (
            f"Error: {error}",
            "",
            {},
            "",
            "",
            "",
            "",
        )


search_history = []


def search_in_dataset_with_history(query):
    global search_history

    if df is None:
        return (
            "Reviews.csv is not available in this Space.",
            "\n".join(search_history),
        )

    if not query or not query.strip():
        return (
            "Please enter a search term.",
            "\n".join(search_history),
        )

    common_keywords = [
        "battery",
        "screen",
        "camera",
        "price",
        "performance",
        "design",
        "software",
        "quality",
        "money",
        "coffee",
        "food",
        "tea",
        "taste",
    ]

    match = process.extractOne(
        query.lower(),
        common_keywords,
    )

    actual_query = (
        match[0]
        if match and match[1] > 80
        else query.lower().strip()
    )

    if query not in search_history:
        search_history.insert(0, query)
        search_history = search_history[:5]

    results = df[
        df["Text"].str.contains(
            actual_query,
            case=False,
            na=False,
            regex=False,
        )
    ]

    output = (
        f"🔍 Found {len(results)} reviews "
        f"for '{actual_query}'.\n\n"
    )

    if results.empty:
        output += "No matching reviews were found."

    for _, row in results.head(5).iterrows():
        review_preview = str(row["Text"])[:180]

        output += (
            f"📝 {review_preview}...\n"
            f"📊 {str(row['Sentiment']).upper()}\n\n"
        )

    return output, "\n".join(search_history)


def generate_dashboard(query):
    if df is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5,
            0.5,
            "Reviews.csv is not available.",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.axis("off")
        return fig

    if query and query.strip():
        filtered_df = df[
            df["Text"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

        title = (
            f"Sentiment Distribution for: "
            f"{query.strip().capitalize()}"
        )
    else:
        filtered_df = df
        title = "General Sentiment Distribution"

    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5,
            0.5,
            "No data found for this feature.",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.axis("off")
        return fig

    sentiment_counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reindex(
            ["positive", "neutral", "negative"],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    sentiment_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=140,
        ax=ax,
        textprops={
            "fontsize": 12,
            "weight": "bold",
        },
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_ylabel("")
    plt.tight_layout()

    return fig


# =========================================================
# 11. Gradio interface
# =========================================================

with gr.Blocks(
    title="LSTM Sentiment Analysis",
    theme=gr.themes.Soft(),
) as app:

    gr.Markdown(
        """
        <div style="
            text-align: center;
            background-color: #1e1e2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        ">
            <h1 style="color: #ff79c6;">
                💻 LSTM Sentiment Analysis & Summarization
            </h1>

            <p style="color: white;">
                Sentiment is predicted from the original review using a trained LSTM model.
                The original review is summarized independently using DistilBART.
            </p>

            <div style="
                display: flex;
                justify-content: space-around;
                color: white;
                font-size: 16px;
            ">
                <div>
                    👤 SAIF KHATTAB
                    <br>
                    🆔 202312556
                </div>

                <div>
                    👤 YAZEED AMEEN
                    <br>
                    🆔 202310058
                </div>
            </div>
        </div>
        """
    )

    with gr.Tabs():

        with gr.Tab("Analyze Text 📝"):
            input_text = gr.Textbox(
                lines=6,
                label="Enter Review",
                placeholder="Write an English product or food review...",
            )

            analyze_button = gr.Button(
                "Analyze 🚀",
                variant="primary",
            )

            with gr.Row():
                summary_output = gr.Textbox(
                    label="Generated Summary",
                    lines=5,
                )

                sentiment_output = gr.Textbox(
                    label="Predicted Sentiment",
                    lines=5,
                )

            probability_output = gr.Label(
                label="Class Probabilities",
                num_top_classes=3,
            )

            with gr.Accordion(
                "Text Processing Stages (Transparency Mode)",
                open=False,
            ):
                raw_tokens_output = gr.Textbox(
                    label="1. Raw Tokens"
                )

                no_punctuation_output = gr.Textbox(
                    label="2. No Punctuation"
                )

                no_stop_words_output = gr.Textbox(
                    label="3. No Stop Words"
                )

                lemmatized_output = gr.Textbox(
                    label="4. Lemmatized"
                )

            analyze_button.click(
                fn=web_interface_fn,
                inputs=input_text,
                outputs=[
                    summary_output,
                    sentiment_output,
                    probability_output,
                    raw_tokens_output,
                    no_punctuation_output,
                    no_stop_words_output,
                    lemmatized_output,
                ],
            )

        with gr.Tab("Search Dataset 🔍"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Search Query"
                    )

                    search_button = gr.Button(
                        "Search Data 🔎"
                    )

                history_output = gr.Textbox(
                    label="Last 5 Searches",
                    interactive=False,
                )

            results_output = gr.Textbox(
                label="Results",
                lines=10,
            )

            search_button.click(
                fn=search_in_dataset_with_history,
                inputs=query_input,
                outputs=[
                    results_output,
                    history_output,
                ],
            )

        with gr.Tab("Analytics Dashboard 📊"):
            dashboard_query = gr.Textbox(
                label="Filter statistics by feature",
                placeholder="Examples: coffee, food, tea, quality",
            )

            dashboard_button = gr.Button(
                "Show Chart 🚀"
            )

            plot_output = gr.Plot(
                label="Sentiment Distribution"
            )

            dashboard_button.click(
                fn=generate_dashboard,
                inputs=dashboard_query,
                outputs=plot_output,
            )


if __name__ == "__main__":
    app.launch()