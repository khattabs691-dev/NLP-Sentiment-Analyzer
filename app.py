import os

# Must be set before importing TensorFlow or PyTorch.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import pickle
import re
import string
import warnings
import zipfile

import gradio as gr
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from thefuzz import process

warnings.filterwarnings("ignore")

# Limit TensorFlow CPU threads to reduce crashes and memory pressure.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================================================
# Files and settings
# =========================================================

LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
REVIEWS_FILE_PATH = "Reviews.csv"

SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
SUMMARY_WORD_THRESHOLD = 15
DEFAULT_MAX_LENGTH = 111

# DistilBART is loaded only when summarization is first requested.
summary_tokenizer = None
summarization_model = None


# =========================================================
# NLTK setup
# =========================================================

for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(package, quiet=True)

stop_words = set(stopwords.words("english"))
stop_words = stop_words.difference({"not", "no", "nor", "never"})
lemmatizer = WordNetLemmatizer()


# =========================================================
# Text preprocessing
# Must match training preprocessing
# =========================================================

def clean_html_tags(text):
    text = re.sub(r"<[^>]*>", " ", str(text))
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
    text = normalize_negations(str(text).lower())
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned_tokens)


# =========================================================
# Load trained LSTM artifacts
# =========================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}. "
            "Upload it to the root of the Hugging Face Space."
        )


for file_path in [
    LSTM_MODEL_PATH,
    TEXT_TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
]:
    require_file(file_path)

def remove_unsupported_keras_config(value):
    """
    Remove configuration fields written by a newer Keras version but not
    recognized by the Keras version bundled with TensorFlow in this Space.
    """
    removed_count = 0

    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            if key == "quantization_config" and item is None:
                removed_count += 1
                continue

            cleaned_item, item_removed = remove_unsupported_keras_config(item)
            cleaned[key] = cleaned_item
            removed_count += item_removed

        return cleaned, removed_count

    if isinstance(value, list):
        cleaned_list = []

        for item in value:
            cleaned_item, item_removed = remove_unsupported_keras_config(item)
            cleaned_list.append(cleaned_item)
            removed_count += item_removed

        return cleaned_list, removed_count

    return value, removed_count


def prepare_compatible_keras_model(model_path):
    """
    A .keras model is a ZIP archive. This function copies the archive to /tmp
    and removes only unsupported null quantization_config fields from
    config.json. The original uploaded model is never modified.
    """
    if not zipfile.is_zipfile(model_path):
        return model_path

    patched_path = "/tmp/lstm_sentiment_model_compatible.keras"

    with zipfile.ZipFile(model_path, "r") as source_archive:
        if "config.json" not in source_archive.namelist():
            return model_path

        model_config = json.loads(
            source_archive.read("config.json").decode("utf-8")
        )

        cleaned_config, removed_count = remove_unsupported_keras_config(
            model_config
        )

        if removed_count == 0:
            return model_path

        with zipfile.ZipFile(
            patched_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target_archive:
            for archive_item in source_archive.infolist():
                file_data = source_archive.read(archive_item.filename)

                if archive_item.filename == "config.json":
                    file_data = json.dumps(
                        cleaned_config,
                        ensure_ascii=False,
                    ).encode("utf-8")

                target_archive.writestr(archive_item, file_data)

    print(
        "Prepared a Keras-compatible model copy. "
        f"Removed {removed_count} unsupported config fields."
    )

    return patched_path


print("Loading trained LSTM model...")

compatible_model_path = prepare_compatible_keras_model(
    LSTM_MODEL_PATH
)

sentiment_model = load_model(
    compatible_model_path,
    compile=False,
)

with open(TEXT_TOKENIZER_PATH, "rb") as file:
    text_tokenizer = pickle.load(file)

with open(LABEL_ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

try:
    MAX_LENGTH = int(sentiment_model.input_shape[1])
except (TypeError, ValueError, IndexError):
    MAX_LENGTH = DEFAULT_MAX_LENGTH

print(f"LSTM model loaded. MAX_LENGTH={MAX_LENGTH}")


# =========================================================
# Lazy DistilBART loading
# =========================================================

def get_summarization_model():
    global summary_tokenizer
    global summarization_model

    if summary_tokenizer is None or summarization_model is None:
        print("Loading DistilBART on first summarization request...")

        # Imported lazily so PyTorch/Transformers do not load at startup.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        summary_tokenizer = AutoTokenizer.from_pretrained(
            SUMMARIZATION_MODEL_NAME
        )

        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL_NAME,
            low_cpu_mem_usage=True,
        )

        summarization_model.eval()
        print("DistilBART loaded.")

    return summary_tokenizer, summarization_model


# =========================================================
# Optional Reviews.csv for search/dashboard
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
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter=",", quotechar='"', strict=True)

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
        print("Reviews.csv not found. Search and dashboard remain unavailable.")
        return None

    try:
        dataset, bad_rows = read_csv_safely(REVIEWS_FILE_PATH)

        required_columns = {"Text", "Summary", "Score"}
        if not required_columns.issubset(dataset.columns):
            print("Reviews.csv is missing required columns.")
            return None

        dataset = dataset[["Text", "Summary", "Score"]].copy()
        dataset["Score"] = pd.to_numeric(dataset["Score"], errors="coerce")
        dataset = dataset.dropna(subset=["Text", "Summary", "Score"])
        dataset = dataset[dataset["Score"].isin([1, 2, 3, 4, 5])].copy()
        dataset["Score"] = dataset["Score"].astype(int)
        dataset["Sentiment"] = dataset["Score"].apply(map_score_to_sentiment)
        dataset["Cleaned_Text"] = dataset["Text"].apply(clean_html_tags)
        dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(preprocess_text)
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
# Inference
# =========================================================

def predict_sentiment(raw_text):
    raw_text = str(raw_text).strip()

    if not raw_text:
        raise ValueError("Please enter a review.")

    processed_text = preprocess_text(clean_html_tags(raw_text))

    if not processed_text:
        raise ValueError("The text became empty after preprocessing.")

    sequence = text_tokenizer.texts_to_sequences([processed_text])

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

    predicted_index = int(np.argmax(probabilities))
    sentiment = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    all_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(label_encoder.classes_)
    }

    return sentiment, confidence, all_probabilities


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

    tokenizer, model = get_summarization_model()

    inputs = tokenizer(
        raw_text,
        max_length=1024,
        return_tensors="pt",
        truncation=True,
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30,
        no_repeat_ngram_size=2,
        num_beams=2,
        early_stopping=True,
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
    ).strip()

    if not summary or len(summary.split()) >= word_count:
        return raw_text

    return summary


def analyze_review(raw_text):
    sentiment, confidence, probabilities = predict_sentiment(raw_text)
    summary = summarize_review(raw_text)

    return {
        "summary": summary,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# =========================================================
# Gradio functions
# =========================================================

def sentiment_emoji(sentiment):
    return {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😢",
    }.get(sentiment.lower(), "🔎")


def web_interface_fn(review_text):
    if not review_text or not review_text.strip():
        return "Please enter a review.", "", {}, "", "", "", ""

    try:
        raw_tokens = word_tokenize(review_text.lower())

        normalized = normalize_negations(review_text.lower())
        no_punctuation_text = normalized.translate(
            str.maketrans("", "", string.punctuation)
        )
        no_punctuation_text = re.sub(r"\d+", " ", no_punctuation_text)
        no_punctuation_tokens = word_tokenize(no_punctuation_text)

        without_stop_words = [
            word
            for word in no_punctuation_tokens
            if word.isalpha() and word not in stop_words
        ]

        lemmatized = [
            lemmatizer.lemmatize(word)
            for word in without_stop_words
        ]

        result = analyze_review(review_text)

        sentiment_text = (
            f"{result['sentiment'].upper()} "
            f"{sentiment_emoji(result['sentiment'])}\n"
            f"Confidence: {result['confidence'] * 100:.2f}%"
        )

        labels = {
            name.capitalize(): probability
            for name, probability in result["probabilities"].items()
        }

        return (
            result["summary"],
            sentiment_text,
            labels,
            " | ".join(raw_tokens),
            " | ".join(no_punctuation_tokens),
            " | ".join(without_stop_words),
            " | ".join(lemmatized),
        )

    except Exception as error:
        return f"Error: {error}", "", {}, "", "", "", ""


search_history = []


def search_in_dataset_with_history(query):
    global search_history

    if df is None:
        return (
            "Reviews.csv is not available in this Space.",
            "\n".join(search_history),
        )

    if not query or not query.strip():
        return "Please enter a search term.", "\n".join(search_history)

    common_keywords = [
        "battery", "screen", "camera", "price", "performance",
        "design", "software", "quality", "money", "coffee",
        "food", "tea", "taste",
    ]

    match = process.extractOne(query.lower(), common_keywords)
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

    output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"

    if results.empty:
        output += "No matching reviews were found."

    for _, row in results.head(5).iterrows():
        output += (
            f"📝 {str(row['Text'])[:180]}...\n"
            f"📊 {str(row['Sentiment']).upper()}\n\n"
        )

    return output, "\n".join(search_history)


def _empty_dashboard_figure(message):
    """Create a dashboard-styled empty state instead of a white Matplotlib box."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    ax.text(
        0.5,
        0.56,
        "◌",
        ha="center",
        va="center",
        fontsize=36,
        color="#79b8ff",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.40,
        message,
        ha="center",
        va="center",
        fontsize=13,
        color="#d0d7de",
        fontweight="bold",
    )

    ax.axis("off")
    fig.tight_layout(pad=1.5)
    return fig


def generate_dashboard(query):
    if df is None:
        return _empty_dashboard_figure(
            "Reviews.csv is not available."
        )

    clean_query = str(query or "").strip()

    if clean_query:
        filtered_df = df[
            df["Text"].str.contains(
                clean_query,
                case=False,
                na=False,
                regex=False,
            )
        ]
        title = f"Sentiment Intelligence · {clean_query.title()}"
    else:
        filtered_df = df
        title = "General Sentiment Intelligence"

    if filtered_df.empty:
        return _empty_dashboard_figure(
            f"No reviews found for “{clean_query}”."
        )

    counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )

    visible_counts = counts[counts > 0]
    color_map = {
        "positive": "#79b8ff",
        "neutral": "#c9d1d9",
        "negative": "#1f6feb",
    }
    chart_colors = [color_map[label] for label in visible_counts.index]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    wedges, label_texts, percentage_texts = ax.pie(
        visible_counts.values,
        labels=[label.title() for label in visible_counts.index],
        colors=chart_colors,
        startangle=110,
        counterclock=False,
        autopct=lambda percentage: (
            f"{percentage:.1f}%" if percentage >= 0.5 else ""
        ),
        pctdistance=0.78,
        labeldistance=1.10,
        wedgeprops={
            "width": 0.34,
            "edgecolor": "#161b22",
            "linewidth": 3,
        },
        textprops={
            "fontsize": 11,
            "fontweight": "bold",
            "color": "#e6edf3",
        },
    )

    for percentage_text in percentage_texts:
        percentage_text.set_color("#ffffff")
        percentage_text.set_fontsize(11)
        percentage_text.set_fontweight("bold")

    total_reviews = int(visible_counts.sum())
    ax.text(
        0,
        0.08,
        f"{total_reviews:,}",
        ha="center",
        va="center",
        fontsize=25,
        color="#ffffff",
        fontweight="bold",
    )
    ax.text(
        0,
        -0.10,
        "MATCHING REVIEWS",
        ha="center",
        va="center",
        fontsize=9,
        color="#8b949e",
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        color="#f0f3f6",
        pad=20,
    )

    legend_labels = [
        f"{label.title()}  ·  {int(value):,}"
        for label, value in visible_counts.items()
    ]
    legend = ax.legend(
        wedges,
        legend_labels,
        title="Sentiment Breakdown",
        loc="center left",
        bbox_to_anchor=(0.96, 0.5),
        frameon=True,
        borderpad=1.0,
        labelspacing=1.0,
        fontsize=10,
    )

    legend.get_frame().set_facecolor("#1c2128")
    legend.get_frame().set_edgecolor("#30363d")
    legend.get_frame().set_linewidth(1.0)
    legend.get_title().set_color("#ffffff")
    legend.get_title().set_fontweight("bold")

    for legend_text in legend.get_texts():
        legend_text.set_color("#c9d1d9")

    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.05, right=0.80, top=0.86, bottom=0.08)

    return fig


# =========================================================
# Gradio interface
# =========================================================

CUSTOM_CSS = """
:root {
    --bg-0: #0d1117;
    --bg-1: #11161d;
    --bg-2: #1c2128;
    --panel: rgba(22, 27, 34, 0.76);
    --panel-strong: rgba(28, 33, 40, 0.88);
    --panel-soft: rgba(255, 255, 255, 0.055);
    --text-0: #f6f8fa;
    --text-1: #c9d1d9;
    --text-2: #8b949e;
    --blue: #388bfd;
    --violet: #2f81f7;
    --cyan: #79b8ff;
    --teal: #d0d7de;
    --pink: #1f6feb;
    --amber: #c9d1d9;
    --line: rgba(139, 148, 158, 0.18);
    --line-strong: rgba(121, 184, 255, 0.34);
    --shadow-lg: 0 32px 100px rgba(0, 0, 0, 0.42);
    --shadow-md: 0 18px 52px rgba(0, 0, 0, 0.28);
}

* { box-sizing: border-box; }

body {
    background: var(--bg-0);
}

.gradio-container {
    position: relative;
    min-height: 100vh;
    overflow-x: hidden;
    color: var(--text-0) !important;
    background:
        radial-gradient(circle at 10% 8%, rgba(121, 184, 255, 0.17), transparent 24%),
        radial-gradient(circle at 88% 12%, rgba(47, 129, 247, 0.19), transparent 27%),
        radial-gradient(circle at 75% 78%, rgba(88, 166, 255, 0.10), transparent 24%),
        radial-gradient(circle at 24% 88%, rgba(208, 215, 222, 0.08), transparent 22%),
        linear-gradient(150deg, var(--bg-0) 0%, var(--bg-1) 42%, var(--bg-2) 100%) !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(139, 148, 158, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(139, 148, 158, 0.055) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(circle at center, black 0%, rgba(0, 0, 0,.92) 62%, transparent 100%);
}

.gradio-container::after {
    content: "";
    position: fixed;
    inset: -18%;
    z-index: 0;
    pointer-events: none;
    background:
        conic-gradient(
            from 0deg at 50% 50%,
            transparent 0deg,
            rgba(121, 184, 255, 0.045) 60deg,
            transparent 120deg,
            rgba(47, 129, 247, 0.05) 200deg,
            transparent 265deg,
            rgba(88, 166, 255, 0.04) 330deg,
            transparent 360deg
        );
    animation: ambientSpin 42s linear infinite;
}

.app-shell {
    position: relative;
    z-index: 3;
    max-width: 1240px;
    margin: 0 auto;
    padding: 10px 8px 34px;
}

.tech-background {
    position: fixed;
    inset: 0;
    z-index: 1;
    overflow: hidden;
    pointer-events: none;
}

.aurora {
    position: absolute;
    border-radius: 999px;
    filter: blur(34px);
    opacity: 0.34;
    mix-blend-mode: screen;
    animation: auroraDrift 18s ease-in-out infinite;
}

.aurora.a1 {
    width: 440px;
    height: 160px;
    top: 8%;
    left: -120px;
    background: linear-gradient(90deg, transparent, rgba(121, 184, 255,.72), transparent);
    transform: rotate(-16deg);
}

.aurora.a2 {
    width: 520px;
    height: 180px;
    top: 34%;
    right: -160px;
    background: linear-gradient(90deg, transparent, rgba(47, 129, 247,.70), transparent);
    animation-delay: -6s;
    transform: rotate(12deg);
}

.aurora.a3 {
    width: 400px;
    height: 150px;
    bottom: 4%;
    left: 30%;
    background: linear-gradient(90deg, transparent, rgba(88, 166, 255,.55), transparent);
    animation-delay: -11s;
    transform: rotate(-7deg);
}

.data-dot {
    position: absolute;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a5d6ff;
    box-shadow: 0 0 16px rgba(121, 184, 255,.95);
    animation: dataFloat 13s ease-in-out infinite;
}

.data-dot.d1 { top: 12%; left: 15%; }
.data-dot.d2 { top: 22%; right: 18%; background: #79b8ff; animation-delay: -2s; }
.data-dot.d3 { top: 48%; left: 8%; background: #d0d7de; animation-delay: -4s; }
.data-dot.d4 { top: 66%; right: 11%; background: #58a6ff; animation-delay: -7s; }
.data-dot.d5 { bottom: 8%; left: 58%; background: #ffffff; animation-delay: -9s; }
.data-dot.d6 { top: 38%; left: 54%; animation-delay: -5s; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin: 4px 0 14px;
    padding: 10px 14px;
    border: 1px solid rgba(139, 148, 158, 0.15);
    border-radius: 18px;
    background: rgba(13, 17, 23, 0.58);
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 28px rgba(0, 0, 0,.20);
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-weight: 850;
    letter-spacing: -0.01em;
}

.brand-mark {
    position: relative;
    width: 34px;
    height: 34px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--cyan), var(--blue), var(--violet));
    box-shadow: 0 8px 22px rgba(31, 111, 235,.35);
}

.brand-mark::before,
.brand-mark::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: white;
    opacity: .88;
}

.brand-mark::before {
    width: 7px;
    height: 7px;
    left: 8px;
    top: 8px;
}

.brand-mark::after {
    width: 5px;
    height: 5px;
    right: 7px;
    bottom: 7px;
}

.status-cluster {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    border: 1px solid rgba(139, 148, 158, 0.18);
    border-radius: 999px;
    color: var(--text-1);
    background: rgba(255, 255, 255,.045);
    font-size: 12px;
    font-weight: 750;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 12px rgba(208, 215, 222,.85);
    animation: statusPulse 2.1s ease-in-out infinite;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 390px;
    border: 1px solid rgba(255, 255, 255,.12);
    border-radius: 34px;
    padding: 54px 48px 40px;
    margin: 8px 0 28px;
    color: white;
    background:
        radial-gradient(circle at 76% 22%, rgba(255, 255, 255,.15), transparent 22%),
        linear-gradient(135deg, rgba(34, 39, 46,.95), rgba(22, 27, 34,.94) 55%, rgba(27, 42, 65,.92));
    backdrop-filter: blur(18px);
    box-shadow: 0 38px 110px rgba(0, 0, 0,.55);
}

.hero::before {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    top: -150px;
    right: -80px;
    border-radius: 48% 52% 61% 39% / 45% 42% 58% 55%;
    border: 1px solid rgba(255, 255, 255,.2);
    background: rgba(255, 255, 255,.06);
    animation: morph 15s ease-in-out infinite;
}

.hero::after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    left: -80px;
    bottom: -140px;
    border-radius: 55% 45% 47% 53% / 46% 58% 42% 54%;
    background: rgba(255, 255, 255,.08);
    animation: morph 19s ease-in-out infinite reverse;
}

.hero-content {
    position: relative;
    z-index: 4;
    max-width: 770px;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding: 8px 11px;
    border: 1px solid rgba(255, 255, 255,.20);
    border-radius: 999px;
    color: #f0f3f6;
    background: rgba(255, 255, 255,.09);
    backdrop-filter: blur(10px);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.hero-kicker i {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 14px rgba(121, 184, 255,.9);
}

.hero h1 {
    margin: 0 0 16px;
    color: #fff !important;
    font-size: clamp(40px, 5.2vw, 64px);
    line-height: 1;
    letter-spacing: -0.055em;
    text-shadow: 0 5px 26px rgba(0, 0, 0,.38);
}

.hero p {
    max-width: 760px;
    margin: 0;
    color: #e6edf3 !important;
    font-size: 16px;
    line-height: 1.82;
    font-weight: 520;
}

.hero-metrics {
    position: relative;
    z-index: 4;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.metric {
    padding: 10px 12px;
    border: 1px solid rgba(255, 255, 255,.18);
    border-radius: 14px;
    background: rgba(255, 255, 255,.09);
    backdrop-filter: blur(10px);
}

.metric strong {
    display: block;
    color: #fff;
    font-size: 15px;
}

.metric span {
    color: #d0d7de;
    font-size: 11px;
    letter-spacing: .04em;
}

.hero-visuals {
    position: absolute;
    inset: 0;
    z-index: 3;
    overflow: hidden;
    pointer-events: none;
}

.neural-canvas {
    position: absolute;
    right: 20px;
    top: 38px;
    width: 340px;
    height: 230px;
    opacity: .78;
}

.neural-line {
    stroke: rgba(201, 209, 217,.34);
    stroke-width: 1.3;
    stroke-dasharray: 5 7;
    animation: dashFlow 9s linear infinite;
}

.neural-node {
    fill: rgba(255, 255, 255,.14);
    stroke: rgba(255, 255, 255,.52);
    stroke-width: 1.2;
    filter: drop-shadow(0 0 8px rgba(121, 184, 255,.35));
    animation: nodeGlow 3.8s ease-in-out infinite;
}

.neural-node.n2 { animation-delay: -1s; }
.neural-node.n3 { animation-delay: -2s; }
.neural-node.n4 { animation-delay: -2.8s; }

.floating-chip {
    position: absolute;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 9px 11px;
    border: 1px solid rgba(255, 255, 255,.22);
    border-radius: 13px;
    color: #fff;
    background: rgba(255, 255, 255,.10);
    backdrop-filter: blur(11px);
    box-shadow: 0 10px 28px rgba(0, 0, 0,.22);
    font-size: 12px;
    font-weight: 800;
    animation: chipFloat 7s ease-in-out infinite;
}

.floating-chip.c1 { right: 9%; bottom: 26%; }
.floating-chip.c2 { right: 28%; top: 12%; animation-delay: -2s; }
.floating-chip.c3 { right: 4%; top: 40%; animation-delay: -4s; }

.team-grid {
    position: relative;
    z-index: 4;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 26px;
}

.team-card {
    position: relative;
    overflow: hidden;
    min-width: 0;
    padding: 16px 12px;
    font-size: 17px;
    letter-spacing: .03em;
    white-space: nowrap;
    border: 1px solid rgba(255, 255, 255,.20);
    border-radius: 20px;
    color: white !important;
    background: linear-gradient(145deg, rgba(255, 255, 255,.17), rgba(255, 255, 255,.075));
    backdrop-filter: blur(14px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255,.16);
    font-weight: 850;
    text-align: center;
    transition: transform .25s ease, border-color .25s ease, background .25s ease;
}

.team-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255, 255, 255,.34);
    background: linear-gradient(145deg, rgba(255, 255, 255,.24), rgba(255, 255, 255,.11));
}

.team-card span {
    display: block;
    margin-bottom: 6px;
    color: #d0d7de !important;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.feature-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 26px;
}

.feature-card {
    position: relative;
    overflow: hidden;
    min-height: 136px;
    padding: 23px 21px 20px;
    border: 1px solid var(--line);
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(28, 33, 40,.88), rgba(13, 17, 23,.78));
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow-md);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

.feature-card::before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: .42;
    background:
        linear-gradient(120deg, transparent 0%, rgba(255, 255, 255,.03) 40%, transparent 62%);
    transform: translateX(-120%);
    transition: transform .65s ease;
}

.feature-card::after {
    content: "";
    position: absolute;
    width: 130px;
    height: 130px;
    right: -55px;
    bottom: -60px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(121, 184, 255,.14), transparent 70%);
}

.feature-card:hover {
    transform: translateY(-6px);
    border-color: rgba(121, 184, 255,.35);
    box-shadow: 0 22px 58px rgba(0, 0, 0,.36);
}

.feature-card:hover::before {
    transform: translateX(120%);
}

.feature-icon {
    display: inline-grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin-bottom: 12px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(56, 139, 253,.30), rgba(121, 184, 255,.20));
    border: 1px solid rgba(139, 148, 158,.18);
    font-size: 18px;
}

.feature-card strong {
    position: relative;
    z-index: 1;
    display: block;
    margin-bottom: 7px;
    color: #f0f3f6 !important;
    font-size: 15px;
}

.feature-card small {
    position: relative;
    z-index: 1;
    color: var(--text-1) !important;
    font-size: 13px;
    line-height: 1.65;
}

.app-tabs {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line) !important;
    border-radius: 28px !important;
    background: rgba(13, 17, 23,.76) !important;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-lg) !important;
    padding: 12px !important;
}

.app-tabs::before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -100px;
    top: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(47, 129, 247,.12), transparent 70%);
    pointer-events: none;
}

.app-tabs button[role="tab"] {
    min-height: 50px;
    border-radius: 15px !important;
    color: #b1bac4 !important;
    font-weight: 780 !important;
    transition: all .22s ease !important;
}

.app-tabs button[role="tab"]:hover {
    color: #fff !important;
    background: rgba(255, 255, 255,.055) !important;
}

.app-tabs button[role="tab"][aria-selected="true"] {
    color: #fff !important;
    background: linear-gradient(135deg, var(--blue), var(--violet) 56%, var(--cyan)) !important;
    box-shadow: 0 12px 30px rgba(31, 111, 235,.38) !important;
}

.section-card {
    border: 1px solid rgba(139, 148, 158,.15) !important;
    border-radius: 24px !important;
    background: linear-gradient(180deg, rgba(28, 33, 40,.90), rgba(13, 17, 23,.84)) !important;
    backdrop-filter: blur(18px);
    box-shadow: 0 16px 42px rgba(0, 0, 0,.24) !important;
    padding: 18px !important;
}

.section-card h1,
.section-card h2,
.section-card h3 {
    color: #f0f3f6 !important;
}

.section-card p,
.section-card label,
.section-card span {
    color: #c9d1d9 !important;
}

.section-card textarea,
.section-card input {
    min-height: 49px;
    border: 1px solid rgba(139, 148, 158,.22) !important;
    border-radius: 17px !important;
    color: #f6f8fa !important;
    background: rgba(255, 255, 255,.055) !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0,.14) !important;
}

.section-card textarea:focus,
.section-card input:focus {
    border-color: rgba(121, 184, 255,.68) !important;
    box-shadow: 0 0 0 4px rgba(121, 184, 255,.10), 0 12px 28px rgba(0, 0, 0,.16) !important;
}

.section-card textarea::placeholder,
.section-card input::placeholder {
    color: #8b949e !important;
    opacity: 1 !important;
}

.primary-action,
.primary-action button {
    min-height: 50px !important;
    border: none !important;
    border-radius: 17px !important;
    color: #fff !important;
    background: linear-gradient(135deg, var(--blue) 0%, var(--violet) 55%, var(--cyan) 100%) !important;
    box-shadow: 0 15px 34px rgba(31, 111, 235,.38) !important;
    font-weight: 850 !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
}

.primary-action:hover {
    transform: translateY(-2px) scale(1.005);
    box-shadow: 0 20px 46px rgba(31, 111, 235,.46) !important;
}

.secondary-action,
.secondary-action button {
    min-height: 50px !important;
    border: 1px solid rgba(139, 148, 158,.25) !important;
    border-radius: 17px !important;
    color: #d0d7de !important;
    background: linear-gradient(180deg, rgba(255, 255, 255,.085), rgba(255, 255, 255,.045)) !important;
    box-shadow: 0 12px 28px rgba(0, 0, 0,.20) !important;
    font-weight: 820 !important;
}

.gradio-container .block,
.gradio-container .label-wrap,
.gradio-container .wrap,
.gradio-container .panel,
.gradio-container .form {
    color: var(--text-0) !important;
    background: transparent !important;
}

footer { display: none !important; }

.footer-note {
    margin: 22px 0 6px;
    padding: 15px;
    color: var(--text-2) !important;
    text-align: center;
    font-size: 13px;
}

@keyframes ambientSpin {
    from { transform: rotate(0deg) scale(1); }
    to { transform: rotate(360deg) scale(1.05); }
}

@keyframes auroraDrift {
    0%, 100% { transform: translate3d(0,0,0) rotate(-10deg); }
    50% { transform: translate3d(55px,-22px,0) rotate(7deg); }
}

@keyframes dataFloat {
    0%,100% { transform: translateY(0) scale(1); opacity:.55; }
    50% { transform: translateY(-30px) scale(1.4); opacity:1; }
}

@keyframes statusPulse {
    0%,100% { opacity:.45; transform: scale(.9); }
    50% { opacity:1; transform: scale(1.25); }
}

@keyframes morph {
    0%,100% { border-radius: 48% 52% 61% 39% / 45% 42% 58% 55%; transform: rotate(0deg); }
    50% { border-radius: 57% 43% 45% 55% / 52% 60% 40% 48%; transform: rotate(12deg); }
}

@keyframes dashFlow {
    to { stroke-dashoffset: -220; }
}

@keyframes nodeGlow {
    0%,100% { opacity:.55; transform: scale(1); }
    50% { opacity:1; transform: scale(1.08); }
}

@keyframes chipFloat {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}


.floating-chip.c4 {
    right: 13%;
    top: 67%;
    bottom: auto;
    animation-delay: -5.5s;
    border-color: rgba(208, 215, 222, 0.40);
    background: linear-gradient(
        135deg,
        rgba(208, 215, 222, 0.20),
        rgba(56, 139, 253, 0.15)
    );
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.24),
        0 0 20px rgba(208, 215, 222, 0.10);
}

.data-dot.d7 {
    top: 82%;
    left: 18%;
    background: #58a6ff;
    animation-delay: -10s;
}

.data-dot.d8 {
    top: 18%;
    left: 72%;
    background: #c9d1d9;
    animation-delay: -6.5s;
}

.tech-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(139, 148, 158, 0.18);
    box-shadow:
        inset 0 0 30px rgba(56, 139, 253, 0.05),
        0 0 24px rgba(121, 184, 255, 0.05);
    animation: ringOrbit 24s linear infinite;
}

.tech-ring::before,
.tech-ring::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: #79b8ff;
    box-shadow: 0 0 14px rgba(121, 184, 255, 0.85);
}

.tech-ring::before {
    width: 7px;
    height: 7px;
    top: 18%;
    left: 8%;
}

.tech-ring::after {
    width: 5px;
    height: 5px;
    right: 12%;
    bottom: 17%;
    background: #58a6ff;
    box-shadow: 0 0 14px rgba(47, 129, 247, 0.82);
}

.tech-ring.r1 {
    width: 230px;
    height: 230px;
    top: 18%;
    left: 4%;
}

.tech-ring.r2 {
    width: 310px;
    height: 310px;
    top: 48%;
    right: -90px;
    animation-direction: reverse;
    animation-duration: 31s;
}

.tech-ring.r3 {
    width: 170px;
    height: 170px;
    bottom: 4%;
    left: 47%;
    animation-duration: 19s;
}

.scan-beam {
    position: absolute;
    width: 2px;
    height: 44vh;
    opacity: 0.24;
    background: linear-gradient(
        to bottom,
        transparent,
        rgba(121, 184, 255, 0.9),
        transparent
    );
    filter: blur(0.5px);
    box-shadow: 0 0 18px rgba(121, 184, 255, 0.35);
    animation: scanMove 13s ease-in-out infinite;
}

.scan-beam.s1 {
    top: 6%;
    left: 26%;
    transform: rotate(18deg);
}

.scan-beam.s2 {
    right: 20%;
    bottom: 3%;
    height: 35vh;
    transform: rotate(-16deg);
    animation-delay: -6s;
    background: linear-gradient(
        to bottom,
        transparent,
        rgba(47, 129, 247, 0.85),
        transparent
    );
}


@keyframes ringOrbit {
    from { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.04); }
    to { transform: rotate(360deg) scale(1); }
}

@keyframes scanMove {
    0%, 100% {
        transform: translate3d(-18px, -8px, 0) rotate(18deg);
        opacity: 0.12;
    }
    50% {
        transform: translate3d(30px, 18px, 0) rotate(18deg);
        opacity: 0.34;
    }
}



/* ---------- refined background details ---------- */
.tech-ring {
    opacity: 0.55;
}

.tech-ring.r1 {
    width: 160px;
    height: 160px;
    top: 20%;
    left: 5%;
}

.tech-ring.r2 {
    width: 190px;
    height: 190px;
    top: 56%;
    right: -35px;
    animation-duration: 27s;
}

.tech-ring.r3 {
    width: 120px;
    height: 120px;
    bottom: 7%;
    left: 49%;
    animation-duration: 18s;
}

.bg-badge {
    position: absolute;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(139, 148, 158, 0.20);
    background: rgba(255, 255, 255, 0.055);
    backdrop-filter: blur(10px);
    color: rgba(201, 209, 217, 0.62);
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 0.05em;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.10);
    animation: badgeFloat 14s ease-in-out infinite;
}

.bg-ai {
    top: 16%;
    left: 8%;
}

.bg-nlp {
    top: 64%;
    left: 13%;
    animation-delay: -4s;
}

.bg-dl {
    top: 24%;
    right: 11%;
    animation-delay: -6s;
}

.bg-robot {
    right: 16%;
    bottom: 14%;
    font-size: 18px;
    padding: 9px 12px;
    animation-delay: -8s;
}

.bg-neural {
    position: absolute;
    overflow: visible;
    opacity: 0.17;
    filter: drop-shadow(0 0 14px rgba(121, 184, 255, 0.18));
    animation: neuralDrift 18s ease-in-out infinite;
}

.bg-neural line {
    stroke: rgba(201, 209, 217, 0.36);
    stroke-width: 1.2;
    stroke-dasharray: 4 5;
}

.bg-neural circle {
    fill: rgba(255, 255, 255, 0.12);
    stroke: rgba(201, 209, 217, 0.58);
    stroke-width: 1.1;
}

.bg-neural.net-one {
    top: 18%;
    left: 34%;
    width: 220px;
    height: 120px;
}

.bg-neural.net-two {
    bottom: 14%;
    right: 8%;
    width: 230px;
    height: 130px;
    animation-delay: -7s;
}

.floating-chip.c1 {
    right: 22%;
    bottom: 18%;
}

.floating-chip.c2 {
    right: 31%;
    top: 16%;
}

.floating-chip.c3 {
    right: 9%;
    top: 44%;
}

.floating-chip.c4 {
    right: 21%;
    top: 52%;
    bottom: auto;
    animation-delay: -5.5s;
    border-color: rgba(208, 215, 222, 0.40);
    background: linear-gradient(
        135deg,
        rgba(208, 215, 222, 0.20),
        rgba(56, 139, 253, 0.15)
    );
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.24),
        0 0 20px rgba(208, 215, 222, 0.10);
}

/* make the hero network a little clearer */
.neural-canvas {
    right: 28px;
    top: 38px;
    width: 330px;
    height: 220px;
    opacity: 0.74;
}

@keyframes badgeFloat {
    0%, 100% {
        transform: translateY(0);
        opacity: 0.46;
    }
    50% {
        transform: translateY(-12px);
        opacity: 0.72;
    }
}

@keyframes neuralDrift {
    0%, 100% {
        transform: translateY(0) translateX(0);
        opacity: 0.13;
    }
    50% {
        transform: translateY(-10px) translateX(7px);
        opacity: 0.22;
    }
}


/* Final cleanup: decorative elements always stay behind the interface */
.tech-background {
    z-index: 0 !important;
}

.app-shell {
    z-index: 3 !important;
}

.hero,
.feature-strip,
.app-tabs,
.section-card,
.topbar {
    position: relative;
    z-index: 4;
}

/* Prevent background graphics from crossing readable content */
.bg-neural {
    opacity: 0.08 !important;
}

.bg-neural.net-one {
    top: 10%;
    left: 3%;
    width: 170px;
    height: 95px;
}

.bg-neural.net-two {
    bottom: 8%;
    right: 3%;
    width: 170px;
    height: 95px;
}

.bg-badge.bg-ai {
    top: 9%;
    left: 3%;
    opacity: 0.28;
    font-size: 10px;
    padding: 6px 9px;
}

.scan-beam {
    opacity: 0.10 !important;
}

.tech-ring {
    opacity: 0.22 !important;
}

/* Keep the four hero chips tidy and non-overlapping */
.floating-chip.c1 {
    right: 25%;
    top: 14%;
    bottom: auto;
}

.floating-chip.c2 {
    right: 8%;
    top: 28%;
}

.floating-chip.c3 {
    right: 8%;
    top: 52%;
}

.floating-chip.c4 {
    right: 26%;
    top: 59%;
    bottom: auto;
}

/* Remove blue blocks behind field labels */
.section-card .label-wrap,
.section-card label,
.section-card .label-wrap span,
.section-card [data-testid="block-info"],
.section-card .block-info {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    color: #d0d7de !important;
}

/* Keep labels readable without glowing rectangles */
.section-card .label-wrap {
    padding-left: 2px !important;
    padding-right: 2px !important;
}

/* Ensure decorative layers never capture clicks */
.tech-background,
.hero-visuals,
.bg-neural,
.bg-badge,
.tech-ring,
.scan-beam,
.data-dot,
.aurora {
    pointer-events: none !important;
}

/* Tone down motion outside the hero */
.tech-background .data-dot {
    opacity: 0.45;
}

.tech-background .aurora {
    opacity: 0.22;
}



/* ---------- Final refined placement ---------- */

/* Hero chips clustered around the neural circle */
.floating-chip.c2 {
    left: auto;
    right: 31%;
    top: 18%;
    bottom: auto;
    animation-delay: -2s;
}
/* c2 = AI Confidence */

.floating-chip.c3 {
    left: auto;
    right: 9%;
    top: 46%;
    bottom: auto;
}
/* c3 = NLP Engine */

.floating-chip.c4 {
    left: auto;
    right: 30%;
    top: 58%;
    bottom: auto;
    animation-delay: -5.5s;
    border-color: rgba(208, 215, 222, 0.40);
    background: linear-gradient(
        135deg,
        rgba(208, 215, 222, 0.20),
        rgba(56, 139, 253, 0.15)
    );
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.24),
        0 0 20px rgba(208, 215, 222, 0.10);
}
/* c4 = Deep Learning */

.floating-chip.c1 {
    left: auto;
    right: 21%;
    top: 68%;
    bottom: auto;
}
/* c1 = Semantic Layer */

/* Background decorative elements: richer and more visible, but far from interaction content */
.bg-badge {
    opacity: 0.24 !important;
    font-size: 11px;
    padding: 7px 10px;
    z-index: 0 !important;
    letter-spacing: 0.06em;
}
.bg-badge.bg-ai  { top: 8%; left: 1%; }
.bg-badge.bg-ai2 { top: 72%; left: 2%; animation-delay: -3s; }
.bg-badge.bg-ai3 { top: 40%; right: 2%; animation-delay: -7s; }
.bg-badge.bg-nlp2 { top: 28%; left: 1.2%; animation-delay: -5s; }
.bg-badge.bg-nlp3 { bottom: 12%; right: 2%; animation-delay: -8s; }
.bg-badge.bg-dl2 { top: 12%; right: 1.5%; animation-delay: -4s; }
.bg-badge.bg-dl3 { bottom: 28%; left: 1.4%; animation-delay: -6s; }
.bg-badge.bg-robot2 { bottom: 18%; right: 1.6%; font-size: 15px; animation-delay: -9s; }
.bg-badge.bg-robot3 { top: 52%; right: 1.2%; font-size: 15px; animation-delay: -11s; }

.tech-ring {
    opacity: 0.18 !important;
    border-width: 1px;
}
.tech-ring.r1 { width: 100px; height: 100px; top: 14%; left: -22px; }
.tech-ring.r2 { width: 120px; height: 120px; top: 58%; right: -26px; }
.tech-ring.r3 { width: 88px; height: 88px; bottom: 5%; left: 48%; }
.tech-ring.r4 { width: 110px; height: 110px; bottom: 26%; left: -22px; animation-duration: 20s; }
.tech-ring.r5 { width: 96px; height: 96px; top: 30%; right: -18px; animation-duration: 17s; animation-direction: reverse; }
.tech-ring.r6 { width: 84px; height: 84px; top: 82%; right: 22%; animation-duration: 16s; }
.tech-ring.r7 { width: 92px; height: 92px; top: 6%; left: 42%; animation-duration: 19s; animation-direction: reverse; }

.scan-beam { opacity: 0.10 !important; }
.scan-beam.s3 {
    left: 52%;
    top: 2%;
    height: 30vh;
    transform: rotate(8deg);
    animation-delay: -4s;
    background: linear-gradient(to bottom, transparent, rgba(208, 215, 222,.72), transparent);
}
.scan-beam.s4 {
    right: 36%;
    bottom: 0%;
    height: 28vh;
    transform: rotate(-7deg);
    animation-delay: -9s;
    background: linear-gradient(to bottom, transparent, rgba(88, 166, 255,.60), transparent);
}

.tech-background .data-dot {
    opacity: 0.55;
    animation-duration: 8.5s;
}
.tech-background .aurora {
    opacity: 0.28;
    animation-duration: 13s;
}

/* More dots */
.data-dot.d9  { top: 18%; right: 33%; background: #a5d6ff; animation-delay: -3s; }
.data-dot.d10 { top: 84%; left: 22%; background: #79b8ff; animation-delay: -5s; }
.data-dot.d11 { top: 33%; left: 51%; background: #d0d7de; animation-delay: -7s; }
.data-dot.d12 { bottom: 22%; right: 28%; background: #58a6ff; animation-delay: -2s; }

@keyframes badgeFloat {
    0%, 100% { transform: translateY(0) translateX(0); opacity: 0.18; }
    50% { transform: translateY(-18px) translateX(8px); opacity: 0.28; }
}

@keyframes neuralDrift {
    0%, 100% { transform: translateY(0) translateX(0) scale(1); opacity: 0.08; }
    50% { transform: translateY(-16px) translateX(10px) scale(1.04); opacity: 0.15; }
}

/* More neural network icons at edges */
.bg-neural {
    opacity: 0.11 !important;
    z-index: 0 !important;
}
.bg-neural.net-one  { top: 10%; left: 12%; width: 150px; height: 85px; }
.bg-neural.net-two  { bottom: 8%; right: 9%; width: 155px; height: 88px; animation-delay: -7s; }
.bg-neural.net-three{ top: 45%; right: 2%; width: 145px; height: 82px; animation-delay: -5s; }
.bg-neural.net-four { bottom: 30%; left: 1%; width: 150px; height: 85px; animation-delay: -9s; }
.bg-neural.net-five { top: 24%; right: 11%; width: 145px; height: 82px; animation-delay: -4s; }

/* keep all decorative tech elements behind interaction content */
.tech-background { z-index: 0 !important; }
.app-shell { z-index: 3 !important; }
.hero, .feature-strip, .app-tabs, .section-card, .topbar { position: relative; z-index: 4; }



/* ===== Final hero network placement ===== */

/* AI Confidence — above-left of the neural network */
.floating-chip.c2 {
    left: auto !important;
    right: 26% !important;
    top: 14% !important;
    bottom: auto !important;
    animation-delay: -2s !important;
}

/* NLP Engine — right side of the neural network */
.floating-chip.c3 {
    left: auto !important;
    right: 6% !important;
    top: 45% !important;
    bottom: auto !important;
}

/* Deep Learning — lower-left of the neural network */
.floating-chip.c4 {
    left: auto !important;
    right: 31% !important;
    top: 63% !important;
    bottom: auto !important;
    animation-delay: -5.5s !important;
    border-color: rgba(208, 215, 222, 0.40) !important;
    background: linear-gradient(
        135deg,
        rgba(208, 215, 222, 0.20),
        rgba(56, 139, 253, 0.15)
    ) !important;
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.24),
        0 0 20px rgba(208, 215, 222, 0.10) !important;
}

/* Semantic Layer — directly below the neural network */
.floating-chip.c1 {
    left: auto !important;
    right: 18% !important;
    top: 74% !important;
    bottom: auto !important;
}

/* Smaller chips so they visually belong to the network */
.floating-chip {
    min-height: auto !important;
    padding: 8px 10px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
}

/* Refine the neural network position */
.neural-canvas {
    right: 10px !important;
    top: 90px !important;
    width: 280px !important;
    height: 190px !important;
    opacity: 0.78 !important;
}


/* ===== Robust neural-network cluster =====
   The network and all four labels now share one coordinate system.
   This prevents them from drifting apart when the screen size changes.
*/
.network-cluster {
    position: absolute !important;
    top: 42px !important;
    right: 26px !important;
    width: 370px !important;
    height: 255px !important;
    z-index: 5 !important;
    pointer-events: none !important;
}

.network-cluster .neural-canvas {
    position: absolute !important;
    top: 24px !important;
    right: 18px !important;
    width: 315px !important;
    height: 210px !important;
    opacity: 0.80 !important;
}

/* AI Confidence — attached to the upper-left of the network */
.network-cluster .floating-chip.c2 {
    left: 4px !important;
    right: auto !important;
    top: 18px !important;
    bottom: auto !important;
}

/* NLP Engine — attached to the right-middle of the network */
.network-cluster .floating-chip.c3 {
    left: auto !important;
    right: 0 !important;
    top: 112px !important;
    bottom: auto !important;
}

/* Deep Learning — attached to the lower-left of the network */
.network-cluster .floating-chip.c4 {
    left: 8px !important;
    right: auto !important;
    top: 174px !important;
    bottom: auto !important;
}

/* Semantic Layer — attached directly below the network */
.network-cluster .floating-chip.c1 {
    left: auto !important;
    right: 72px !important;
    top: 215px !important;
    bottom: auto !important;
}

.network-cluster .floating-chip {
    min-height: auto !important;
    padding: 8px 11px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
    white-space: nowrap !important;
}

/* Keep the text area clear from the visualization on wide screens */
@media (min-width: 821px) {
    .hero-content {
        max-width: 60% !important;
    }
}

/* Compact but still grouped on smaller screens */
@media (max-width: 820px) {
    .network-cluster {
        top: 76px !important;
        right: -24px !important;
        transform: scale(0.72) !important;
        transform-origin: top right !important;
        opacity: 0.48 !important;
    }

    .hero-content {
        max-width: 100% !important;
    }
}


/* ===== Restore styled backgrounds behind field titles =====
   This overrides the earlier transparent-label cleanup.
*/
.section-card .label-wrap,
.section-card [data-testid="block-info"],
.section-card .block-info {
    display: inline-flex !important;
    align-items: center !important;
    width: fit-content !important;
    max-width: calc(100% - 12px) !important;
    min-height: 34px !important;
    margin: 2px 0 9px 3px !important;
    padding: 7px 12px !important;
    border: 1px solid rgba(121, 184, 255, 0.34) !important;
    border-radius: 11px !important;
    color: #ffffff !important;
    background:
        linear-gradient(
            135deg,
            rgba(31, 111, 235, 0.90),
            rgba(47, 129, 247, 0.88) 58%,
            rgba(88, 166, 255, 0.82)
        ) !important;
    box-shadow:
        0 8px 20px rgba(31, 111, 235, 0.24),
        inset 0 1px 0 rgba(255, 255, 255, 0.16) !important;
    font-weight: 800 !important;
}

.section-card .label-wrap span,
.section-card [data-testid="block-info"] span,
.section-card .block-info span {
    color: #ffffff !important;
    background: transparent !important;
    font-weight: 800 !important;
}

/* Keep ordinary explanatory text unchanged */
.section-card p {
    background: transparent !important;
}


/* ===== Premium dashboard chart styling ===== */
.dashboard-plot {
    overflow: hidden !important;
    border: 1px solid rgba(139, 148, 158, 0.22) !important;
    border-radius: 22px !important;
    padding: 10px !important;
    background:
        radial-gradient(
            circle at 12% 8%,
            rgba(121, 184, 255, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 88% 92%,
            rgba(47, 129, 247, 0.10),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            rgba(22, 27, 34, 0.96),
            rgba(13, 17, 23, 0.94)
        ) !important;
    box-shadow:
        0 18px 44px rgba(0, 0, 0, 0.24),
        inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
}

.dashboard-plot img,
.dashboard-plot canvas,
.dashboard-plot .plot-container {
    border-radius: 16px !important;
    background: #161b22 !important;
}

.dashboard-plot .label-wrap,
.dashboard-plot [data-testid="block-info"],
.dashboard-plot .block-info {
    color: #ffffff !important;
    background:
        linear-gradient(
            135deg,
            rgba(31, 111, 235, 0.92),
            rgba(47, 129, 247, 0.88),
            rgba(88, 166, 255, 0.84)
        ) !important;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}

@media (max-width: 820px) {
    .bg-neural,
    .bg-badge,
    .scan-beam,
    .tech-ring {
        opacity: 0.07 !important;
        transform: scale(0.82);
    }


    .bg-neural,
    .bg-badge,
    .scan-beam,
    .tech-ring {
        opacity: 0.08 !important;
        transform: scale(0.85);
    }


    .bg-neural,
    .bg-badge,
    .scan-beam {
        opacity: 0.12;
        transform: scale(0.9);
    }


    .app-shell { padding: 4px 3px 22px; }

    .topbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .hero {
        min-height: auto;
        padding: 32px 22px;
        border-radius: 25px;
    }

    .hero h1 { font-size: 36px; }

    .hero-visuals {
        opacity: .38;
    }

    .neural-canvas {
        right: -70px;
        top: 80px;
        transform: scale(.78);
    }

    .team-grid,
    .feature-strip {
        grid-template-columns: 1fr;
    }

    .team-card { text-align: left; }

    .app-tabs { border-radius: 22px !important; }

    .section-card { padding: 11px !important; }
}
"""

with gr.Blocks(
    title="LSTM Sentiment Analysis",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as app:

    with gr.Column(elem_classes=["app-shell"]):

        gr.HTML(
            """
            
<div class="tech-background" aria-hidden="true">
                <div class="aurora a1"></div>
                <div class="aurora a2"></div>
                <div class="aurora a3"></div>

                <div class="data-dot d1"></div>
                <div class="data-dot d2"></div>
                <div class="data-dot d3"></div>
                <div class="data-dot d4"></div>
                <div class="data-dot d5"></div>
                <div class="data-dot d6"></div>
                <div class="data-dot d7"></div>
                <div class="data-dot d8"></div>
                <div class="data-dot d9"></div>
                <div class="data-dot d10"></div>
                <div class="data-dot d11"></div>
                <div class="data-dot d12"></div>

                <div class="tech-ring r1"></div>
                <div class="tech-ring r2"></div>
                <div class="tech-ring r3"></div>
                <div class="tech-ring r4"></div>
                <div class="tech-ring r5"></div>
                <div class="tech-ring r6"></div>
                <div class="tech-ring r7"></div>

                <div class="scan-beam s1"></div>
                <div class="scan-beam s2"></div>
                <div class="scan-beam s3"></div>
                <div class="scan-beam s4"></div>

                <div class="bg-badge bg-ai">AI</div>
                <div class="bg-badge bg-ai2">AI</div>
                <div class="bg-badge bg-ai3">AI</div>
                <div class="bg-badge bg-nlp2">NLP</div>
                <div class="bg-badge bg-nlp3">NLP</div>
                <div class="bg-badge bg-dl2">Deep Learning</div>
                <div class="bg-badge bg-dl3">Deep Learning</div>
                <div class="bg-badge bg-robot2">🤖</div>
                <div class="bg-badge bg-robot3">🤖</div>

                <svg class="bg-neural net-one" viewBox="0 0 220 120">
                    <line x1="25" y1="20" x2="95" y2="35"/><line x1="25" y1="20" x2="95" y2="85"/>
                    <line x1="25" y1="60" x2="95" y2="35"/><line x1="25" y1="60" x2="95" y2="85"/>
                    <line x1="25" y1="100" x2="95" y2="35"/><line x1="25" y1="100" x2="95" y2="85"/>
                    <line x1="95" y1="35" x2="165" y2="60"/><line x1="95" y1="85" x2="165" y2="60"/>
                    <circle cx="25" cy="20" r="6"/><circle cx="25" cy="60" r="6"/><circle cx="25" cy="100" r="6"/>
                    <circle cx="95" cy="35" r="7"/><circle cx="95" cy="85" r="7"/><circle cx="165" cy="60" r="8"/>
                </svg>

                <svg class="bg-neural net-two" viewBox="0 0 240 130">
                    <line x1="30" y1="20" x2="110" y2="40"/><line x1="30" y1="65" x2="110" y2="40"/>
                    <line x1="30" y1="110" x2="110" y2="90"/><line x1="110" y1="40" x2="185" y2="65"/>
                    <line x1="110" y1="90" x2="185" y2="65"/>
                    <circle cx="30" cy="20" r="6"/><circle cx="30" cy="65" r="6"/><circle cx="30" cy="110" r="6"/>
                    <circle cx="110" cy="40" r="7"/><circle cx="110" cy="90" r="7"/><circle cx="185" cy="65" r="8"/>
                </svg>

                <svg class="bg-neural net-three" viewBox="0 0 210 120">
                    <line x1="20" y1="25" x2="85" y2="60"/><line x1="20" y1="95" x2="85" y2="60"/>
                    <line x1="85" y1="60" x2="155" y2="30"/><line x1="85" y1="60" x2="155" y2="90"/>
                    <line x1="155" y1="30" x2="190" y2="60"/><line x1="155" y1="90" x2="190" y2="60"/>
                    <circle cx="20" cy="25" r="6"/><circle cx="20" cy="95" r="6"/><circle cx="85" cy="60" r="8"/>
                    <circle cx="155" cy="30" r="6"/><circle cx="155" cy="90" r="6"/><circle cx="190" cy="60" r="8"/>
                </svg>

                <svg class="bg-neural net-four" viewBox="0 0 250 130">
                    <line x1="20" y1="30" x2="85" y2="40"/><line x1="20" y1="95" x2="85" y2="40"/>
                    <line x1="85" y1="40" x2="150" y2="65"/><line x1="85" y1="40" x2="150" y2="105"/>
                    <line x1="150" y1="65" x2="220" y2="80"/><line x1="150" y1="105" x2="220" y2="80"/>
                    <circle cx="20" cy="30" r="6"/><circle cx="20" cy="95" r="6"/><circle cx="85" cy="40" r="7"/>
                    <circle cx="150" cy="65" r="8"/><circle cx="150" cy="105" r="7"/><circle cx="220" cy="80" r="8"/>
                </svg>

                <svg class="bg-neural net-five" viewBox="0 0 230 120">
                    <line x1="25" y1="25" x2="80" y2="55"/><line x1="25" y1="95" x2="80" y2="55"/>
                    <line x1="80" y1="55" x2="145" y2="25"/><line x1="80" y1="55" x2="145" y2="95"/>
                    <line x1="145" y1="25" x2="200" y2="55"/><line x1="145" y1="95" x2="200" y2="55"/>
                    <circle cx="25" cy="25" r="6"/><circle cx="25" cy="95" r="6"/><circle cx="80" cy="55" r="8"/>
                    <circle cx="145" cy="25" r="6"/><circle cx="145" cy="95" r="6"/><circle cx="200" cy="55" r="8"/>
                </svg>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="topbar">
                <div class="brand">
                    <div class="brand-mark"></div>
                    <span>Sentiment Intelligence Studio</span>
                </div>
                <div class="status-cluster">
                    <div class="status-pill"><span class="status-dot"></span> Model Ready</div>
                    <div class="status-pill">LSTM + DistilBART</div>
                    <div class="status-pill">Interactive NLP</div>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <section class="hero">
                <div class="hero-visuals" aria-hidden="true">
                    <div class="network-cluster">
                    <svg class="neural-canvas" viewBox="0 0 340 230">
                        <line class="neural-line" x1="35" y1="45" x2="125" y2="95"/>
                        <line class="neural-line" x1="35" y1="45" x2="125" y2="165"/>
                        <line class="neural-line" x1="35" y1="115" x2="125" y2="95"/>
                        <line class="neural-line" x1="35" y1="115" x2="125" y2="165"/>
                        <line class="neural-line" x1="35" y1="185" x2="125" y2="95"/>
                        <line class="neural-line" x1="35" y1="185" x2="125" y2="165"/>
                        <line class="neural-line" x1="125" y1="95" x2="225" y2="70"/>
                        <line class="neural-line" x1="125" y1="95" x2="225" y2="150"/>
                        <line class="neural-line" x1="125" y1="165" x2="225" y2="70"/>
                        <line class="neural-line" x1="125" y1="165" x2="225" y2="150"/>
                        <line class="neural-line" x1="225" y1="70" x2="305" y2="112"/>
                        <line class="neural-line" x1="225" y1="150" x2="305" y2="112"/>

                        <circle class="neural-node" cx="35" cy="45" r="10"/>
                        <circle class="neural-node n2" cx="35" cy="115" r="10"/>
                        <circle class="neural-node n3" cx="35" cy="185" r="10"/>
                        <circle class="neural-node n2" cx="125" cy="95" r="12"/>
                        <circle class="neural-node n4" cx="125" cy="165" r="12"/>
                        <circle class="neural-node n3" cx="225" cy="70" r="13"/>
                        <circle class="neural-node n2" cx="225" cy="150" r="13"/>
                        <circle class="neural-node n4" cx="305" cy="112" r="15"/>
                    </svg>

                    <div class="floating-chip c1">✦ Semantic Layer</div>
                    <div class="floating-chip c2">AI Confidence</div>
                    <div class="floating-chip c3">NLP Engine</div>
                    <div class="floating-chip c4">Deep Learning</div>
                    </div>
                </div>

                <div class="hero-content">
                    <div class="hero-kicker"><i></i> Intelligent Language Analytics</div>
                    <h1>Sentiment Analysis & Smart Summarization</h1>
                    <p>
                        A polished AI workspace for classifying review sentiment,
                        generating concise summaries, searching textual datasets,
                        and exploring interactive sentiment insights.
                    </p>

                    <div class="hero-metrics">
                        <div class="metric">
                            <strong>3-Class</strong>
                            <span>Sentiment Prediction</span>
                        </div>
                        <div class="metric">
                            <strong>LSTM</strong>
                            <span>Neural Classification</span>
                        </div>
                        <div class="metric">
                            <strong>DistilBART</strong>
                            <span>Smart Summarization</span>
                        </div>
                    </div>
                </div>

                <div class="team-grid">
                    <div class="team-card">
                        <span>Team Member</span>
                        SAIF KHATTAB
                    </div>
                    <div class="team-card">
                        <span>Team Member</span>
                        ASEEL MANNOUN
                    </div>
                    <div class="team-card">
                        <span>Team Member</span>
                        YAZEED AMEEN
                    </div>
                </div>
            </section>
            """
        )

        gr.HTML(
            """
            <div class="feature-strip">
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <strong>LSTM Classification</strong>
                    <small>Predicts positive, neutral, or negative sentiment with confidence scores.</small>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">✦</div>
                    <strong>Smart Summarization</strong>
                    <small>Transforms longer reviews into focused and readable summaries.</small>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⌁</div>
                    <strong>Dataset Intelligence</strong>
                    <small>Searches reviews and visualizes sentiment distribution interactively.</small>
                </div>
            </div>
            """
        )

        with gr.Tabs(elem_classes=["app-tabs"]):

            with gr.Tab("📝 Analyze Review"):
                with gr.Column(elem_classes=["section-card"]):
                    gr.Markdown(
                        "### Analyze a Review\n"
                        "Enter an English review to generate its summary, sentiment, "
                        "confidence score, and processing stages."
                    )

                    input_text = gr.Textbox(
                        lines=7,
                        label="Review Text",
                        placeholder="Write or paste an English food or product review...",
                    )

                    analyze_button = gr.Button(
                        "Analyze Review",
                        variant="primary",
                        elem_classes=["primary-action"],
                    )

                    with gr.Row(equal_height=True):
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
                        "🔍 Text Processing Stages",
                        open=False,
                    ):
                        gr.Markdown(
                            "These fields show how the review is transformed "
                            "before it reaches the LSTM model."
                        )
                        raw_tokens_output = gr.Textbox(label="1. Raw Tokens")
                        no_punctuation_output = gr.Textbox(label="2. No Punctuation")
                        no_stop_words_output = gr.Textbox(label="3. No Stop Words")
                        lemmatized_output = gr.Textbox(label="4. Lemmatized")

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

            with gr.Tab("🔎 Search Dataset"):
                with gr.Column(elem_classes=["section-card"]):
                    gr.Markdown(
                        "### Search Reviews\n"
                        "Search the dataset by a word or feature and view the closest results."
                    )

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                label="Search Query",
                                placeholder="Examples: coffee, taste, quality, price",
                            )
                            search_button = gr.Button(
                                "Search Dataset",
                                elem_classes=["secondary-action"],
                            )

                        with gr.Column(scale=1):
                            history_output = gr.Textbox(
                                label="Recent Searches",
                                lines=5,
                                interactive=False,
                            )

                    results_output = gr.Textbox(
                        label="Matching Reviews",
                        lines=12,
                    )

                    search_button.click(
                        fn=search_in_dataset_with_history,
                        inputs=query_input,
                        outputs=[results_output, history_output],
                    )

            with gr.Tab("📊 Analytics Dashboard"):
                with gr.Column(elem_classes=["section-card"]):
                    gr.Markdown(
                        "### Sentiment Dashboard\n"
                        "Filter the dataset by a feature to view its sentiment distribution."
                    )

                    with gr.Row(equal_height=True):
                        dashboard_query = gr.Textbox(
                            label="Feature Filter",
                            placeholder="Examples: coffee, food, tea, quality",
                            scale=3,
                        )
                        dashboard_button = gr.Button(
                            "Generate Chart",
                            elem_classes=["primary-action"],
                            scale=1,
                        )

                    plot_output = gr.Plot(
                        label="Sentiment Distribution",
                        elem_classes=["dashboard-plot"],
                    )

                    dashboard_button.click(
                        fn=generate_dashboard,
                        inputs=dashboard_query,
                        outputs=plot_output,
                    )

        gr.HTML(
            """
            <div class="footer-note">
                Built with TensorFlow, Keras, Transformers, and Gradio.
            </div>
            """
        )


if __name__ == "__main__":
    app.launch()