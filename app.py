# # # # # # import pandas as pd
# # # # # # import re
# # # # # # import nltk
# # # # # # from nltk.corpus import stopwords
# # # # # # from nltk.tokenize import word_tokenize
# # # # # # from nltk.stem import WordNetLemmatizer
# # # # # # import string
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # from sklearn.linear_model import LogisticRegression  # تم التعديل هنا
# # # # # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # # # # # import gradio as ui
# # # # # # import matplotlib.pyplot as plt
# # # # # # from thefuzz import process

# # # # # # # 1. تنزيل الحزم الأساسية من nltk للسيرفر السحابي
# # # # # # nltk.download('punkt')
# # # # # # nltk.download('punkt_tab')
# # # # # # nltk.download('stopwords')
# # # # # # nltk.download('wordnet')

# # # # # # # إعداد أدوات تنظيف النص
# # # # # # stop_words = set(stopwords.words('english'))
# # # # # # negation_words = {'not', 'no', 'nor', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "never"}
# # # # # # stop_words = stop_words.difference(negation_words)
# # # # # # lemmatizer = WordNetLemmatizer()

# # # # # # # 2. تحميل البيانات (تأكد أن ملف Reviews.csv مرفوع في المساحة)
# # # # # # print("جاري تحميل البيانات...")
# # # # # # df = pd.read_csv('Reviews.csv', nrows=100000)
# # # # # # df = df[['Text', 'Summary', 'Score']].dropna()

# # # # # # # 3. دوال المعالجة
# # # # # # def map_score_to_sentiment(score):
# # # # # #     if score in [1, 2]: return 'negative'
# # # # # #     elif score == 3: return 'neutral'
# # # # # #     else: return 'positive'

# # # # # # df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)

# # # # # # def clean_html_tags(text):
# # # # # #     cleaned_text = re.sub(r'<.*?>', ' ', str(text))
# # # # # #     return ' '.join(cleaned_text.split())

# # # # # # def preprocess_text(text):
# # # # # #     text = text.lower()
# # # # # #     text = text.translate(str.maketrans('', '', string.punctuation))
# # # # # #     tokens = word_tokenize(text)
# # # # # #     cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
# # # # # #     return " ".join(cleaned_tokens)

# # # # # # print("جاري تنظيف ومعالجة النصوص...")
# # # # # # df['Cleaned_Text'] = df['Text'].apply(clean_html_tags)
# # # # # # df['Processed_Text'] = df['Cleaned_Text'].apply(preprocess_text)

# # # # # # # 4. بناء وتدريب نموذج Logistic Regression
# # # # # # print("جاري تدريب نموذج تحليل المشاعر...")
# # # # # # X = df['Processed_Text']
# # # # # # y = df['Sentiment']
# # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # # # tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# # # # # # X_train_tfidf = tfidf.fit_transform(X_train)

# # # # # # # تم التعديل هنا لتدريب الموديل الجديد
# # # # # # model = LogisticRegression(class_weight='balanced', max_iter=1000)
# # # # # # model.fit(X_train_tfidf, y_train)

# # # # # # # 5. تحميل نموذج التلخيص (Transformers)
# # # # # # print("جاري تحميل نموذج التلخيص...")
# # # # # # model_name = "sshleifer/distilbart-cnn-12-6"
# # # # # # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # # # # # sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # # # # # # def analyze_and_summarize_review(raw_text):
# # # # # # #     cleaned = clean_html_tags(raw_text)
# # # # # # #     processed = preprocess_text(cleaned)
    
# # # # # # #     text_tfidf = tfidf.transform([processed])
# # # # # # #     sentiment_prediction = model.predict(text_tfidf)[0]
    
# # # # # # #     if len(raw_text.split()) > 10:
# # # # # # #         inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
# # # # # # #         summary_ids = sum_model.generate(inputs["input_ids"], max_length=30, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
# # # # # # #         summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# # # # # # #     else:
# # # # # # #         summary_text = "النص قصير جداً ولا يحتاج إلى تلخيص."
        
# # # # # # #     return {
# # # # # # #         "الملخص الذكي (Summary)": summary_text,
# # # # # # #         "تحليل المشاعر (Sentiment)": sentiment_prediction
# # # # # # #     }
# # # # # # def analyze_and_summarize_review(raw_text):
# # # # # #     cleaned = clean_html_tags(raw_text)
# # # # # #     processed = preprocess_text(cleaned)
    
# # # # # #     # 1. تحليل المشاعر
# # # # # #     text_tfidf = tfidf.transform([processed])
# # # # # #     sentiment_prediction = model.predict(text_tfidf)[0]
    
# # # # # #     # 2. حساب عدد كلمات النص المدخل
# # # # # #     word_count = len(raw_text.split())
    
# # # # # #     # الشرط الحاسم الذي طلبته: إذا كان أكبر من 15 كلمة يلخص، غير ذلك يطبع الرسالة
# # # # # #     if word_count > 10:
# # # # # #         inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
        
# # # # # #         # بارامترات صارمة تمنع النموذج من زيادة أي كلام أو تأليف
# # # # # #         summary_ids = sum_model.generate(
# # # # # #             inputs["input_ids"], 
# # # # # #             max_new_tokens=20,           # حد أقصى صغير جداً للملخص ليجبره على الاختصار الشديد
# # # # # #             no_repeat_ngram_size=2,      # يمنع تكرار العبارات نهائياً
# # # # # #             num_beams=2,                 # تقليل عدد الاحتمالات لمنع التخريف
# # # # # #             early_stopping=True
# # # # # #         )
# # # # # #         summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
# # # # # #         # حماية إضافية: لو الموديل هلوَس برضه وطلع التلخيص أطول أو مساوي للنص الأصلي، يعرض النص الأصلي فقط
# # # # # #         if len(summary_text.split()) >= word_count:
# # # # # #             summary_text = raw_text.strip()
# # # # # #     else:
# # # # # #         # إذا كان 15 كلمة أو أقل، يطبع هذه الرسالة مباشرة ولا يشغل نموذج التلخيص
# # # # # #         summary_text = "The text is too short to require summarization. / النص قصير جداً ولا يحتاج إلى تلخيص."
        
# # # # # #     return {
# # # # # #         "الملخص الذكي (Summary)": summary_text,
# # # # # #         "تحليل المشاعر (Sentiment)": sentiment_prediction
# # # # # #     }

# # # # # # # 6. إعداد دوال واجهة Gradio
# # # # # # search_history = []

# # # # # # def generate_dashboard(query):
# # # # # #     if query and query.strip():
# # # # # #         filtered_df = df[df['Processed_Text'].str.contains(query.lower(), case=False, na=False)]
# # # # # #         title = f"Sentiment Distribution for: {query.capitalize()}"
# # # # # #     else:
# # # # # #         filtered_df = df
# # # # # #         title = "General Sentiment Distribution"
        
# # # # # #     if filtered_df.empty:
# # # # # #         fig, ax = plt.subplots()
# # # # # #         ax.text(0.5, 0.5, 'No data found for this feature', ha='center', fontsize=12)
# # # # # #         ax.axis('off')
# # # # # #         return fig
        
# # # # # #     sentiment_counts = filtered_df['Sentiment'].value_counts()
# # # # # #     fig, ax = plt.subplots(figsize=(6, 6))
# # # # # #     colors = ['#50fa7b', '#ff5555', '#8be9fd']
    
# # # # # #     sentiment_counts.plot(
# # # # # #         kind='pie', 
# # # # # #         autopct='%1.1f%%', 
# # # # # #         startangle=140, 
# # # # # #         colors=colors, 
# # # # # #         ax=ax,
# # # # # #         textprops={'fontsize': 12, 'weight': 'bold'}
# # # # # #     )
    
# # # # # #     ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
# # # # # #     ax.set_ylabel('') 
# # # # # #     plt.tight_layout()
# # # # # #     return fig

# # # # # # def web_interface_fn(review_text):
# # # # # #     if not review_text.strip():
# # # # # #         return "Please enter a review.", "", "", "", "", ""
    
# # # # # #     tokens_raw = word_tokenize(review_text.lower())
# # # # # #     text_no_punct = review_text.lower().translate(str.maketrans('', '', string.punctuation))
# # # # # #     tokens_no_punct = word_tokenize(text_no_punct)
# # # # # #     no_stop_words = [w for w in tokens_no_punct if w not in stop_words]
# # # # # #     lemmatized = [lemmatizer.lemmatize(w) for w in no_stop_words]
    
# # # # # #     results = analyze_and_summarize_review(review_text) 
    
# # # # # #     return (
# # # # # #         results["الملخص الذكي (Summary)"], 
# # # # # #         results["تحليل المشاعر (Sentiment)"].upper() + (" 😊" if results["تحليل المشاعر (Sentiment)"] == 'positive' else " 😢"),
# # # # # #         " | ".join(tokens_raw),
# # # # # #         " | ".join(tokens_no_punct),
# # # # # #         " | ".join(no_stop_words),
# # # # # #         " | ".join(lemmatized)
# # # # # #     )

# # # # # # def search_in_dataset_with_history(query):
# # # # # #     global search_history
# # # # # #     if not query.strip(): return "Please enter a search term.", "\n".join(search_history)
    
# # # # # #     common_keywords = ['battery', 'screen', 'camera', 'price', 'performance', 'design', 'software', 'quality', 'money']
# # # # # #     match = process.extractOne(query.lower(), common_keywords)
# # # # # #     actual_query = match[0] if match and match[1] > 80 else query.lower().strip()

# # # # # #     if query not in search_history:
# # # # # #         search_history.insert(0, query)
# # # # # #         search_history = search_history[:5]
    
# # # # # #     results = df[df['Text'].str.contains(actual_query, case=False, na=False)]
    
# # # # # #     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"
# # # # # #     for idx, row in results.head(5).iterrows():
# # # # # #         output += f"📝: {str(row['Text'])[:120]}...\n📊: {str(row['Sentiment']).upper()}\n\n"
        
# # # # # #     return output, "\n".join(search_history)

# # # # # # # 7. بناء وتشغيل واجهة Gradio
# # # # # # with ui.Blocks(title="NLP Project", theme=ui.themes.Soft()) as app:
# # # # # #     ui.Markdown("""
# # # # # #     <div style="text-align: center; background-color: #1e1e2e; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
# # # # # #         <h1 style="color: #ff79c6;">💻 NLP Project</h1>
# # # # # #         <div style="display: flex; justify-content: space-around; color: white; font-size: 16px;">
# # # # # #             <div>👤 SAIF KHATTAB <br> 🆔 202312556</div>
# # # # # #             <div>👤 YAZEED AMEEN <br> 🆔 202310058</div>
# # # # # #         </div>
# # # # # #     </div>
# # # # # #     """)
    
# # # # # #     with ui.Tabs():
# # # # # #         with ui.TabItem("Analyze Text 📝"):
# # # # # #             input_text = ui.Textbox(lines=3, label="Enter Review")
# # # # # #             btn = ui.Button("Analyze 🚀", variant="primary")
            
# # # # # #             with ui.Row():
# # # # # #                 out1 = ui.Textbox(label="Summary")
# # # # # #                 out2 = ui.Textbox(label="Sentiment")
            
# # # # # #             with ui.Accordion("Text Processing Stages (Transparency Mode)", open=False):
# # # # # #                 out_tokens = ui.Textbox(label="1. Raw Tokens")
# # # # # #                 out_no_punct = ui.Textbox(label="1.5 No Punctuation")
# # # # # #                 out_no_stop = ui.Textbox(label="2. No Stop Words")
# # # # # #                 out_lemmatized = ui.Textbox(label="3. Lemmatized")
            
# # # # # #             btn.click(web_interface_fn, input_text, [out1, out2, out_tokens, out_no_punct, out_no_stop, out_lemmatized])
            
# # # # # #         with ui.TabItem("Search Dataset 🔍"):
# # # # # #             with ui.Row():
# # # # # #                 with ui.Column():
# # # # # #                     query_input = ui.Textbox(label="Search Query")
# # # # # #                     search_btn = ui.Button("Search Data 🔎")
# # # # # #                 history_output = ui.Textbox(label="Last 5 Searches", interactive=False)
# # # # # #             results_out = ui.Textbox(label="Results", lines=8)
# # # # # #             search_btn.click(search_in_dataset_with_history, query_input, [results_out, history_output])
            
# # # # # #         with ui.TabItem("Analytics Dashboard 📊"):
# # # # # #             dash_query = ui.Textbox(label="Filter stats by feature (Optional)", placeholder="e.g., food,tea")
# # # # # #             dashboard_btn = ui.Button("Show Chart 🚀")
# # # # # #             plot_output = ui.Plot(label="Sentiment Distribution")
# # # # # #             dashboard_btn.click(generate_dashboard, inputs=dash_query, outputs=plot_output)

# # # # # # # السطر الأخير بدون أي إضافات ليعمل على السحابة
# # # # # # app.launch()
# # # # # import os
# # # # # import re
# # # # # import csv
# # # # # import string
# # # # # import pickle
# # # # # import warnings

# # # # # import gradio as gr
# # # # # import matplotlib.pyplot as plt
# # # # # import nltk
# # # # # import numpy as np
# # # # # import pandas as pd

# # # # # from nltk.corpus import stopwords
# # # # # from nltk.tokenize import word_tokenize
# # # # # from nltk.stem import WordNetLemmatizer

# # # # # from tensorflow.keras.models import load_model
# # # # # from tensorflow.keras.preprocessing.sequence import pad_sequences

# # # # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # # # # from thefuzz import process


# # # # # # =========================================================
# # # # # # 1. General settings
# # # # # # =========================================================

# # # # # warnings.filterwarnings("ignore")

# # # # # LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
# # # # # TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
# # # # # LABEL_ENCODER_PATH = "label_encoder.pkl"
# # # # # REVIEWS_FILE_PATH = "Reviews.csv"

# # # # # SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# # # # # SUMMARY_WORD_THRESHOLD = 15
# # # # # DEFAULT_MAX_LENGTH = 111


# # # # # # =========================================================
# # # # # # 2. Download NLTK packages
# # # # # # =========================================================

# # # # # for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
# # # # #     nltk.download(package, quiet=True)

# # # # # stop_words = set(stopwords.words("english"))

# # # # # # Keep negation words because they affect sentiment strongly.
# # # # # negation_words = {"not", "no", "nor", "never"}
# # # # # stop_words = stop_words.difference(negation_words)

# # # # # lemmatizer = WordNetLemmatizer()


# # # # # # =========================================================
# # # # # # 3. Text preprocessing
# # # # # # Must match the preprocessing used during training
# # # # # # =========================================================

# # # # # def clean_html_tags(text):
# # # # #     text = str(text)
# # # # #     text = re.sub(r"<[^>]*>", " ", text)
# # # # #     text = re.sub(r"\s+", " ", text)
# # # # #     return text.strip()


# # # # # def normalize_negations(text):
# # # # #     text = str(text).replace("’", "'")

# # # # #     contractions = {
# # # # #         "can't": "can not",
# # # # #         "cannot": "can not",
# # # # #         "won't": "will not",
# # # # #         "don't": "do not",
# # # # #         "doesn't": "does not",
# # # # #         "didn't": "did not",
# # # # #         "isn't": "is not",
# # # # #         "aren't": "are not",
# # # # #         "wasn't": "was not",
# # # # #         "weren't": "were not",
# # # # #         "hasn't": "has not",
# # # # #         "haven't": "have not",
# # # # #         "hadn't": "had not",
# # # # #         "shouldn't": "should not",
# # # # #         "wouldn't": "would not",
# # # # #         "couldn't": "could not",
# # # # #         "mustn't": "must not",
# # # # #         "needn't": "need not",
# # # # #         "mightn't": "might not",
# # # # #         "shan't": "shall not",
# # # # #     }

# # # # #     for contraction, expanded_form in contractions.items():
# # # # #         text = re.sub(
# # # # #             rf"\b{re.escape(contraction)}\b",
# # # # #             expanded_form,
# # # # #             text,
# # # # #             flags=re.IGNORECASE,
# # # # #         )

# # # # #     return text


# # # # # def preprocess_text(text):
# # # # #     text = str(text).lower()
# # # # #     text = normalize_negations(text)

# # # # #     text = re.sub(r"http\S+|www\S+", " ", text)
# # # # #     text = re.sub(r"\S+@\S+", " ", text)

# # # # #     text = text.translate(str.maketrans("", "", string.punctuation))
# # # # #     text = re.sub(r"\d+", " ", text)
# # # # #     text = re.sub(r"\s+", " ", text).strip()

# # # # #     tokens = word_tokenize(text)

# # # # #     cleaned_tokens = []

# # # # #     for word in tokens:
# # # # #         if not word.isalpha():
# # # # #             continue

# # # # #         if word in stop_words:
# # # # #             continue

# # # # #         cleaned_tokens.append(lemmatizer.lemmatize(word))

# # # # #     return " ".join(cleaned_tokens)


# # # # # # =========================================================
# # # # # # 4. Load trained LSTM files
# # # # # # =========================================================

# # # # # def require_file(file_path):
# # # # #     if not os.path.exists(file_path):
# # # # #         raise FileNotFoundError(
# # # # #             f"Required file '{file_path}' was not found. "
# # # # #             "Upload it to the root of the Hugging Face Space."
# # # # #         )


# # # # # for required_file in [
# # # # #     LSTM_MODEL_PATH,
# # # # #     TEXT_TOKENIZER_PATH,
# # # # #     LABEL_ENCODER_PATH,
# # # # # ]:
# # # # #     require_file(required_file)

# # # # # print("Loading trained LSTM sentiment model...")

# # # # # sentiment_model = load_model(LSTM_MODEL_PATH)

# # # # # with open(TEXT_TOKENIZER_PATH, "rb") as tokenizer_file:
# # # # #     text_tokenizer = pickle.load(tokenizer_file)

# # # # # with open(LABEL_ENCODER_PATH, "rb") as encoder_file:
# # # # #     label_encoder = pickle.load(encoder_file)

# # # # # try:
# # # # #     model_input_length = sentiment_model.input_shape[1]
# # # # #     MAX_LENGTH = int(model_input_length) if model_input_length else DEFAULT_MAX_LENGTH
# # # # # except (TypeError, ValueError, IndexError):
# # # # #     MAX_LENGTH = DEFAULT_MAX_LENGTH

# # # # # print(f"LSTM model loaded successfully. MAX_LENGTH = {MAX_LENGTH}")


# # # # # # =========================================================
# # # # # # 5. Load pretrained summarization model
# # # # # # =========================================================

# # # # # print("Loading DistilBART summarization model...")

# # # # # summary_tokenizer = AutoTokenizer.from_pretrained(
# # # # #     SUMMARIZATION_MODEL_NAME
# # # # # )

# # # # # summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
# # # # #     SUMMARIZATION_MODEL_NAME
# # # # # )

# # # # # print("Summarization model loaded successfully.")


# # # # # # =========================================================
# # # # # # 6. Optional dataset loading
# # # # # # =========================================================

# # # # # def map_score_to_sentiment(score):
# # # # #     if score in [1, 2]:
# # # # #         return "negative"
# # # # #     if score == 3:
# # # # #         return "neutral"
# # # # #     return "positive"


# # # # # def read_csv_safely(file_path, max_valid_rows=100000):
# # # # #     valid_rows = []
# # # # #     bad_rows = []

# # # # #     with open(
# # # # #         file_path,
# # # # #         mode="r",
# # # # #         encoding="utf-8",
# # # # #         errors="replace",
# # # # #         newline="",
# # # # #     ) as file:
# # # # #         reader = csv.reader(
# # # # #             file,
# # # # #             delimiter=",",
# # # # #             quotechar='"',
# # # # #             strict=True,
# # # # #         )

# # # # #         try:
# # # # #             columns = next(reader)
# # # # #         except StopIteration:
# # # # #             raise ValueError("Reviews.csv is empty.")

# # # # #         columns = [
# # # # #             column.strip().replace("\ufeff", "")
# # # # #             for column in columns
# # # # #         ]

# # # # #         expected_columns = len(columns)

# # # # #         while len(valid_rows) < max_valid_rows:
# # # # #             try:
# # # # #                 row = next(reader)
# # # # #             except StopIteration:
# # # # #                 break
# # # # #             except csv.Error as error:
# # # # #                 bad_rows.append(
# # # # #                     {"line": reader.line_num, "error": str(error)}
# # # # #                 )
# # # # #                 continue

# # # # #             if len(row) != expected_columns:
# # # # #                 bad_rows.append(
# # # # #                     {
# # # # #                         "line": reader.line_num,
# # # # #                         "error": (
# # # # #                             f"Expected {expected_columns} fields, "
# # # # #                             f"found {len(row)}."
# # # # #                         ),
# # # # #                     }
# # # # #                 )
# # # # #                 continue

# # # # #             valid_rows.append(row)

# # # # #     return pd.DataFrame(valid_rows, columns=columns), bad_rows


# # # # # def load_reviews_dataset():
# # # # #     if not os.path.exists(REVIEWS_FILE_PATH):
# # # # #         print(
# # # # #             "Reviews.csv was not found. "
# # # # #             "Search and dashboard tabs will be unavailable."
# # # # #         )
# # # # #         return None

# # # # #     try:
# # # # #         dataset, bad_rows = read_csv_safely(
# # # # #             REVIEWS_FILE_PATH,
# # # # #             max_valid_rows=100000,
# # # # #         )

# # # # #         required_columns = {"Text", "Summary", "Score"}

# # # # #         if not required_columns.issubset(dataset.columns):
# # # # #             print(
# # # # #                 "Reviews.csv is missing required columns. "
# # # # #                 "Search and dashboard tabs will be unavailable."
# # # # #             )
# # # # #             return None

# # # # #         dataset = dataset[["Text", "Summary", "Score"]].copy()

# # # # #         dataset["Score"] = pd.to_numeric(
# # # # #             dataset["Score"],
# # # # #             errors="coerce",
# # # # #         )

# # # # #         dataset = dataset.dropna(
# # # # #             subset=["Text", "Summary", "Score"]
# # # # #         )

# # # # #         dataset = dataset[
# # # # #             dataset["Score"].isin([1, 2, 3, 4, 5])
# # # # #         ].copy()

# # # # #         dataset["Score"] = dataset["Score"].astype(int)
# # # # #         dataset["Sentiment"] = dataset["Score"].apply(
# # # # #             map_score_to_sentiment
# # # # #         )

# # # # #         dataset["Cleaned_Text"] = dataset["Text"].apply(
# # # # #             clean_html_tags
# # # # #         )

# # # # #         dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(
# # # # #             preprocess_text
# # # # #         )

# # # # #         dataset = dataset[
# # # # #             dataset["Processed_Text"].str.strip() != ""
# # # # #         ].reset_index(drop=True)

# # # # #         print(
# # # # #             f"Reviews.csv loaded: {len(dataset)} valid rows, "
# # # # #             f"{len(bad_rows)} corrupted rows skipped."
# # # # #         )

# # # # #         return dataset

# # # # #     except Exception as error:
# # # # #         print(f"Could not load Reviews.csv: {error}")
# # # # #         return None


# # # # # df = load_reviews_dataset()


# # # # # # =========================================================
# # # # # # 7. Sentiment prediction
# # # # # # =========================================================

# # # # # def predict_sentiment(raw_text):
# # # # #     raw_text = str(raw_text).strip()

# # # # #     if not raw_text:
# # # # #         raise ValueError("Please enter a review.")

# # # # #     cleaned_text = clean_html_tags(raw_text)
# # # # #     processed_text = preprocess_text(cleaned_text)

# # # # #     if not processed_text:
# # # # #         raise ValueError(
# # # # #             "The review became empty after preprocessing."
# # # # #         )

# # # # #     sequence = text_tokenizer.texts_to_sequences(
# # # # #         [processed_text]
# # # # #     )

# # # # #     padded_sequence = pad_sequences(
# # # # #         sequence,
# # # # #         maxlen=MAX_LENGTH,
# # # # #         padding="post",
# # # # #         truncating="post",
# # # # #     )

# # # # #     probabilities = sentiment_model.predict(
# # # # #         padded_sequence,
# # # # #         verbose=0,
# # # # #     )[0]

# # # # #     predicted_class_index = int(np.argmax(probabilities))

# # # # #     predicted_sentiment = label_encoder.inverse_transform(
# # # # #         [predicted_class_index]
# # # # #     )[0]

# # # # #     confidence = float(
# # # # #         probabilities[predicted_class_index]
# # # # #     )

# # # # #     probability_dictionary = {
# # # # #         class_name: float(probabilities[index])
# # # # #         for index, class_name in enumerate(
# # # # #             label_encoder.classes_
# # # # #         )
# # # # #     }

# # # # #     return (
# # # # #         predicted_sentiment,
# # # # #         confidence,
# # # # #         probability_dictionary,
# # # # #         processed_text,
# # # # #     )


# # # # # # =========================================================
# # # # # # 8. Summarization
# # # # # # =========================================================

# # # # # def summarize_review(raw_text):
# # # # #     raw_text = str(raw_text).strip()

# # # # #     if not raw_text:
# # # # #         raise ValueError("Please enter a review.")

# # # # #     word_count = len(raw_text.split())

# # # # #     if word_count <= SUMMARY_WORD_THRESHOLD:
# # # # #         return (
# # # # #             "The text is too short to require summarization. "
# # # # #             "/ النص قصير جداً ولا يحتاج إلى تلخيص."
# # # # #         )

# # # # #     inputs = summary_tokenizer(
# # # # #         raw_text,
# # # # #         max_length=1024,
# # # # #         return_tensors="pt",
# # # # #         truncation=True,
# # # # #     )

# # # # #     summary_ids = summarization_model.generate(
# # # # #         input_ids=inputs["input_ids"],
# # # # #         attention_mask=inputs["attention_mask"],
# # # # #         max_new_tokens=30,
# # # # #         no_repeat_ngram_size=2,
# # # # #         num_beams=4,
# # # # #         early_stopping=True,
# # # # #     )

# # # # #     summary_text = summary_tokenizer.decode(
# # # # #         summary_ids[0],
# # # # #         skip_special_tokens=True,
# # # # #     ).strip()

# # # # #     if not summary_text:
# # # # #         return raw_text

# # # # #     if len(summary_text.split()) >= word_count:
# # # # #         return raw_text

# # # # #     return summary_text


# # # # # # =========================================================
# # # # # # 9. Combined pipeline
# # # # # # Both models use the original review independently
# # # # # # =========================================================

# # # # # def analyze_and_summarize_review(raw_text):
# # # # #     (
# # # # #         sentiment,
# # # # #         confidence,
# # # # #         probabilities,
# # # # #         processed_text,
# # # # #     ) = predict_sentiment(raw_text)

# # # # #     summary_text = summarize_review(raw_text)

# # # # #     return {
# # # # #         "summary": summary_text,
# # # # #         "sentiment": sentiment,
# # # # #         "confidence": confidence,
# # # # #         "probabilities": probabilities,
# # # # #         "processed_text": processed_text,
# # # # #     }


# # # # # # =========================================================
# # # # # # 10. Gradio helper functions
# # # # # # =========================================================

# # # # # def sentiment_emoji(sentiment):
# # # # #     emoji_map = {
# # # # #         "positive": "😊",
# # # # #         "neutral": "😐",
# # # # #         "negative": "😢",
# # # # #     }

# # # # #     return emoji_map.get(sentiment.lower(), "🔎")


# # # # # def web_interface_fn(review_text):
# # # # #     if not review_text or not review_text.strip():
# # # # #         return (
# # # # #             "Please enter a review.",
# # # # #             "",
# # # # #             {},
# # # # #             "",
# # # # #             "",
# # # # #             "",
# # # # #             "",
# # # # #         )

# # # # #     try:
# # # # #         tokens_raw = word_tokenize(review_text.lower())

# # # # #         normalized_text = normalize_negations(
# # # # #             review_text.lower()
# # # # #         )

# # # # #         text_no_punct = normalized_text.translate(
# # # # #             str.maketrans("", "", string.punctuation)
# # # # #         )

# # # # #         text_no_punct = re.sub(
# # # # #             r"\d+",
# # # # #             " ",
# # # # #             text_no_punct,
# # # # #         )

# # # # #         tokens_no_punct = word_tokenize(
# # # # #             text_no_punct
# # # # #         )

# # # # #         no_stop_words = [
# # # # #             word
# # # # #             for word in tokens_no_punct
# # # # #             if word.isalpha() and word not in stop_words
# # # # #         ]

# # # # #         lemmatized = [
# # # # #             lemmatizer.lemmatize(word)
# # # # #             for word in no_stop_words
# # # # #         ]

# # # # #         results = analyze_and_summarize_review(
# # # # #             review_text
# # # # #         )

# # # # #         sentiment_text = (
# # # # #             f"{results['sentiment'].upper()} "
# # # # #             f"{sentiment_emoji(results['sentiment'])}\n"
# # # # #             f"Confidence: {results['confidence'] * 100:.2f}%"
# # # # #         )

# # # # #         probability_labels = {
# # # # #             class_name.capitalize(): probability
# # # # #             for class_name, probability in results[
# # # # #                 "probabilities"
# # # # #             ].items()
# # # # #         }

# # # # #         return (
# # # # #             results["summary"],
# # # # #             sentiment_text,
# # # # #             probability_labels,
# # # # #             " | ".join(tokens_raw),
# # # # #             " | ".join(tokens_no_punct),
# # # # #             " | ".join(no_stop_words),
# # # # #             " | ".join(lemmatized),
# # # # #         )

# # # # #     except Exception as error:
# # # # #         return (
# # # # #             f"Error: {error}",
# # # # #             "",
# # # # #             {},
# # # # #             "",
# # # # #             "",
# # # # #             "",
# # # # #             "",
# # # # #         )


# # # # # search_history = []


# # # # # def search_in_dataset_with_history(query):
# # # # #     global search_history

# # # # #     if df is None:
# # # # #         return (
# # # # #             "Reviews.csv is not available in this Space.",
# # # # #             "\n".join(search_history),
# # # # #         )

# # # # #     if not query or not query.strip():
# # # # #         return (
# # # # #             "Please enter a search term.",
# # # # #             "\n".join(search_history),
# # # # #         )

# # # # #     common_keywords = [
# # # # #         "battery",
# # # # #         "screen",
# # # # #         "camera",
# # # # #         "price",
# # # # #         "performance",
# # # # #         "design",
# # # # #         "software",
# # # # #         "quality",
# # # # #         "money",
# # # # #         "coffee",
# # # # #         "food",
# # # # #         "tea",
# # # # #         "taste",
# # # # #     ]

# # # # #     match = process.extractOne(
# # # # #         query.lower(),
# # # # #         common_keywords,
# # # # #     )

# # # # #     actual_query = (
# # # # #         match[0]
# # # # #         if match and match[1] > 80
# # # # #         else query.lower().strip()
# # # # #     )

# # # # #     if query not in search_history:
# # # # #         search_history.insert(0, query)
# # # # #         search_history = search_history[:5]

# # # # #     results = df[
# # # # #         df["Text"].str.contains(
# # # # #             actual_query,
# # # # #             case=False,
# # # # #             na=False,
# # # # #             regex=False,
# # # # #         )
# # # # #     ]

# # # # #     output = (
# # # # #         f"🔍 Found {len(results)} reviews "
# # # # #         f"for '{actual_query}'.\n\n"
# # # # #     )

# # # # #     if results.empty:
# # # # #         output += "No matching reviews were found."

# # # # #     for _, row in results.head(5).iterrows():
# # # # #         review_preview = str(row["Text"])[:180]

# # # # #         output += (
# # # # #             f"📝 {review_preview}...\n"
# # # # #             f"📊 {str(row['Sentiment']).upper()}\n\n"
# # # # #         )

# # # # #     return output, "\n".join(search_history)


# # # # # def generate_dashboard(query):
# # # # #     if df is None:
# # # # #         fig, ax = plt.subplots(figsize=(7, 5))
# # # # #         ax.text(
# # # # #             0.5,
# # # # #             0.5,
# # # # #             "Reviews.csv is not available.",
# # # # #             ha="center",
# # # # #             va="center",
# # # # #             fontsize=13,
# # # # #         )
# # # # #         ax.axis("off")
# # # # #         return fig

# # # # #     if query and query.strip():
# # # # #         filtered_df = df[
# # # # #             df["Text"].str.contains(
# # # # #                 query.strip(),
# # # # #                 case=False,
# # # # #                 na=False,
# # # # #                 regex=False,
# # # # #             )
# # # # #         ]

# # # # #         title = (
# # # # #             f"Sentiment Distribution for: "
# # # # #             f"{query.strip().capitalize()}"
# # # # #         )
# # # # #     else:
# # # # #         filtered_df = df
# # # # #         title = "General Sentiment Distribution"

# # # # #     if filtered_df.empty:
# # # # #         fig, ax = plt.subplots(figsize=(7, 5))
# # # # #         ax.text(
# # # # #             0.5,
# # # # #             0.5,
# # # # #             "No data found for this feature.",
# # # # #             ha="center",
# # # # #             va="center",
# # # # #             fontsize=13,
# # # # #         )
# # # # #         ax.axis("off")
# # # # #         return fig

# # # # #     sentiment_counts = (
# # # # #         filtered_df["Sentiment"]
# # # # #         .value_counts()
# # # # #         .reindex(
# # # # #             ["positive", "neutral", "negative"],
# # # # #             fill_value=0,
# # # # #         )
# # # # #     )

# # # # #     fig, ax = plt.subplots(figsize=(6, 6))

# # # # #     sentiment_counts.plot(
# # # # #         kind="pie",
# # # # #         autopct="%1.1f%%",
# # # # #         startangle=140,
# # # # #         ax=ax,
# # # # #         textprops={
# # # # #             "fontsize": 12,
# # # # #             "weight": "bold",
# # # # #         },
# # # # #     )

# # # # #     ax.set_title(
# # # # #         title,
# # # # #         fontsize=14,
# # # # #         fontweight="bold",
# # # # #         pad=20,
# # # # #     )

# # # # #     ax.set_ylabel("")
# # # # #     plt.tight_layout()

# # # # #     return fig


# # # # # # =========================================================
# # # # # # 11. Gradio interface
# # # # # # =========================================================

# # # # # with gr.Blocks(
# # # # #     title="LSTM Sentiment Analysis",
# # # # #     theme=gr.themes.Soft(),
# # # # # ) as app:

# # # # #     gr.Markdown(
# # # # #         """
# # # # #         <div style="
# # # # #             text-align: center;
# # # # #             background-color: #1e1e2e;
# # # # #             padding: 20px;
# # # # #             border-radius: 10px;
# # # # #             margin-bottom: 20px;
# # # # #         ">
# # # # #             <h1 style="color: #ff79c6;">
# # # # #                 💻 LSTM Sentiment Analysis & Summarization
# # # # #             </h1>

# # # # #             <p style="color: white;">
# # # # #                 Sentiment is predicted from the original review using a trained LSTM model.
# # # # #                 The original review is summarized independently using DistilBART.
# # # # #             </p>

# # # # #             <div style="
# # # # #                 display: flex;
# # # # #                 justify-content: space-around;
# # # # #                 color: white;
# # # # #                 font-size: 16px;
# # # # #             ">
# # # # #                 <div>
# # # # #                     👤 SAIF KHATTAB
# # # # #                     <br>
# # # # #                     🆔 202312556
# # # # #                 </div>

# # # # #                 <div>
# # # # #                     👤 YAZEED AMEEN
# # # # #                     <br>
# # # # #                     🆔 202310058
# # # # #                 </div>
# # # # #             </div>
# # # # #         </div>
# # # # #         """
# # # # #     )

# # # # #     with gr.Tabs():

# # # # #         with gr.Tab("Analyze Text 📝"):
# # # # #             input_text = gr.Textbox(
# # # # #                 lines=6,
# # # # #                 label="Enter Review",
# # # # #                 placeholder="Write an English product or food review...",
# # # # #             )

# # # # #             analyze_button = gr.Button(
# # # # #                 "Analyze 🚀",
# # # # #                 variant="primary",
# # # # #             )

# # # # #             with gr.Row():
# # # # #                 summary_output = gr.Textbox(
# # # # #                     label="Generated Summary",
# # # # #                     lines=5,
# # # # #                 )

# # # # #                 sentiment_output = gr.Textbox(
# # # # #                     label="Predicted Sentiment",
# # # # #                     lines=5,
# # # # #                 )

# # # # #             probability_output = gr.Label(
# # # # #                 label="Class Probabilities",
# # # # #                 num_top_classes=3,
# # # # #             )

# # # # #             with gr.Accordion(
# # # # #                 "Text Processing Stages (Transparency Mode)",
# # # # #                 open=False,
# # # # #             ):
# # # # #                 raw_tokens_output = gr.Textbox(
# # # # #                     label="1. Raw Tokens"
# # # # #                 )

# # # # #                 no_punctuation_output = gr.Textbox(
# # # # #                     label="2. No Punctuation"
# # # # #                 )

# # # # #                 no_stop_words_output = gr.Textbox(
# # # # #                     label="3. No Stop Words"
# # # # #                 )

# # # # #                 lemmatized_output = gr.Textbox(
# # # # #                     label="4. Lemmatized"
# # # # #                 )

# # # # #             analyze_button.click(
# # # # #                 fn=web_interface_fn,
# # # # #                 inputs=input_text,
# # # # #                 outputs=[
# # # # #                     summary_output,
# # # # #                     sentiment_output,
# # # # #                     probability_output,
# # # # #                     raw_tokens_output,
# # # # #                     no_punctuation_output,
# # # # #                     no_stop_words_output,
# # # # #                     lemmatized_output,
# # # # #                 ],
# # # # #             )

# # # # #         with gr.Tab("Search Dataset 🔍"):
# # # # #             with gr.Row():
# # # # #                 with gr.Column():
# # # # #                     query_input = gr.Textbox(
# # # # #                         label="Search Query"
# # # # #                     )

# # # # #                     search_button = gr.Button(
# # # # #                         "Search Data 🔎"
# # # # #                     )

# # # # #                 history_output = gr.Textbox(
# # # # #                     label="Last 5 Searches",
# # # # #                     interactive=False,
# # # # #                 )

# # # # #             results_output = gr.Textbox(
# # # # #                 label="Results",
# # # # #                 lines=10,
# # # # #             )

# # # # #             search_button.click(
# # # # #                 fn=search_in_dataset_with_history,
# # # # #                 inputs=query_input,
# # # # #                 outputs=[
# # # # #                     results_output,
# # # # #                     history_output,
# # # # #                 ],
# # # # #             )

# # # # #         with gr.Tab("Analytics Dashboard 📊"):
# # # # #             dashboard_query = gr.Textbox(
# # # # #                 label="Filter statistics by feature",
# # # # #                 placeholder="Examples: coffee, food, tea, quality",
# # # # #             )

# # # # #             dashboard_button = gr.Button(
# # # # #                 "Show Chart 🚀"
# # # # #             )

# # # # #             plot_output = gr.Plot(
# # # # #                 label="Sentiment Distribution"
# # # # #             )

# # # # #             dashboard_button.click(
# # # # #                 fn=generate_dashboard,
# # # # #                 inputs=dashboard_query,
# # # # #                 outputs=plot_output,
# # # # #             )


# # # # # if __name__ == "__main__":
# # # # #     app.launch()
# # # # import os

# # # # # Must be set before importing TensorFlow or PyTorch.
# # # # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # # # os.environ["OMP_NUM_THREADS"] = "1"
# # # # os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# # # # os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# # # # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # # # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # # # import csv
# # # # import pickle
# # # # import re
# # # # import string
# # # # import warnings

# # # # import gradio as gr
# # # # import matplotlib.pyplot as plt
# # # # import nltk
# # # # import numpy as np
# # # # import pandas as pd
# # # # import tensorflow as tf

# # # # from nltk.corpus import stopwords
# # # # from nltk.stem import WordNetLemmatizer
# # # # from nltk.tokenize import word_tokenize
# # # # from tensorflow.keras.models import load_model
# # # # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # # # from thefuzz import process

# # # # warnings.filterwarnings("ignore")

# # # # # Limit TensorFlow CPU threads to reduce crashes and memory pressure.
# # # # tf.config.threading.set_intra_op_parallelism_threads(1)
# # # # tf.config.threading.set_inter_op_parallelism_threads(1)

# # # # # =========================================================
# # # # # Files and settings
# # # # # =========================================================

# # # # LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
# # # # TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
# # # # LABEL_ENCODER_PATH = "label_encoder.pkl"
# # # # REVIEWS_FILE_PATH = "Reviews.csv"

# # # # SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# # # # SUMMARY_WORD_THRESHOLD = 15
# # # # DEFAULT_MAX_LENGTH = 111

# # # # # DistilBART is loaded only when summarization is first requested.
# # # # summary_tokenizer = None
# # # # summarization_model = None


# # # # # =========================================================
# # # # # NLTK setup
# # # # # =========================================================

# # # # for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
# # # #     nltk.download(package, quiet=True)

# # # # stop_words = set(stopwords.words("english"))
# # # # stop_words = stop_words.difference({"not", "no", "nor", "never"})
# # # # lemmatizer = WordNetLemmatizer()


# # # # # =========================================================
# # # # # Text preprocessing
# # # # # Must match training preprocessing
# # # # # =========================================================

# # # # def clean_html_tags(text):
# # # #     text = re.sub(r"<[^>]*>", " ", str(text))
# # # #     text = re.sub(r"\s+", " ", text)
# # # #     return text.strip()


# # # # def normalize_negations(text):
# # # #     text = str(text).replace("’", "'")

# # # #     contractions = {
# # # #         "can't": "can not",
# # # #         "cannot": "can not",
# # # #         "won't": "will not",
# # # #         "don't": "do not",
# # # #         "doesn't": "does not",
# # # #         "didn't": "did not",
# # # #         "isn't": "is not",
# # # #         "aren't": "are not",
# # # #         "wasn't": "was not",
# # # #         "weren't": "were not",
# # # #         "hasn't": "has not",
# # # #         "haven't": "have not",
# # # #         "hadn't": "had not",
# # # #         "shouldn't": "should not",
# # # #         "wouldn't": "would not",
# # # #         "couldn't": "could not",
# # # #         "mustn't": "must not",
# # # #         "needn't": "need not",
# # # #         "mightn't": "might not",
# # # #         "shan't": "shall not",
# # # #     }

# # # #     for contraction, expanded_form in contractions.items():
# # # #         text = re.sub(
# # # #             rf"\b{re.escape(contraction)}\b",
# # # #             expanded_form,
# # # #             text,
# # # #             flags=re.IGNORECASE,
# # # #         )

# # # #     return text


# # # # def preprocess_text(text):
# # # #     text = normalize_negations(str(text).lower())
# # # #     text = re.sub(r"http\S+|www\S+", " ", text)
# # # #     text = re.sub(r"\S+@\S+", " ", text)
# # # #     text = text.translate(str.maketrans("", "", string.punctuation))
# # # #     text = re.sub(r"\d+", " ", text)
# # # #     text = re.sub(r"\s+", " ", text).strip()

# # # #     tokens = word_tokenize(text)

# # # #     cleaned_tokens = [
# # # #         lemmatizer.lemmatize(word)
# # # #         for word in tokens
# # # #         if word.isalpha() and word not in stop_words
# # # #     ]

# # # #     return " ".join(cleaned_tokens)


# # # # # =========================================================
# # # # # Load trained LSTM artifacts
# # # # # =========================================================

# # # # def require_file(path):
# # # #     if not os.path.exists(path):
# # # #         raise FileNotFoundError(
# # # #             f"Missing required file: {path}. "
# # # #             "Upload it to the root of the Hugging Face Space."
# # # #         )


# # # # for file_path in [
# # # #     LSTM_MODEL_PATH,
# # # #     TEXT_TOKENIZER_PATH,
# # # #     LABEL_ENCODER_PATH,
# # # # ]:
# # # #     require_file(file_path)

# # # # print("Loading trained LSTM model...")

# # # # sentiment_model = load_model(
# # # #     LSTM_MODEL_PATH,
# # # #     compile=False,
# # # # )

# # # # with open(TEXT_TOKENIZER_PATH, "rb") as file:
# # # #     text_tokenizer = pickle.load(file)

# # # # with open(LABEL_ENCODER_PATH, "rb") as file:
# # # #     label_encoder = pickle.load(file)

# # # # try:
# # # #     MAX_LENGTH = int(sentiment_model.input_shape[1])
# # # # except (TypeError, ValueError, IndexError):
# # # #     MAX_LENGTH = DEFAULT_MAX_LENGTH

# # # # print(f"LSTM model loaded. MAX_LENGTH={MAX_LENGTH}")


# # # # # =========================================================
# # # # # Lazy DistilBART loading
# # # # # =========================================================

# # # # def get_summarization_model():
# # # #     global summary_tokenizer
# # # #     global summarization_model

# # # #     if summary_tokenizer is None or summarization_model is None:
# # # #         print("Loading DistilBART on first summarization request...")

# # # #         # Imported lazily so PyTorch/Transformers do not load at startup.
# # # #         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # # #         summary_tokenizer = AutoTokenizer.from_pretrained(
# # # #             SUMMARIZATION_MODEL_NAME
# # # #         )

# # # #         summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
# # # #             SUMMARIZATION_MODEL_NAME,
# # # #             low_cpu_mem_usage=True,
# # # #         )

# # # #         summarization_model.eval()
# # # #         print("DistilBART loaded.")

# # # #     return summary_tokenizer, summarization_model


# # # # # =========================================================
# # # # # Optional Reviews.csv for search/dashboard
# # # # # =========================================================

# # # # def map_score_to_sentiment(score):
# # # #     if score in [1, 2]:
# # # #         return "negative"
# # # #     if score == 3:
# # # #         return "neutral"
# # # #     return "positive"


# # # # def read_csv_safely(file_path, max_valid_rows=100000):
# # # #     valid_rows = []
# # # #     bad_rows = []

# # # #     with open(
# # # #         file_path,
# # # #         "r",
# # # #         encoding="utf-8",
# # # #         errors="replace",
# # # #         newline="",
# # # #     ) as file:
# # # #         reader = csv.reader(file, delimiter=",", quotechar='"', strict=True)

# # # #         try:
# # # #             columns = next(reader)
# # # #         except StopIteration:
# # # #             raise ValueError("Reviews.csv is empty.")

# # # #         columns = [
# # # #             column.strip().replace("\ufeff", "")
# # # #             for column in columns
# # # #         ]

# # # #         expected_columns = len(columns)

# # # #         while len(valid_rows) < max_valid_rows:
# # # #             try:
# # # #                 row = next(reader)
# # # #             except StopIteration:
# # # #                 break
# # # #             except csv.Error as error:
# # # #                 bad_rows.append(
# # # #                     {"line": reader.line_num, "error": str(error)}
# # # #                 )
# # # #                 continue

# # # #             if len(row) != expected_columns:
# # # #                 bad_rows.append(
# # # #                     {
# # # #                         "line": reader.line_num,
# # # #                         "error": (
# # # #                             f"Expected {expected_columns} fields, "
# # # #                             f"found {len(row)}."
# # # #                         ),
# # # #                     }
# # # #                 )
# # # #                 continue

# # # #             valid_rows.append(row)

# # # #     return pd.DataFrame(valid_rows, columns=columns), bad_rows


# # # # def load_reviews_dataset():
# # # #     if not os.path.exists(REVIEWS_FILE_PATH):
# # # #         print("Reviews.csv not found. Search and dashboard remain unavailable.")
# # # #         return None

# # # #     try:
# # # #         dataset, bad_rows = read_csv_safely(REVIEWS_FILE_PATH)

# # # #         required_columns = {"Text", "Summary", "Score"}
# # # #         if not required_columns.issubset(dataset.columns):
# # # #             print("Reviews.csv is missing required columns.")
# # # #             return None

# # # #         dataset = dataset[["Text", "Summary", "Score"]].copy()
# # # #         dataset["Score"] = pd.to_numeric(dataset["Score"], errors="coerce")
# # # #         dataset = dataset.dropna(subset=["Text", "Summary", "Score"])
# # # #         dataset = dataset[dataset["Score"].isin([1, 2, 3, 4, 5])].copy()
# # # #         dataset["Score"] = dataset["Score"].astype(int)
# # # #         dataset["Sentiment"] = dataset["Score"].apply(map_score_to_sentiment)
# # # #         dataset["Cleaned_Text"] = dataset["Text"].apply(clean_html_tags)
# # # #         dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(preprocess_text)
# # # #         dataset = dataset[
# # # #             dataset["Processed_Text"].str.strip() != ""
# # # #         ].reset_index(drop=True)

# # # #         print(
# # # #             f"Reviews.csv loaded: {len(dataset)} valid rows, "
# # # #             f"{len(bad_rows)} corrupted rows skipped."
# # # #         )

# # # #         return dataset

# # # #     except Exception as error:
# # # #         print(f"Could not load Reviews.csv: {error}")
# # # #         return None


# # # # df = load_reviews_dataset()


# # # # # =========================================================
# # # # # Inference
# # # # # =========================================================

# # # # def predict_sentiment(raw_text):
# # # #     raw_text = str(raw_text).strip()

# # # #     if not raw_text:
# # # #         raise ValueError("Please enter a review.")

# # # #     processed_text = preprocess_text(clean_html_tags(raw_text))

# # # #     if not processed_text:
# # # #         raise ValueError("The text became empty after preprocessing.")

# # # #     sequence = text_tokenizer.texts_to_sequences([processed_text])

# # # #     padded_sequence = pad_sequences(
# # # #         sequence,
# # # #         maxlen=MAX_LENGTH,
# # # #         padding="post",
# # # #         truncating="post",
# # # #     )

# # # #     probabilities = sentiment_model.predict(
# # # #         padded_sequence,
# # # #         verbose=0,
# # # #     )[0]

# # # #     predicted_index = int(np.argmax(probabilities))
# # # #     sentiment = label_encoder.inverse_transform([predicted_index])[0]
# # # #     confidence = float(probabilities[predicted_index])

# # # #     all_probabilities = {
# # # #         class_name: float(probabilities[index])
# # # #         for index, class_name in enumerate(label_encoder.classes_)
# # # #     }

# # # #     return sentiment, confidence, all_probabilities


# # # # def summarize_review(raw_text):
# # # #     raw_text = str(raw_text).strip()

# # # #     if not raw_text:
# # # #         raise ValueError("Please enter a review.")

# # # #     word_count = len(raw_text.split())

# # # #     if word_count <= SUMMARY_WORD_THRESHOLD:
# # # #         return (
# # # #             "The text is too short to require summarization. "
# # # #             "/ النص قصير جداً ولا يحتاج إلى تلخيص."
# # # #         )

# # # #     tokenizer, model = get_summarization_model()

# # # #     inputs = tokenizer(
# # # #         raw_text,
# # # #         max_length=1024,
# # # #         return_tensors="pt",
# # # #         truncation=True,
# # # #     )

# # # #     summary_ids = model.generate(
# # # #         input_ids=inputs["input_ids"],
# # # #         attention_mask=inputs["attention_mask"],
# # # #         max_new_tokens=30,
# # # #         no_repeat_ngram_size=2,
# # # #         num_beams=2,
# # # #         early_stopping=True,
# # # #     )

# # # #     summary = tokenizer.decode(
# # # #         summary_ids[0],
# # # #         skip_special_tokens=True,
# # # #     ).strip()

# # # #     if not summary or len(summary.split()) >= word_count:
# # # #         return raw_text

# # # #     return summary


# # # # def analyze_review(raw_text):
# # # #     sentiment, confidence, probabilities = predict_sentiment(raw_text)
# # # #     summary = summarize_review(raw_text)

# # # #     return {
# # # #         "summary": summary,
# # # #         "sentiment": sentiment,
# # # #         "confidence": confidence,
# # # #         "probabilities": probabilities,
# # # #     }


# # # # # =========================================================
# # # # # Gradio functions
# # # # # =========================================================

# # # # def sentiment_emoji(sentiment):
# # # #     return {
# # # #         "positive": "😊",
# # # #         "neutral": "😐",
# # # #         "negative": "😢",
# # # #     }.get(sentiment.lower(), "🔎")


# # # # def web_interface_fn(review_text):
# # # #     if not review_text or not review_text.strip():
# # # #         return "Please enter a review.", "", {}, "", "", "", ""

# # # #     try:
# # # #         raw_tokens = word_tokenize(review_text.lower())

# # # #         normalized = normalize_negations(review_text.lower())
# # # #         no_punctuation_text = normalized.translate(
# # # #             str.maketrans("", "", string.punctuation)
# # # #         )
# # # #         no_punctuation_text = re.sub(r"\d+", " ", no_punctuation_text)
# # # #         no_punctuation_tokens = word_tokenize(no_punctuation_text)

# # # #         without_stop_words = [
# # # #             word
# # # #             for word in no_punctuation_tokens
# # # #             if word.isalpha() and word not in stop_words
# # # #         ]

# # # #         lemmatized = [
# # # #             lemmatizer.lemmatize(word)
# # # #             for word in without_stop_words
# # # #         ]

# # # #         result = analyze_review(review_text)

# # # #         sentiment_text = (
# # # #             f"{result['sentiment'].upper()} "
# # # #             f"{sentiment_emoji(result['sentiment'])}\n"
# # # #             f"Confidence: {result['confidence'] * 100:.2f}%"
# # # #         )

# # # #         labels = {
# # # #             name.capitalize(): probability
# # # #             for name, probability in result["probabilities"].items()
# # # #         }

# # # #         return (
# # # #             result["summary"],
# # # #             sentiment_text,
# # # #             labels,
# # # #             " | ".join(raw_tokens),
# # # #             " | ".join(no_punctuation_tokens),
# # # #             " | ".join(without_stop_words),
# # # #             " | ".join(lemmatized),
# # # #         )

# # # #     except Exception as error:
# # # #         return f"Error: {error}", "", {}, "", "", "", ""


# # # # search_history = []


# # # # def search_in_dataset_with_history(query):
# # # #     global search_history

# # # #     if df is None:
# # # #         return (
# # # #             "Reviews.csv is not available in this Space.",
# # # #             "\n".join(search_history),
# # # #         )

# # # #     if not query or not query.strip():
# # # #         return "Please enter a search term.", "\n".join(search_history)

# # # #     common_keywords = [
# # # #         "battery", "screen", "camera", "price", "performance",
# # # #         "design", "software", "quality", "money", "coffee",
# # # #         "food", "tea", "taste",
# # # #     ]

# # # #     match = process.extractOne(query.lower(), common_keywords)
# # # #     actual_query = (
# # # #         match[0]
# # # #         if match and match[1] > 80
# # # #         else query.lower().strip()
# # # #     )

# # # #     if query not in search_history:
# # # #         search_history.insert(0, query)
# # # #         search_history = search_history[:5]

# # # #     results = df[
# # # #         df["Text"].str.contains(
# # # #             actual_query,
# # # #             case=False,
# # # #             na=False,
# # # #             regex=False,
# # # #         )
# # # #     ]

# # # #     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"

# # # #     if results.empty:
# # # #         output += "No matching reviews were found."

# # # #     for _, row in results.head(5).iterrows():
# # # #         output += (
# # # #             f"📝 {str(row['Text'])[:180]}...\n"
# # # #             f"📊 {str(row['Sentiment']).upper()}\n\n"
# # # #         )

# # # #     return output, "\n".join(search_history)


# # # # def generate_dashboard(query):
# # # #     if df is None:
# # # #         fig, ax = plt.subplots(figsize=(7, 5))
# # # #         ax.text(
# # # #             0.5, 0.5,
# # # #             "Reviews.csv is not available.",
# # # #             ha="center",
# # # #             va="center",
# # # #             fontsize=13,
# # # #         )
# # # #         ax.axis("off")
# # # #         return fig

# # # #     if query and query.strip():
# # # #         filtered_df = df[
# # # #             df["Text"].str.contains(
# # # #                 query.strip(),
# # # #                 case=False,
# # # #                 na=False,
# # # #                 regex=False,
# # # #             )
# # # #         ]
# # # #         title = f"Sentiment Distribution for: {query.strip().capitalize()}"
# # # #     else:
# # # #         filtered_df = df
# # # #         title = "General Sentiment Distribution"

# # # #     if filtered_df.empty:
# # # #         fig, ax = plt.subplots(figsize=(7, 5))
# # # #         ax.text(
# # # #             0.5, 0.5,
# # # #             "No data found for this feature.",
# # # #             ha="center",
# # # #             va="center",
# # # #             fontsize=13,
# # # #         )
# # # #         ax.axis("off")
# # # #         return fig

# # # #     counts = (
# # # #         filtered_df["Sentiment"]
# # # #         .value_counts()
# # # #         .reindex(["positive", "neutral", "negative"], fill_value=0)
# # # #     )

# # # #     fig, ax = plt.subplots(figsize=(6, 6))
# # # #     counts.plot(
# # # #         kind="pie",
# # # #         autopct="%1.1f%%",
# # # #         startangle=140,
# # # #         ax=ax,
# # # #         textprops={"fontsize": 12, "weight": "bold"},
# # # #     )

# # # #     ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
# # # #     ax.set_ylabel("")
# # # #     plt.tight_layout()

# # # #     return fig


# # # # # =========================================================
# # # # # Gradio interface
# # # # # =========================================================

# # # # with gr.Blocks(
# # # #     title="LSTM Sentiment Analysis",
# # # #     theme=gr.themes.Soft(),
# # # # ) as app:

# # # #     gr.Markdown(
# # # #         """
# # # #         <div style="text-align:center;background:#1e1e2e;padding:20px;
# # # #                     border-radius:10px;margin-bottom:20px;">
# # # #             <h1 style="color:#ff79c6;">
# # # #                 💻 LSTM Sentiment Analysis & Summarization
# # # #             </h1>
# # # #             <p style="color:white;">
# # # #                 Sentiment is predicted using the trained LSTM model.
# # # #                 DistilBART is loaded only when a long review needs summarization.
# # # #             </p>
# # # #             <div style="display:flex;justify-content:space-around;
# # # #                         color:white;font-size:16px;">
# # # #                 <div>👤 SAIF KHATTAB<br>🆔 202312556</div>
# # # #                 <div>👤 YAZEED AMEEN<br>🆔 202310058</div>
# # # #             </div>
# # # #         </div>
# # # #         """
# # # #     )

# # # #     with gr.Tabs():

# # # #         with gr.Tab("Analyze Text 📝"):
# # # #             input_text = gr.Textbox(
# # # #                 lines=6,
# # # #                 label="Enter Review",
# # # #                 placeholder="Write an English food or product review...",
# # # #             )

# # # #             analyze_button = gr.Button("Analyze 🚀", variant="primary")

# # # #             with gr.Row():
# # # #                 summary_output = gr.Textbox(
# # # #                     label="Generated Summary",
# # # #                     lines=5,
# # # #                 )
# # # #                 sentiment_output = gr.Textbox(
# # # #                     label="Predicted Sentiment",
# # # #                     lines=5,
# # # #                 )

# # # #             probability_output = gr.Label(
# # # #                 label="Class Probabilities",
# # # #                 num_top_classes=3,
# # # #             )

# # # #             with gr.Accordion(
# # # #                 "Text Processing Stages (Transparency Mode)",
# # # #                 open=False,
# # # #             ):
# # # #                 raw_tokens_output = gr.Textbox(label="1. Raw Tokens")
# # # #                 no_punctuation_output = gr.Textbox(label="2. No Punctuation")
# # # #                 no_stop_words_output = gr.Textbox(label="3. No Stop Words")
# # # #                 lemmatized_output = gr.Textbox(label="4. Lemmatized")

# # # #             analyze_button.click(
# # # #                 fn=web_interface_fn,
# # # #                 inputs=input_text,
# # # #                 outputs=[
# # # #                     summary_output,
# # # #                     sentiment_output,
# # # #                     probability_output,
# # # #                     raw_tokens_output,
# # # #                     no_punctuation_output,
# # # #                     no_stop_words_output,
# # # #                     lemmatized_output,
# # # #                 ],
# # # #             )

# # # #         with gr.Tab("Search Dataset 🔍"):
# # # #             with gr.Row():
# # # #                 with gr.Column():
# # # #                     query_input = gr.Textbox(label="Search Query")
# # # #                     search_button = gr.Button("Search Data 🔎")

# # # #                 history_output = gr.Textbox(
# # # #                     label="Last 5 Searches",
# # # #                     interactive=False,
# # # #                 )

# # # #             results_output = gr.Textbox(label="Results", lines=10)

# # # #             search_button.click(
# # # #                 fn=search_in_dataset_with_history,
# # # #                 inputs=query_input,
# # # #                 outputs=[results_output, history_output],
# # # #             )

# # # #         with gr.Tab("Analytics Dashboard 📊"):
# # # #             dashboard_query = gr.Textbox(
# # # #                 label="Filter statistics by feature",
# # # #                 placeholder="Examples: coffee, food, tea, quality",
# # # #             )

# # # #             dashboard_button = gr.Button("Show Chart 🚀")
# # # #             plot_output = gr.Plot(label="Sentiment Distribution")

# # # #             dashboard_button.click(
# # # #                 fn=generate_dashboard,
# # # #                 inputs=dashboard_query,
# # # #                 outputs=plot_output,
# # # #             )


# # # # if __name__ == "__main__":
# # # #     app.launch()


# # # import os

# # # # Must be set before importing TensorFlow or PyTorch.
# # # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # # os.environ["OMP_NUM_THREADS"] = "1"
# # # os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# # # os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# # # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # # import csv
# # # import json
# # # import pickle
# # # import re
# # # import string
# # # import warnings
# # # import zipfile

# # # import gradio as gr
# # # import matplotlib.pyplot as plt
# # # import nltk
# # # import numpy as np
# # # import pandas as pd
# # # import tensorflow as tf

# # # from nltk.corpus import stopwords
# # # from nltk.stem import WordNetLemmatizer
# # # from nltk.tokenize import word_tokenize
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # # from thefuzz import process

# # # warnings.filterwarnings("ignore")

# # # # Limit TensorFlow CPU threads to reduce crashes and memory pressure.
# # # tf.config.threading.set_intra_op_parallelism_threads(1)
# # # tf.config.threading.set_inter_op_parallelism_threads(1)

# # # # =========================================================
# # # # Files and settings
# # # # =========================================================

# # # LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
# # # TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
# # # LABEL_ENCODER_PATH = "label_encoder.pkl"
# # # REVIEWS_FILE_PATH = "Reviews.csv"

# # # SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# # # SUMMARY_WORD_THRESHOLD = 15
# # # DEFAULT_MAX_LENGTH = 111

# # # # DistilBART is loaded only when summarization is first requested.
# # # summary_tokenizer = None
# # # summarization_model = None


# # # # =========================================================
# # # # NLTK setup
# # # # =========================================================

# # # for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
# # #     nltk.download(package, quiet=True)

# # # stop_words = set(stopwords.words("english"))
# # # stop_words = stop_words.difference({"not", "no", "nor", "never"})
# # # lemmatizer = WordNetLemmatizer()


# # # # =========================================================
# # # # Text preprocessing
# # # # Must match training preprocessing
# # # # =========================================================

# # # def clean_html_tags(text):
# # #     text = re.sub(r"<[^>]*>", " ", str(text))
# # #     text = re.sub(r"\s+", " ", text)
# # #     return text.strip()


# # # def normalize_negations(text):
# # #     text = str(text).replace("’", "'")

# # #     contractions = {
# # #         "can't": "can not",
# # #         "cannot": "can not",
# # #         "won't": "will not",
# # #         "don't": "do not",
# # #         "doesn't": "does not",
# # #         "didn't": "did not",
# # #         "isn't": "is not",
# # #         "aren't": "are not",
# # #         "wasn't": "was not",
# # #         "weren't": "were not",
# # #         "hasn't": "has not",
# # #         "haven't": "have not",
# # #         "hadn't": "had not",
# # #         "shouldn't": "should not",
# # #         "wouldn't": "would not",
# # #         "couldn't": "could not",
# # #         "mustn't": "must not",
# # #         "needn't": "need not",
# # #         "mightn't": "might not",
# # #         "shan't": "shall not",
# # #     }

# # #     for contraction, expanded_form in contractions.items():
# # #         text = re.sub(
# # #             rf"\b{re.escape(contraction)}\b",
# # #             expanded_form,
# # #             text,
# # #             flags=re.IGNORECASE,
# # #         )

# # #     return text


# # # def preprocess_text(text):
# # #     text = normalize_negations(str(text).lower())
# # #     text = re.sub(r"http\S+|www\S+", " ", text)
# # #     text = re.sub(r"\S+@\S+", " ", text)
# # #     text = text.translate(str.maketrans("", "", string.punctuation))
# # #     text = re.sub(r"\d+", " ", text)
# # #     text = re.sub(r"\s+", " ", text).strip()

# # #     tokens = word_tokenize(text)

# # #     cleaned_tokens = [
# # #         lemmatizer.lemmatize(word)
# # #         for word in tokens
# # #         if word.isalpha() and word not in stop_words
# # #     ]

# # #     return " ".join(cleaned_tokens)


# # # # =========================================================
# # # # Load trained LSTM artifacts
# # # # =========================================================

# # # def require_file(path):
# # #     if not os.path.exists(path):
# # #         raise FileNotFoundError(
# # #             f"Missing required file: {path}. "
# # #             "Upload it to the root of the Hugging Face Space."
# # #         )


# # # for file_path in [
# # #     LSTM_MODEL_PATH,
# # #     TEXT_TOKENIZER_PATH,
# # #     LABEL_ENCODER_PATH,
# # # ]:
# # #     require_file(file_path)

# # # def remove_unsupported_keras_config(value):
# # #     """
# # #     Remove configuration fields written by a newer Keras version but not
# # #     recognized by the Keras version bundled with TensorFlow in this Space.
# # #     """
# # #     removed_count = 0

# # #     if isinstance(value, dict):
# # #         cleaned = {}

# # #         for key, item in value.items():
# # #             if key == "quantization_config" and item is None:
# # #                 removed_count += 1
# # #                 continue

# # #             cleaned_item, item_removed = remove_unsupported_keras_config(item)
# # #             cleaned[key] = cleaned_item
# # #             removed_count += item_removed

# # #         return cleaned, removed_count

# # #     if isinstance(value, list):
# # #         cleaned_list = []

# # #         for item in value:
# # #             cleaned_item, item_removed = remove_unsupported_keras_config(item)
# # #             cleaned_list.append(cleaned_item)
# # #             removed_count += item_removed

# # #         return cleaned_list, removed_count

# # #     return value, removed_count


# # # def prepare_compatible_keras_model(model_path):
# # #     """
# # #     A .keras model is a ZIP archive. This function copies the archive to /tmp
# # #     and removes only unsupported null quantization_config fields from
# # #     config.json. The original uploaded model is never modified.
# # #     """
# # #     if not zipfile.is_zipfile(model_path):
# # #         return model_path

# # #     patched_path = "/tmp/lstm_sentiment_model_compatible.keras"

# # #     with zipfile.ZipFile(model_path, "r") as source_archive:
# # #         if "config.json" not in source_archive.namelist():
# # #             return model_path

# # #         model_config = json.loads(
# # #             source_archive.read("config.json").decode("utf-8")
# # #         )

# # #         cleaned_config, removed_count = remove_unsupported_keras_config(
# # #             model_config
# # #         )

# # #         if removed_count == 0:
# # #             return model_path

# # #         with zipfile.ZipFile(
# # #             patched_path,
# # #             "w",
# # #             compression=zipfile.ZIP_DEFLATED,
# # #         ) as target_archive:
# # #             for archive_item in source_archive.infolist():
# # #                 file_data = source_archive.read(archive_item.filename)

# # #                 if archive_item.filename == "config.json":
# # #                     file_data = json.dumps(
# # #                         cleaned_config,
# # #                         ensure_ascii=False,
# # #                     ).encode("utf-8")

# # #                 target_archive.writestr(archive_item, file_data)

# # #     print(
# # #         "Prepared a Keras-compatible model copy. "
# # #         f"Removed {removed_count} unsupported config fields."
# # #     )

# # #     return patched_path


# # # print("Loading trained LSTM model...")

# # # compatible_model_path = prepare_compatible_keras_model(
# # #     LSTM_MODEL_PATH
# # # )

# # # sentiment_model = load_model(
# # #     compatible_model_path,
# # #     compile=False,
# # # )

# # # with open(TEXT_TOKENIZER_PATH, "rb") as file:
# # #     text_tokenizer = pickle.load(file)

# # # with open(LABEL_ENCODER_PATH, "rb") as file:
# # #     label_encoder = pickle.load(file)

# # # try:
# # #     MAX_LENGTH = int(sentiment_model.input_shape[1])
# # # except (TypeError, ValueError, IndexError):
# # #     MAX_LENGTH = DEFAULT_MAX_LENGTH

# # # print(f"LSTM model loaded. MAX_LENGTH={MAX_LENGTH}")


# # # # =========================================================
# # # # Lazy DistilBART loading
# # # # =========================================================

# # # def get_summarization_model():
# # #     global summary_tokenizer
# # #     global summarization_model

# # #     if summary_tokenizer is None or summarization_model is None:
# # #         print("Loading DistilBART on first summarization request...")

# # #         # Imported lazily so PyTorch/Transformers do not load at startup.
# # #         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # #         summary_tokenizer = AutoTokenizer.from_pretrained(
# # #             SUMMARIZATION_MODEL_NAME
# # #         )

# # #         summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
# # #             SUMMARIZATION_MODEL_NAME,
# # #             low_cpu_mem_usage=True,
# # #         )

# # #         summarization_model.eval()
# # #         print("DistilBART loaded.")

# # #     return summary_tokenizer, summarization_model


# # # # =========================================================
# # # # Optional Reviews.csv for search/dashboard
# # # # =========================================================

# # # def map_score_to_sentiment(score):
# # #     if score in [1, 2]:
# # #         return "negative"
# # #     if score == 3:
# # #         return "neutral"
# # #     return "positive"


# # # def read_csv_safely(file_path, max_valid_rows=100000):
# # #     valid_rows = []
# # #     bad_rows = []

# # #     with open(
# # #         file_path,
# # #         "r",
# # #         encoding="utf-8",
# # #         errors="replace",
# # #         newline="",
# # #     ) as file:
# # #         reader = csv.reader(file, delimiter=",", quotechar='"', strict=True)

# # #         try:
# # #             columns = next(reader)
# # #         except StopIteration:
# # #             raise ValueError("Reviews.csv is empty.")

# # #         columns = [
# # #             column.strip().replace("\ufeff", "")
# # #             for column in columns
# # #         ]

# # #         expected_columns = len(columns)

# # #         while len(valid_rows) < max_valid_rows:
# # #             try:
# # #                 row = next(reader)
# # #             except StopIteration:
# # #                 break
# # #             except csv.Error as error:
# # #                 bad_rows.append(
# # #                     {"line": reader.line_num, "error": str(error)}
# # #                 )
# # #                 continue

# # #             if len(row) != expected_columns:
# # #                 bad_rows.append(
# # #                     {
# # #                         "line": reader.line_num,
# # #                         "error": (
# # #                             f"Expected {expected_columns} fields, "
# # #                             f"found {len(row)}."
# # #                         ),
# # #                     }
# # #                 )
# # #                 continue

# # #             valid_rows.append(row)

# # #     return pd.DataFrame(valid_rows, columns=columns), bad_rows


# # # def load_reviews_dataset():
# # #     if not os.path.exists(REVIEWS_FILE_PATH):
# # #         print("Reviews.csv not found. Search and dashboard remain unavailable.")
# # #         return None

# # #     try:
# # #         dataset, bad_rows = read_csv_safely(REVIEWS_FILE_PATH)

# # #         required_columns = {"Text", "Summary", "Score"}
# # #         if not required_columns.issubset(dataset.columns):
# # #             print("Reviews.csv is missing required columns.")
# # #             return None

# # #         dataset = dataset[["Text", "Summary", "Score"]].copy()
# # #         dataset["Score"] = pd.to_numeric(dataset["Score"], errors="coerce")
# # #         dataset = dataset.dropna(subset=["Text", "Summary", "Score"])
# # #         dataset = dataset[dataset["Score"].isin([1, 2, 3, 4, 5])].copy()
# # #         dataset["Score"] = dataset["Score"].astype(int)
# # #         dataset["Sentiment"] = dataset["Score"].apply(map_score_to_sentiment)
# # #         dataset["Cleaned_Text"] = dataset["Text"].apply(clean_html_tags)
# # #         dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(preprocess_text)
# # #         dataset = dataset[
# # #             dataset["Processed_Text"].str.strip() != ""
# # #         ].reset_index(drop=True)

# # #         print(
# # #             f"Reviews.csv loaded: {len(dataset)} valid rows, "
# # #             f"{len(bad_rows)} corrupted rows skipped."
# # #         )

# # #         return dataset

# # #     except Exception as error:
# # #         print(f"Could not load Reviews.csv: {error}")
# # #         return None


# # # df = load_reviews_dataset()


# # # # =========================================================
# # # # Inference
# # # # =========================================================

# # # def predict_sentiment(raw_text):
# # #     raw_text = str(raw_text).strip()

# # #     if not raw_text:
# # #         raise ValueError("Please enter a review.")

# # #     processed_text = preprocess_text(clean_html_tags(raw_text))

# # #     if not processed_text:
# # #         raise ValueError("The text became empty after preprocessing.")

# # #     sequence = text_tokenizer.texts_to_sequences([processed_text])

# # #     padded_sequence = pad_sequences(
# # #         sequence,
# # #         maxlen=MAX_LENGTH,
# # #         padding="post",
# # #         truncating="post",
# # #     )

# # #     probabilities = sentiment_model.predict(
# # #         padded_sequence,
# # #         verbose=0,
# # #     )[0]

# # #     predicted_index = int(np.argmax(probabilities))
# # #     sentiment = label_encoder.inverse_transform([predicted_index])[0]
# # #     confidence = float(probabilities[predicted_index])

# # #     all_probabilities = {
# # #         class_name: float(probabilities[index])
# # #         for index, class_name in enumerate(label_encoder.classes_)
# # #     }

# # #     return sentiment, confidence, all_probabilities


# # # def summarize_review(raw_text):
# # #     raw_text = str(raw_text).strip()

# # #     if not raw_text:
# # #         raise ValueError("Please enter a review.")

# # #     word_count = len(raw_text.split())

# # #     if word_count <= SUMMARY_WORD_THRESHOLD:
# # #         return (
# # #             "The text is too short to require summarization. "
# # #             "/ النص قصير جداً ولا يحتاج إلى تلخيص."
# # #         )

# # #     tokenizer, model = get_summarization_model()

# # #     inputs = tokenizer(
# # #         raw_text,
# # #         max_length=1024,
# # #         return_tensors="pt",
# # #         truncation=True,
# # #     )

# # #     summary_ids = model.generate(
# # #         input_ids=inputs["input_ids"],
# # #         attention_mask=inputs["attention_mask"],
# # #         max_new_tokens=30,
# # #         no_repeat_ngram_size=2,
# # #         num_beams=2,
# # #         early_stopping=True,
# # #     )

# # #     summary = tokenizer.decode(
# # #         summary_ids[0],
# # #         skip_special_tokens=True,
# # #     ).strip()

# # #     if not summary or len(summary.split()) >= word_count:
# # #         return raw_text

# # #     return summary


# # # def analyze_review(raw_text):
# # #     sentiment, confidence, probabilities = predict_sentiment(raw_text)
# # #     summary = summarize_review(raw_text)

# # #     return {
# # #         "summary": summary,
# # #         "sentiment": sentiment,
# # #         "confidence": confidence,
# # #         "probabilities": probabilities,
# # #     }


# # # # =========================================================
# # # # Gradio functions
# # # # =========================================================

# # # def sentiment_emoji(sentiment):
# # #     return {
# # #         "positive": "😊",
# # #         "neutral": "😐",
# # #         "negative": "😢",
# # #     }.get(sentiment.lower(), "🔎")


# # # def web_interface_fn(review_text):
# # #     if not review_text or not review_text.strip():
# # #         return "Please enter a review.", "", {}, "", "", "", ""

# # #     try:
# # #         raw_tokens = word_tokenize(review_text.lower())

# # #         normalized = normalize_negations(review_text.lower())
# # #         no_punctuation_text = normalized.translate(
# # #             str.maketrans("", "", string.punctuation)
# # #         )
# # #         no_punctuation_text = re.sub(r"\d+", " ", no_punctuation_text)
# # #         no_punctuation_tokens = word_tokenize(no_punctuation_text)

# # #         without_stop_words = [
# # #             word
# # #             for word in no_punctuation_tokens
# # #             if word.isalpha() and word not in stop_words
# # #         ]

# # #         lemmatized = [
# # #             lemmatizer.lemmatize(word)
# # #             for word in without_stop_words
# # #         ]

# # #         result = analyze_review(review_text)

# # #         sentiment_text = (
# # #             f"{result['sentiment'].upper()} "
# # #             f"{sentiment_emoji(result['sentiment'])}\n"
# # #             f"Confidence: {result['confidence'] * 100:.2f}%"
# # #         )

# # #         labels = {
# # #             name.capitalize(): probability
# # #             for name, probability in result["probabilities"].items()
# # #         }

# # #         return (
# # #             result["summary"],
# # #             sentiment_text,
# # #             labels,
# # #             " | ".join(raw_tokens),
# # #             " | ".join(no_punctuation_tokens),
# # #             " | ".join(without_stop_words),
# # #             " | ".join(lemmatized),
# # #         )

# # #     except Exception as error:
# # #         return f"Error: {error}", "", {}, "", "", "", ""


# # # search_history = []


# # # def search_in_dataset_with_history(query):
# # #     global search_history

# # #     if df is None:
# # #         return (
# # #             "Reviews.csv is not available in this Space.",
# # #             "\n".join(search_history),
# # #         )

# # #     if not query or not query.strip():
# # #         return "Please enter a search term.", "\n".join(search_history)

# # #     common_keywords = [
# # #         "battery", "screen", "camera", "price", "performance",
# # #         "design", "software", "quality", "money", "coffee",
# # #         "food", "tea", "taste",
# # #     ]

# # #     match = process.extractOne(query.lower(), common_keywords)
# # #     actual_query = (
# # #         match[0]
# # #         if match and match[1] > 80
# # #         else query.lower().strip()
# # #     )

# # #     if query not in search_history:
# # #         search_history.insert(0, query)
# # #         search_history = search_history[:5]

# # #     results = df[
# # #         df["Text"].str.contains(
# # #             actual_query,
# # #             case=False,
# # #             na=False,
# # #             regex=False,
# # #         )
# # #     ]

# # #     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"

# # #     if results.empty:
# # #         output += "No matching reviews were found."

# # #     for _, row in results.head(5).iterrows():
# # #         output += (
# # #             f"📝 {str(row['Text'])[:180]}...\n"
# # #             f"📊 {str(row['Sentiment']).upper()}\n\n"
# # #         )

# # #     return output, "\n".join(search_history)


# # # def generate_dashboard(query):
# # #     if df is None:
# # #         fig, ax = plt.subplots(figsize=(7, 5))
# # #         ax.text(
# # #             0.5, 0.5,
# # #             "Reviews.csv is not available.",
# # #             ha="center",
# # #             va="center",
# # #             fontsize=13,
# # #         )
# # #         ax.axis("off")
# # #         return fig

# # #     if query and query.strip():
# # #         filtered_df = df[
# # #             df["Text"].str.contains(
# # #                 query.strip(),
# # #                 case=False,
# # #                 na=False,
# # #                 regex=False,
# # #             )
# # #         ]
# # #         title = f"Sentiment Distribution for: {query.strip().capitalize()}"
# # #     else:
# # #         filtered_df = df
# # #         title = "General Sentiment Distribution"

# # #     if filtered_df.empty:
# # #         fig, ax = plt.subplots(figsize=(7, 5))
# # #         ax.text(
# # #             0.5, 0.5,
# # #             "No data found for this feature.",
# # #             ha="center",
# # #             va="center",
# # #             fontsize=13,
# # #         )
# # #         ax.axis("off")
# # #         return fig

# # #     counts = (
# # #         filtered_df["Sentiment"]
# # #         .value_counts()
# # #         .reindex(["positive", "neutral", "negative"], fill_value=0)
# # #     )

# # #     fig, ax = plt.subplots(figsize=(6, 6))
# # #     counts.plot(
# # #         kind="pie",
# # #         autopct="%1.1f%%",
# # #         startangle=140,
# # #         ax=ax,
# # #         textprops={"fontsize": 12, "weight": "bold"},
# # #     )

# # #     ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
# # #     ax.set_ylabel("")
# # #     plt.tight_layout()

# # #     return fig


# # # # =========================================================
# # # # Gradio interface
# # # # =========================================================

# # # CUSTOM_CSS = """
# # # :root {
# # #     --app-bg: #f4f7fb;
# # #     --surface: rgba(255, 255, 255, 0.94);
# # #     --surface-strong: #ffffff;
# # #     --text-main: #172033;
# # #     --text-muted: #64748b;
# # #     --primary: #4f46e5;
# # #     --primary-dark: #3730a3;
# # #     --accent: #7c3aed;
# # #     --border: #e2e8f0;
# # #     --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
# # #     --shadow-soft: 0 8px 24px rgba(15, 23, 42, 0.07);
# # # }

# # # .gradio-container {
# # #     background:
# # #         radial-gradient(circle at 10% 0%, rgba(79, 70, 229, 0.12), transparent 30%),
# # #         radial-gradient(circle at 95% 10%, rgba(124, 58, 237, 0.10), transparent 28%),
# # #         var(--app-bg);
# # #     color: var(--text-main);
# # #     min-height: 100vh;
# # # }

# # # .app-shell {
# # #     max-width: 1180px;
# # #     margin: 0 auto;
# # # }

# # # .hero {
# # #     position: relative;
# # #     overflow: hidden;
# # #     border-radius: 26px;
# # #     padding: 34px;
# # #     margin: 8px 0 22px;
# # #     color: white;
# # #     background: linear-gradient(135deg, #312e81 0%, #4f46e5 52%, #7c3aed 100%);
# # #     box-shadow: 0 24px 60px rgba(79, 70, 229, 0.28);
# # # }

# # # .hero::before,
# # # .hero::after {
# # #     content: "";
# # #     position: absolute;
# # #     border-radius: 999px;
# # #     background: rgba(255, 255, 255, 0.10);
# # # }

# # # .hero::before {
# # #     width: 220px;
# # #     height: 220px;
# # #     top: -90px;
# # #     right: -60px;
# # # }

# # # .hero::after {
# # #     width: 150px;
# # #     height: 150px;
# # #     bottom: -85px;
# # #     left: 8%;
# # # }

# # # .hero-content {
# # #     position: relative;
# # #     z-index: 1;
# # # }

# # # .hero-badge {
# # #     display: inline-flex;
# # #     align-items: center;
# # #     gap: 8px;
# # #     padding: 7px 12px;
# # #     border: 1px solid rgba(255, 255, 255, 0.25);
# # #     border-radius: 999px;
# # #     background: rgba(255, 255, 255, 0.12);
# # #     font-size: 13px;
# # #     font-weight: 700;
# # #     letter-spacing: 0.02em;
# # # }

# # # .hero h1 {
# # #     margin: 15px 0 10px;
# # #     font-size: clamp(30px, 4vw, 48px);
# # #     line-height: 1.08;
# # #     color: white;
# # # }

# # # .hero p {
# # #     max-width: 780px;
# # #     margin: 0;
# # #     color: rgba(255, 255, 255, 0.86);
# # #     font-size: 16px;
# # #     line-height: 1.7;
# # # }

# # # .team-grid {
# # #     position: relative;
# # #     z-index: 1;
# # #     display: grid;
# # #     grid-template-columns: repeat(3, minmax(0, 1fr));
# # #     gap: 12px;
# # #     margin-top: 24px;
# # # }

# # # .team-card {
# # #     min-width: 0;
# # #     padding: 14px 16px;
# # #     border: 1px solid rgba(255, 255, 255, 0.18);
# # #     border-radius: 16px;
# # #     background: rgba(255, 255, 255, 0.11);
# # #     backdrop-filter: blur(8px);
# # #     font-weight: 700;
# # #     text-align: center;
# # #     white-space: normal;
# # #     overflow-wrap: anywhere;
# # # }

# # # .team-card span {
# # #     display: block;
# # #     margin-bottom: 5px;
# # #     color: rgba(255, 255, 255, 0.70);
# # #     font-size: 11px;
# # #     font-weight: 700;
# # #     letter-spacing: 0.10em;
# # #     text-transform: uppercase;
# # # }

# # # .feature-strip {
# # #     display: grid;
# # #     grid-template-columns: repeat(3, minmax(0, 1fr));
# # #     gap: 12px;
# # #     margin-bottom: 20px;
# # # }

# # # .feature-card {
# # #     padding: 16px 18px;
# # #     border: 1px solid var(--border);
# # #     border-radius: 18px;
# # #     background: var(--surface);
# # #     box-shadow: var(--shadow-soft);
# # # }

# # # .feature-card strong {
# # #     display: block;
# # #     margin-bottom: 4px;
# # #     color: var(--text-main);
# # # }

# # # .feature-card small {
# # #     color: var(--text-muted);
# # #     line-height: 1.5;
# # # }

# # # .app-tabs {
# # #     border: 1px solid var(--border) !important;
# # #     border-radius: 22px !important;
# # #     background: var(--surface) !important;
# # #     box-shadow: var(--shadow) !important;
# # #     padding: 8px !important;
# # # }

# # # .section-card {
# # #     border: 1px solid var(--border) !important;
# # #     border-radius: 18px !important;
# # #     background: var(--surface-strong) !important;
# # #     box-shadow: var(--shadow-soft) !important;
# # #     padding: 6px !important;
# # # }

# # # .primary-action button,
# # # .primary-action {
# # #     border: none !important;
# # #     border-radius: 14px !important;
# # #     background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
# # #     color: white !important;
# # #     font-weight: 800 !important;
# # #     box-shadow: 0 10px 22px rgba(79, 70, 229, 0.22) !important;
# # #     transition: transform 0.18s ease, box-shadow 0.18s ease !important;
# # # }

# # # .primary-action:hover {
# # #     transform: translateY(-1px);
# # #     box-shadow: 0 14px 28px rgba(79, 70, 229, 0.28) !important;
# # # }

# # # .secondary-action button,
# # # .secondary-action {
# # #     border-radius: 14px !important;
# # #     border: 1px solid #c7d2fe !important;
# # #     background: #eef2ff !important;
# # #     color: var(--primary-dark) !important;
# # #     font-weight: 800 !important;
# # # }

# # # textarea,
# # # input {
# # #     border-radius: 14px !important;
# # # }

# # # footer {
# # #     display: none !important;
# # # }

# # # .footer-note {
# # #     margin: 18px 0 4px;
# # #     padding: 14px;
# # #     text-align: center;
# # #     color: var(--text-muted);
# # #     font-size: 13px;
# # # }

# # # @media (max-width: 760px) {
# # #     .hero {
# # #         padding: 24px 20px;
# # #         border-radius: 20px;
# # #     }

# # #     .team-grid,
# # #     .feature-strip {
# # #         grid-template-columns: 1fr;
# # #     }

# # #     .team-card {
# # #         text-align: left;
# # #     }
# # # }
# # # """

# # # with gr.Blocks(
# # #     title="LSTM Sentiment Analysis",
# # #     theme=gr.themes.Soft(),
# # #     css=CUSTOM_CSS,
# # # ) as app:

# # #     with gr.Column(elem_classes=["app-shell"]):

# # #         gr.HTML(
# # #             """
# # #             <section class="hero">
# # #                 <div class="hero-content">
# # #                     <div class="hero-badge">✨ AI-Powered NLP Project</div>
# # #                     <h1>Sentiment Analysis & Smart Summarization</h1>
# # #                     <p>
# # #                         Analyze English product and food reviews using a trained
# # #                         LSTM model, generate concise summaries with DistilBART,
# # #                         and explore dataset insights through a clean interactive dashboard.
# # #                     </p>
# # #                 </div>

# # #                 <div class="team-grid">
# # #                     <div class="team-card">
# # #                         <span>Team Member</span>
# # #                         SAIF KHATTAB
# # #                     </div>
# # #                     <div class="team-card">
# # #                         <span>Team Member</span>
# # #                         YAZEED AMEEN
# # #                     </div>
# # #                     <div class="team-card">
# # #                         <span>Team Member</span>
# # #                         ASEEL MANNOUN
# # #                     </div>
# # #                 </div>
# # #             </section>
# # #             """
# # #         )

# # #         gr.HTML(
# # #             """
# # #             <div class="feature-strip">
# # #                 <div class="feature-card">
# # #                     <strong>🧠 LSTM Classification</strong>
# # #                     <small>Predicts positive, neutral, or negative sentiment.</small>
# # #                 </div>
# # #                 <div class="feature-card">
# # #                     <strong>📝 Smart Summarization</strong>
# # #                     <small>Creates a concise summary for longer reviews.</small>
# # #                 </div>
# # #                 <div class="feature-card">
# # #                     <strong>📊 Dataset Insights</strong>
# # #                     <small>Search reviews and visualize sentiment distribution.</small>
# # #                 </div>
# # #             </div>
# # #             """
# # #         )

# # #         with gr.Tabs(elem_classes=["app-tabs"]):

# # #             with gr.Tab("📝 Analyze Review"):
# # #                 with gr.Column(elem_classes=["section-card"]):
# # #                     gr.Markdown(
# # #                         "### Analyze a Review\n"
# # #                         "Enter an English review to generate its summary, sentiment, "
# # #                         "confidence score, and processing stages."
# # #                     )

# # #                     input_text = gr.Textbox(
# # #                         lines=7,
# # #                         label="Review Text",
# # #                         placeholder="Write or paste an English food or product review...",
# # #                     )

# # #                     analyze_button = gr.Button(
# # #                         "Analyze Review",
# # #                         variant="primary",
# # #                         elem_classes=["primary-action"],
# # #                     )

# # #                     with gr.Row(equal_height=True):
# # #                         summary_output = gr.Textbox(
# # #                             label="Generated Summary",
# # #                             lines=5,
# # #                         )
# # #                         sentiment_output = gr.Textbox(
# # #                             label="Predicted Sentiment",
# # #                             lines=5,
# # #                         )

# # #                     probability_output = gr.Label(
# # #                         label="Class Probabilities",
# # #                         num_top_classes=3,
# # #                     )

# # #                     with gr.Accordion(
# # #                         "🔍 Text Processing Stages",
# # #                         open=False,
# # #                     ):
# # #                         gr.Markdown(
# # #                             "These fields show how the review is transformed "
# # #                             "before it reaches the LSTM model."
# # #                         )
# # #                         raw_tokens_output = gr.Textbox(label="1. Raw Tokens")
# # #                         no_punctuation_output = gr.Textbox(label="2. No Punctuation")
# # #                         no_stop_words_output = gr.Textbox(label="3. No Stop Words")
# # #                         lemmatized_output = gr.Textbox(label="4. Lemmatized")

# # #                     analyze_button.click(
# # #                         fn=web_interface_fn,
# # #                         inputs=input_text,
# # #                         outputs=[
# # #                             summary_output,
# # #                             sentiment_output,
# # #                             probability_output,
# # #                             raw_tokens_output,
# # #                             no_punctuation_output,
# # #                             no_stop_words_output,
# # #                             lemmatized_output,
# # #                         ],
# # #                     )

# # #             with gr.Tab("🔎 Search Dataset"):
# # #                 with gr.Column(elem_classes=["section-card"]):
# # #                     gr.Markdown(
# # #                         "### Search Reviews\n"
# # #                         "Search the dataset by a word or feature and view the closest results."
# # #                     )

# # #                     with gr.Row(equal_height=True):
# # #                         with gr.Column(scale=2):
# # #                             query_input = gr.Textbox(
# # #                                 label="Search Query",
# # #                                 placeholder="Examples: coffee, taste, quality, price",
# # #                             )
# # #                             search_button = gr.Button(
# # #                                 "Search Dataset",
# # #                                 elem_classes=["secondary-action"],
# # #                             )

# # #                         with gr.Column(scale=1):
# # #                             history_output = gr.Textbox(
# # #                                 label="Recent Searches",
# # #                                 lines=5,
# # #                                 interactive=False,
# # #                             )

# # #                     results_output = gr.Textbox(
# # #                         label="Matching Reviews",
# # #                         lines=12,
# # #                     )

# # #                     search_button.click(
# # #                         fn=search_in_dataset_with_history,
# # #                         inputs=query_input,
# # #                         outputs=[results_output, history_output],
# # #                     )

# # #             with gr.Tab("📊 Analytics Dashboard"):
# # #                 with gr.Column(elem_classes=["section-card"]):
# # #                     gr.Markdown(
# # #                         "### Sentiment Dashboard\n"
# # #                         "Filter the dataset by a feature to view its sentiment distribution."
# # #                     )

# # #                     with gr.Row(equal_height=True):
# # #                         dashboard_query = gr.Textbox(
# # #                             label="Feature Filter",
# # #                             placeholder="Examples: coffee, food, tea, quality",
# # #                             scale=3,
# # #                         )
# # #                         dashboard_button = gr.Button(
# # #                             "Generate Chart",
# # #                             elem_classes=["primary-action"],
# # #                             scale=1,
# # #                         )

# # #                     plot_output = gr.Plot(label="Sentiment Distribution")

# # #                     dashboard_button.click(
# # #                         fn=generate_dashboard,
# # #                         inputs=dashboard_query,
# # #                         outputs=plot_output,
# # #                     )

# # #         gr.HTML(
# # #             """
# # #             <div class="footer-note">
# # #                 Built with TensorFlow, Keras, Transformers, and Gradio.
# # #             </div>
# # #             """
# # #         )


# # # if __name__ == "__main__":
# # #     app.launch()
# # import os

# # # Must be set before importing TensorFlow or PyTorch.
# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # os.environ["OMP_NUM_THREADS"] = "1"
# # os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# # os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # import csv
# # import json
# # import pickle
# # import re
# # import string
# # import warnings
# # import zipfile

# # import gradio as gr
# # import matplotlib.pyplot as plt
# # import nltk
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf

# # from nltk.corpus import stopwords
# # from nltk.stem import WordNetLemmatizer
# # from nltk.tokenize import word_tokenize
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from thefuzz import process

# # warnings.filterwarnings("ignore")

# # # Limit TensorFlow CPU threads to reduce crashes and memory pressure.
# # tf.config.threading.set_intra_op_parallelism_threads(1)
# # tf.config.threading.set_inter_op_parallelism_threads(1)

# # # =========================================================
# # # Files and settings
# # # =========================================================

# # LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
# # TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
# # LABEL_ENCODER_PATH = "label_encoder.pkl"
# # REVIEWS_FILE_PATH = "Reviews.csv"

# # SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# # SUMMARY_WORD_THRESHOLD = 15
# # DEFAULT_MAX_LENGTH = 111

# # # DistilBART is loaded only when summarization is first requested.
# # summary_tokenizer = None
# # summarization_model = None


# # # =========================================================
# # # NLTK setup
# # # =========================================================

# # for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
# #     nltk.download(package, quiet=True)

# # stop_words = set(stopwords.words("english"))
# # stop_words = stop_words.difference({"not", "no", "nor", "never"})
# # lemmatizer = WordNetLemmatizer()


# # # =========================================================
# # # Text preprocessing
# # # Must match training preprocessing
# # # =========================================================

# # def clean_html_tags(text):
# #     text = re.sub(r"<[^>]*>", " ", str(text))
# #     text = re.sub(r"\s+", " ", text)
# #     return text.strip()


# # def normalize_negations(text):
# #     text = str(text).replace("’", "'")

# #     contractions = {
# #         "can't": "can not",
# #         "cannot": "can not",
# #         "won't": "will not",
# #         "don't": "do not",
# #         "doesn't": "does not",
# #         "didn't": "did not",
# #         "isn't": "is not",
# #         "aren't": "are not",
# #         "wasn't": "was not",
# #         "weren't": "were not",
# #         "hasn't": "has not",
# #         "haven't": "have not",
# #         "hadn't": "had not",
# #         "shouldn't": "should not",
# #         "wouldn't": "would not",
# #         "couldn't": "could not",
# #         "mustn't": "must not",
# #         "needn't": "need not",
# #         "mightn't": "might not",
# #         "shan't": "shall not",
# #     }

# #     for contraction, expanded_form in contractions.items():
# #         text = re.sub(
# #             rf"\b{re.escape(contraction)}\b",
# #             expanded_form,
# #             text,
# #             flags=re.IGNORECASE,
# #         )

# #     return text


# # def preprocess_text(text):
# #     text = normalize_negations(str(text).lower())
# #     text = re.sub(r"http\S+|www\S+", " ", text)
# #     text = re.sub(r"\S+@\S+", " ", text)
# #     text = text.translate(str.maketrans("", "", string.punctuation))
# #     text = re.sub(r"\d+", " ", text)
# #     text = re.sub(r"\s+", " ", text).strip()

# #     tokens = word_tokenize(text)

# #     cleaned_tokens = [
# #         lemmatizer.lemmatize(word)
# #         for word in tokens
# #         if word.isalpha() and word not in stop_words
# #     ]

# #     return " ".join(cleaned_tokens)


# # # =========================================================
# # # Load trained LSTM artifacts
# # # =========================================================

# # def require_file(path):
# #     if not os.path.exists(path):
# #         raise FileNotFoundError(
# #             f"Missing required file: {path}. "
# #             "Upload it to the root of the Hugging Face Space."
# #         )


# # for file_path in [
# #     LSTM_MODEL_PATH,
# #     TEXT_TOKENIZER_PATH,
# #     LABEL_ENCODER_PATH,
# # ]:
# #     require_file(file_path)

# # def remove_unsupported_keras_config(value):
# #     """
# #     Remove configuration fields written by a newer Keras version but not
# #     recognized by the Keras version bundled with TensorFlow in this Space.
# #     """
# #     removed_count = 0

# #     if isinstance(value, dict):
# #         cleaned = {}

# #         for key, item in value.items():
# #             if key == "quantization_config" and item is None:
# #                 removed_count += 1
# #                 continue

# #             cleaned_item, item_removed = remove_unsupported_keras_config(item)
# #             cleaned[key] = cleaned_item
# #             removed_count += item_removed

# #         return cleaned, removed_count

# #     if isinstance(value, list):
# #         cleaned_list = []

# #         for item in value:
# #             cleaned_item, item_removed = remove_unsupported_keras_config(item)
# #             cleaned_list.append(cleaned_item)
# #             removed_count += item_removed

# #         return cleaned_list, removed_count

# #     return value, removed_count


# # def prepare_compatible_keras_model(model_path):
# #     """
# #     A .keras model is a ZIP archive. This function copies the archive to /tmp
# #     and removes only unsupported null quantization_config fields from
# #     config.json. The original uploaded model is never modified.
# #     """
# #     if not zipfile.is_zipfile(model_path):
# #         return model_path

# #     patched_path = "/tmp/lstm_sentiment_model_compatible.keras"

# #     with zipfile.ZipFile(model_path, "r") as source_archive:
# #         if "config.json" not in source_archive.namelist():
# #             return model_path

# #         model_config = json.loads(
# #             source_archive.read("config.json").decode("utf-8")
# #         )

# #         cleaned_config, removed_count = remove_unsupported_keras_config(
# #             model_config
# #         )

# #         if removed_count == 0:
# #             return model_path

# #         with zipfile.ZipFile(
# #             patched_path,
# #             "w",
# #             compression=zipfile.ZIP_DEFLATED,
# #         ) as target_archive:
# #             for archive_item in source_archive.infolist():
# #                 file_data = source_archive.read(archive_item.filename)

# #                 if archive_item.filename == "config.json":
# #                     file_data = json.dumps(
# #                         cleaned_config,
# #                         ensure_ascii=False,
# #                     ).encode("utf-8")

# #                 target_archive.writestr(archive_item, file_data)

# #     print(
# #         "Prepared a Keras-compatible model copy. "
# #         f"Removed {removed_count} unsupported config fields."
# #     )

# #     return patched_path


# # print("Loading trained LSTM model...")

# # compatible_model_path = prepare_compatible_keras_model(
# #     LSTM_MODEL_PATH
# # )

# # sentiment_model = load_model(
# #     compatible_model_path,
# #     compile=False,
# # )

# # with open(TEXT_TOKENIZER_PATH, "rb") as file:
# #     text_tokenizer = pickle.load(file)

# # with open(LABEL_ENCODER_PATH, "rb") as file:
# #     label_encoder = pickle.load(file)

# # try:
# #     MAX_LENGTH = int(sentiment_model.input_shape[1])
# # except (TypeError, ValueError, IndexError):
# #     MAX_LENGTH = DEFAULT_MAX_LENGTH

# # print(f"LSTM model loaded. MAX_LENGTH={MAX_LENGTH}")


# # # =========================================================
# # # Lazy DistilBART loading
# # # =========================================================

# # def get_summarization_model():
# #     global summary_tokenizer
# #     global summarization_model

# #     if summary_tokenizer is None or summarization_model is None:
# #         print("Loading DistilBART on first summarization request...")

# #         # Imported lazily so PyTorch/Transformers do not load at startup.
# #         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# #         summary_tokenizer = AutoTokenizer.from_pretrained(
# #             SUMMARIZATION_MODEL_NAME
# #         )

# #         summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
# #             SUMMARIZATION_MODEL_NAME,
# #             low_cpu_mem_usage=True,
# #         )

# #         summarization_model.eval()
# #         print("DistilBART loaded.")

# #     return summary_tokenizer, summarization_model


# # # =========================================================
# # # Optional Reviews.csv for search/dashboard
# # # =========================================================

# # def map_score_to_sentiment(score):
# #     if score in [1, 2]:
# #         return "negative"
# #     if score == 3:
# #         return "neutral"
# #     return "positive"


# # def read_csv_safely(file_path, max_valid_rows=100000):
# #     valid_rows = []
# #     bad_rows = []

# #     with open(
# #         file_path,
# #         "r",
# #         encoding="utf-8",
# #         errors="replace",
# #         newline="",
# #     ) as file:
# #         reader = csv.reader(file, delimiter=",", quotechar='"', strict=True)

# #         try:
# #             columns = next(reader)
# #         except StopIteration:
# #             raise ValueError("Reviews.csv is empty.")

# #         columns = [
# #             column.strip().replace("\ufeff", "")
# #             for column in columns
# #         ]

# #         expected_columns = len(columns)

# #         while len(valid_rows) < max_valid_rows:
# #             try:
# #                 row = next(reader)
# #             except StopIteration:
# #                 break
# #             except csv.Error as error:
# #                 bad_rows.append(
# #                     {"line": reader.line_num, "error": str(error)}
# #                 )
# #                 continue

# #             if len(row) != expected_columns:
# #                 bad_rows.append(
# #                     {
# #                         "line": reader.line_num,
# #                         "error": (
# #                             f"Expected {expected_columns} fields, "
# #                             f"found {len(row)}."
# #                         ),
# #                     }
# #                 )
# #                 continue

# #             valid_rows.append(row)

# #     return pd.DataFrame(valid_rows, columns=columns), bad_rows


# # def load_reviews_dataset():
# #     if not os.path.exists(REVIEWS_FILE_PATH):
# #         print("Reviews.csv not found. Search and dashboard remain unavailable.")
# #         return None

# #     try:
# #         dataset, bad_rows = read_csv_safely(REVIEWS_FILE_PATH)

# #         required_columns = {"Text", "Summary", "Score"}
# #         if not required_columns.issubset(dataset.columns):
# #             print("Reviews.csv is missing required columns.")
# #             return None

# #         dataset = dataset[["Text", "Summary", "Score"]].copy()
# #         dataset["Score"] = pd.to_numeric(dataset["Score"], errors="coerce")
# #         dataset = dataset.dropna(subset=["Text", "Summary", "Score"])
# #         dataset = dataset[dataset["Score"].isin([1, 2, 3, 4, 5])].copy()
# #         dataset["Score"] = dataset["Score"].astype(int)
# #         dataset["Sentiment"] = dataset["Score"].apply(map_score_to_sentiment)
# #         dataset["Cleaned_Text"] = dataset["Text"].apply(clean_html_tags)
# #         dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(preprocess_text)
# #         dataset = dataset[
# #             dataset["Processed_Text"].str.strip() != ""
# #         ].reset_index(drop=True)

# #         print(
# #             f"Reviews.csv loaded: {len(dataset)} valid rows, "
# #             f"{len(bad_rows)} corrupted rows skipped."
# #         )

# #         return dataset

# #     except Exception as error:
# #         print(f"Could not load Reviews.csv: {error}")
# #         return None


# # df = load_reviews_dataset()


# # # =========================================================
# # # Inference
# # # =========================================================

# # def predict_sentiment(raw_text):
# #     raw_text = str(raw_text).strip()

# #     if not raw_text:
# #         raise ValueError("Please enter a review.")

# #     processed_text = preprocess_text(clean_html_tags(raw_text))

# #     if not processed_text:
# #         raise ValueError("The text became empty after preprocessing.")

# #     sequence = text_tokenizer.texts_to_sequences([processed_text])

# #     padded_sequence = pad_sequences(
# #         sequence,
# #         maxlen=MAX_LENGTH,
# #         padding="post",
# #         truncating="post",
# #     )

# #     probabilities = sentiment_model.predict(
# #         padded_sequence,
# #         verbose=0,
# #     )[0]

# #     predicted_index = int(np.argmax(probabilities))
# #     sentiment = label_encoder.inverse_transform([predicted_index])[0]
# #     confidence = float(probabilities[predicted_index])

# #     all_probabilities = {
# #         class_name: float(probabilities[index])
# #         for index, class_name in enumerate(label_encoder.classes_)
# #     }

# #     return sentiment, confidence, all_probabilities


# # def summarize_review(raw_text):
# #     raw_text = str(raw_text).strip()

# #     if not raw_text:
# #         raise ValueError("Please enter a review.")

# #     word_count = len(raw_text.split())

# #     if word_count <= SUMMARY_WORD_THRESHOLD:
# #         return (
# #             "The text is too short to require summarization. "
# #             "/ النص قصير جداً ولا يحتاج إلى تلخيص."
# #         )

# #     tokenizer, model = get_summarization_model()

# #     inputs = tokenizer(
# #         raw_text,
# #         max_length=1024,
# #         return_tensors="pt",
# #         truncation=True,
# #     )

# #     summary_ids = model.generate(
# #         input_ids=inputs["input_ids"],
# #         attention_mask=inputs["attention_mask"],
# #         max_new_tokens=30,
# #         no_repeat_ngram_size=2,
# #         num_beams=2,
# #         early_stopping=True,
# #     )

# #     summary = tokenizer.decode(
# #         summary_ids[0],
# #         skip_special_tokens=True,
# #     ).strip()

# #     if not summary or len(summary.split()) >= word_count:
# #         return raw_text

# #     return summary


# # def analyze_review(raw_text):
# #     sentiment, confidence, probabilities = predict_sentiment(raw_text)
# #     summary = summarize_review(raw_text)

# #     return {
# #         "summary": summary,
# #         "sentiment": sentiment,
# #         "confidence": confidence,
# #         "probabilities": probabilities,
# #     }


# # # =========================================================
# # # Gradio functions
# # # =========================================================

# # def sentiment_emoji(sentiment):
# #     return {
# #         "positive": "😊",
# #         "neutral": "😐",
# #         "negative": "😢",
# #     }.get(sentiment.lower(), "🔎")


# # def web_interface_fn(review_text):
# #     if not review_text or not review_text.strip():
# #         return "Please enter a review.", "", {}, "", "", "", ""

# #     try:
# #         raw_tokens = word_tokenize(review_text.lower())

# #         normalized = normalize_negations(review_text.lower())
# #         no_punctuation_text = normalized.translate(
# #             str.maketrans("", "", string.punctuation)
# #         )
# #         no_punctuation_text = re.sub(r"\d+", " ", no_punctuation_text)
# #         no_punctuation_tokens = word_tokenize(no_punctuation_text)

# #         without_stop_words = [
# #             word
# #             for word in no_punctuation_tokens
# #             if word.isalpha() and word not in stop_words
# #         ]

# #         lemmatized = [
# #             lemmatizer.lemmatize(word)
# #             for word in without_stop_words
# #         ]

# #         result = analyze_review(review_text)

# #         sentiment_text = (
# #             f"{result['sentiment'].upper()} "
# #             f"{sentiment_emoji(result['sentiment'])}\n"
# #             f"Confidence: {result['confidence'] * 100:.2f}%"
# #         )

# #         labels = {
# #             name.capitalize(): probability
# #             for name, probability in result["probabilities"].items()
# #         }

# #         return (
# #             result["summary"],
# #             sentiment_text,
# #             labels,
# #             " | ".join(raw_tokens),
# #             " | ".join(no_punctuation_tokens),
# #             " | ".join(without_stop_words),
# #             " | ".join(lemmatized),
# #         )

# #     except Exception as error:
# #         return f"Error: {error}", "", {}, "", "", "", ""


# # search_history = []


# # def search_in_dataset_with_history(query):
# #     global search_history

# #     if df is None:
# #         return (
# #             "Reviews.csv is not available in this Space.",
# #             "\n".join(search_history),
# #         )

# #     if not query or not query.strip():
# #         return "Please enter a search term.", "\n".join(search_history)

# #     common_keywords = [
# #         "battery", "screen", "camera", "price", "performance",
# #         "design", "software", "quality", "money", "coffee",
# #         "food", "tea", "taste",
# #     ]

# #     match = process.extractOne(query.lower(), common_keywords)
# #     actual_query = (
# #         match[0]
# #         if match and match[1] > 80
# #         else query.lower().strip()
# #     )

# #     if query not in search_history:
# #         search_history.insert(0, query)
# #         search_history = search_history[:5]

# #     results = df[
# #         df["Text"].str.contains(
# #             actual_query,
# #             case=False,
# #             na=False,
# #             regex=False,
# #         )
# #     ]

# #     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"

# #     if results.empty:
# #         output += "No matching reviews were found."

# #     for _, row in results.head(5).iterrows():
# #         output += (
# #             f"📝 {str(row['Text'])[:180]}...\n"
# #             f"📊 {str(row['Sentiment']).upper()}\n\n"
# #         )

# #     return output, "\n".join(search_history)


# # def generate_dashboard(query):
# #     if df is None:
# #         fig, ax = plt.subplots(figsize=(7, 5))
# #         ax.text(
# #             0.5, 0.5,
# #             "Reviews.csv is not available.",
# #             ha="center",
# #             va="center",
# #             fontsize=13,
# #         )
# #         ax.axis("off")
# #         return fig

# #     if query and query.strip():
# #         filtered_df = df[
# #             df["Text"].str.contains(
# #                 query.strip(),
# #                 case=False,
# #                 na=False,
# #                 regex=False,
# #             )
# #         ]
# #         title = f"Sentiment Distribution for: {query.strip().capitalize()}"
# #     else:
# #         filtered_df = df
# #         title = "General Sentiment Distribution"

# #     if filtered_df.empty:
# #         fig, ax = plt.subplots(figsize=(7, 5))
# #         ax.text(
# #             0.5, 0.5,
# #             "No data found for this feature.",
# #             ha="center",
# #             va="center",
# #             fontsize=13,
# #         )
# #         ax.axis("off")
# #         return fig

# #     counts = (
# #         filtered_df["Sentiment"]
# #         .value_counts()
# #         .reindex(["positive", "neutral", "negative"], fill_value=0)
# #     )

# #     fig, ax = plt.subplots(figsize=(6, 6))
# #     counts.plot(
# #         kind="pie",
# #         autopct="%1.1f%%",
# #         startangle=140,
# #         ax=ax,
# #         textprops={"fontsize": 12, "weight": "bold"},
# #     )

# #     ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
# #     ax.set_ylabel("")
# #     plt.tight_layout()

# #     return fig


# # # =========================================================
# # # Gradio interface
# # # =========================================================

# # CUSTOM_CSS = """
# # :root {
# #     --app-bg: #eef3ff;
# #     --surface: #ffffff;
# #     --surface-soft: #f7f9ff;
# #     --text-main: #172033;
# #     --text-muted: #475569;
# #     --primary: #4667e8;
# #     --primary-dark: #2f49c7;
# #     --accent: #6d5dfc;
# #     --border: #d7def2;
# #     --shadow: 0 16px 36px rgba(34, 57, 145, 0.12);
# #     --shadow-soft: 0 8px 22px rgba(34, 57, 145, 0.08);
# # }

# # .gradio-container {
# #     background:
# #         linear-gradient(180deg, #eef3ff 0%, #f7f9ff 38%, #eef3ff 100%) !important;
# #     color: var(--text-main) !important;
# #     min-height: 100vh;
# # }

# # .app-shell {
# #     max-width: 1180px;
# #     margin: 0 auto;
# # }

# # .hero {
# #     position: relative;
# #     overflow: hidden;
# #     border-radius: 24px;
# #     padding: 32px;
# #     margin: 8px 0 22px;
# #     color: #ffffff;
# #     background: linear-gradient(135deg, #2f49c7 0%, #4667e8 58%, #6d5dfc 100%);
# #     box-shadow: 0 22px 52px rgba(59, 91, 219, 0.28);
# # }

# # .hero::before,
# # .hero::after {
# #     content: "";
# #     position: absolute;
# #     border-radius: 999px;
# #     background: rgba(255, 255, 255, 0.10);
# # }

# # .hero::before {
# #     width: 210px;
# #     height: 210px;
# #     top: -90px;
# #     right: -55px;
# # }

# # .hero::after {
# #     width: 145px;
# #     height: 145px;
# #     bottom: -75px;
# #     left: 7%;
# # }

# # .hero-content {
# #     position: relative;
# #     z-index: 1;
# # }

# # .hero h1 {
# #     margin: 0 0 12px;
# #     font-size: clamp(30px, 4vw, 46px);
# #     line-height: 1.08;
# #     color: #ffffff !important;
# # }

# # .hero p {
# #     max-width: 780px;
# #     margin: 0;
# #     color: #eef2ff !important;
# #     font-size: 16px;
# #     line-height: 1.7;
# #     font-weight: 500;
# # }

# # .team-grid {
# #     position: relative;
# #     z-index: 1;
# #     display: grid;
# #     grid-template-columns: repeat(3, minmax(0, 1fr));
# #     gap: 12px;
# #     margin-top: 24px;
# # }

# # .team-card {
# #     min-width: 0;
# #     padding: 15px 16px;
# #     border: 1px solid rgba(255, 255, 255, 0.30);
# #     border-radius: 16px;
# #     background: rgba(255, 255, 255, 0.16);
# #     backdrop-filter: blur(8px);
# #     color: #ffffff !important;
# #     font-weight: 800;
# #     text-align: center;
# #     white-space: normal;
# #     overflow-wrap: anywhere;
# # }

# # .team-card span {
# #     display: block;
# #     margin-bottom: 5px;
# #     color: #dbe4ff !important;
# #     font-size: 11px;
# #     font-weight: 800;
# #     letter-spacing: 0.08em;
# #     text-transform: uppercase;
# # }

# # .feature-strip {
# #     display: grid;
# #     grid-template-columns: repeat(3, minmax(0, 1fr));
# #     gap: 12px;
# #     margin-bottom: 20px;
# # }

# # .feature-card {
# #     padding: 17px 18px;
# #     border: 1px solid var(--border);
# #     border-radius: 17px;
# #     background: linear-gradient(180deg, #ffffff 0%, #f7f9ff 100%);
# #     box-shadow: var(--shadow-soft);
# # }

# # .feature-card strong {
# #     display: block;
# #     margin-bottom: 5px;
# #     color: #24335f !important;
# #     font-size: 15px;
# # }

# # .feature-card small {
# #     color: var(--text-muted) !important;
# #     line-height: 1.55;
# #     font-size: 13px;
# # }

# # .app-tabs {
# #     border: 1px solid var(--border) !important;
# #     border-radius: 20px !important;
# #     background: #ffffff !important;
# #     box-shadow: var(--shadow) !important;
# #     padding: 8px !important;
# # }

# # .app-tabs button[role="tab"] {
# #     color: #32436f !important;
# #     font-weight: 700 !important;
# # }

# # .app-tabs button[role="tab"][aria-selected="true"] {
# #     color: #ffffff !important;
# #     background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
# #     border-radius: 12px !important;
# # }

# # .section-card {
# #     border: 1px solid var(--border) !important;
# #     border-radius: 17px !important;
# #     background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%) !important;
# #     box-shadow: var(--shadow-soft) !important;
# #     padding: 10px !important;
# # }

# # .section-card h1,
# # .section-card h2,
# # .section-card h3,
# # .section-card p,
# # .section-card label,
# # .section-card span {
# #     color: var(--text-main) !important;
# # }

# # .section-card textarea,
# # .section-card input {
# #     background: #ffffff !important;
# #     color: #111827 !important;
# #     border: 1px solid #cfd8f2 !important;
# #     border-radius: 13px !important;
# # }

# # .section-card textarea::placeholder,
# # .section-card input::placeholder {
# #     color: #7c8aa8 !important;
# #     opacity: 1 !important;
# # }

# # .primary-action,
# # .primary-action button {
# #     border: none !important;
# #     border-radius: 13px !important;
# #     background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
# #     color: #ffffff !important;
# #     font-weight: 800 !important;
# #     box-shadow: 0 10px 22px rgba(59, 91, 219, 0.24) !important;
# # }

# # .secondary-action,
# # .secondary-action button {
# #     border: 1px solid #b9c7f5 !important;
# #     border-radius: 13px !important;
# #     background: #eaf0ff !important;
# #     color: #2f49c7 !important;
# #     font-weight: 800 !important;
# # }

# # .gradio-container .block {
# #     color: var(--text-main) !important;
# # }

# # .gradio-container .wrap,
# # .gradio-container .panel,
# # .gradio-container .form {
# #     background: transparent !important;
# # }

# # footer {
# #     display: none !important;
# # }

# # .footer-note {
# #     margin: 18px 0 4px;
# #     padding: 14px;
# #     text-align: center;
# #     color: #52627f !important;
# #     font-size: 13px;
# # }

# # @media (max-width: 760px) {
# #     .hero {
# #         padding: 24px 20px;
# #         border-radius: 20px;
# #     }

# #     .team-grid,
# #     .feature-strip {
# #         grid-template-columns: 1fr;
# #     }

# #     .team-card {
# #         text-align: left;
# #     }
# # }
# # """

# # with gr.Blocks(
# #     title="LSTM Sentiment Analysis",
# #     theme=gr.themes.Soft(),
# #     css=CUSTOM_CSS,
# # ) as app:

# #     with gr.Column(elem_classes=["app-shell"]):

# #         gr.HTML(
# #             """
# #             <section class="hero">
# #                 <div class="hero-content">
# #                     <h1>Sentiment Analysis & Smart Summarization</h1>
# #                     <p>
# #                         Analyze English product and food reviews using a trained
# #                         LSTM model, generate concise summaries with DistilBART,
# #                         and explore dataset insights through a clean interactive dashboard.
# #                     </p>
# #                 </div>

# #                 <div class="team-grid">
# #                     <div class="team-card">
# #                         <span>Team Member</span>
# #                         SAIF KHATTAB
# #                     </div>
# #                     <div class="team-card">
# #                         <span>Team Member</span>
# #                         YAZEED AMEEN
# #                     </div>
# #                     <div class="team-card">
# #                         <span>Team Member</span>
# #                         ASEEL MANNOUN
# #                     </div>
# #                 </div>
# #             </section>
# #             """
# #         )

# #         gr.HTML(
# #             """
# #             <div class="feature-strip">
# #                 <div class="feature-card">
# #                     <strong>🧠 LSTM Classification</strong>
# #                     <small>Predicts positive, neutral, or negative sentiment.</small>
# #                 </div>
# #                 <div class="feature-card">
# #                     <strong>📝 Smart Summarization</strong>
# #                     <small>Creates a concise summary for longer reviews.</small>
# #                 </div>
# #                 <div class="feature-card">
# #                     <strong>📊 Dataset Insights</strong>
# #                     <small>Search reviews and visualize sentiment distribution.</small>
# #                 </div>
# #             </div>
# #             """
# #         )

# #         with gr.Tabs(elem_classes=["app-tabs"]):

# #             with gr.Tab("📝 Analyze Review"):
# #                 with gr.Column(elem_classes=["section-card"]):
# #                     gr.Markdown(
# #                         "### Analyze a Review\n"
# #                         "Enter an English review to generate its summary, sentiment, "
# #                         "confidence score, and processing stages."
# #                     )

# #                     input_text = gr.Textbox(
# #                         lines=7,
# #                         label="Review Text",
# #                         placeholder="Write or paste an English food or product review...",
# #                     )

# #                     analyze_button = gr.Button(
# #                         "Analyze Review",
# #                         variant="primary",
# #                         elem_classes=["primary-action"],
# #                     )

# #                     with gr.Row(equal_height=True):
# #                         summary_output = gr.Textbox(
# #                             label="Generated Summary",
# #                             lines=5,
# #                         )
# #                         sentiment_output = gr.Textbox(
# #                             label="Predicted Sentiment",
# #                             lines=5,
# #                         )

# #                     probability_output = gr.Label(
# #                         label="Class Probabilities",
# #                         num_top_classes=3,
# #                     )

# #                     with gr.Accordion(
# #                         "🔍 Text Processing Stages",
# #                         open=False,
# #                     ):
# #                         gr.Markdown(
# #                             "These fields show how the review is transformed "
# #                             "before it reaches the LSTM model."
# #                         )
# #                         raw_tokens_output = gr.Textbox(label="1. Raw Tokens")
# #                         no_punctuation_output = gr.Textbox(label="2. No Punctuation")
# #                         no_stop_words_output = gr.Textbox(label="3. No Stop Words")
# #                         lemmatized_output = gr.Textbox(label="4. Lemmatized")

# #                     analyze_button.click(
# #                         fn=web_interface_fn,
# #                         inputs=input_text,
# #                         outputs=[
# #                             summary_output,
# #                             sentiment_output,
# #                             probability_output,
# #                             raw_tokens_output,
# #                             no_punctuation_output,
# #                             no_stop_words_output,
# #                             lemmatized_output,
# #                         ],
# #                     )

# #             with gr.Tab("🔎 Search Dataset"):
# #                 with gr.Column(elem_classes=["section-card"]):
# #                     gr.Markdown(
# #                         "### Search Reviews\n"
# #                         "Search the dataset by a word or feature and view the closest results."
# #                     )

# #                     with gr.Row(equal_height=True):
# #                         with gr.Column(scale=2):
# #                             query_input = gr.Textbox(
# #                                 label="Search Query",
# #                                 placeholder="Examples: coffee, taste, quality, price",
# #                             )
# #                             search_button = gr.Button(
# #                                 "Search Dataset",
# #                                 elem_classes=["secondary-action"],
# #                             )

# #                         with gr.Column(scale=1):
# #                             history_output = gr.Textbox(
# #                                 label="Recent Searches",
# #                                 lines=5,
# #                                 interactive=False,
# #                             )

# #                     results_output = gr.Textbox(
# #                         label="Matching Reviews",
# #                         lines=12,
# #                     )

# #                     search_button.click(
# #                         fn=search_in_dataset_with_history,
# #                         inputs=query_input,
# #                         outputs=[results_output, history_output],
# #                     )

# #             with gr.Tab("📊 Analytics Dashboard"):
# #                 with gr.Column(elem_classes=["section-card"]):
# #                     gr.Markdown(
# #                         "### Sentiment Dashboard\n"
# #                         "Filter the dataset by a feature to view its sentiment distribution."
# #                     )

# #                     with gr.Row(equal_height=True):
# #                         dashboard_query = gr.Textbox(
# #                             label="Feature Filter",
# #                             placeholder="Examples: coffee, food, tea, quality",
# #                             scale=3,
# #                         )
# #                         dashboard_button = gr.Button(
# #                             "Generate Chart",
# #                             elem_classes=["primary-action"],
# #                             scale=1,
# #                         )

# #                     plot_output = gr.Plot(label="Sentiment Distribution")

# #                     dashboard_button.click(
# #                         fn=generate_dashboard,
# #                         inputs=dashboard_query,
# #                         outputs=plot_output,
# #                     )

# #         gr.HTML(
# #             """
# #             <div class="footer-note">
# #                 Built with TensorFlow, Keras, Transformers, and Gradio.
# #             </div>
# #             """
# #         )


# # if __name__ == "__main__":
# #     app.launch()
# import os

# # Must be set before importing TensorFlow or PyTorch.
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import csv
# import json
# import pickle
# import re
# import string
# import warnings
# import zipfile

# import gradio as gr
# import matplotlib.pyplot as plt
# import nltk
# import numpy as np
# import pandas as pd
# import tensorflow as tf

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from thefuzz import process

# warnings.filterwarnings("ignore")

# # Limit TensorFlow CPU threads to reduce crashes and memory pressure.
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# # =========================================================
# # Files and settings
# # =========================================================

# LSTM_MODEL_PATH = "lstm_sentiment_model.keras"
# TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
# LABEL_ENCODER_PATH = "label_encoder.pkl"
# REVIEWS_FILE_PATH = "Reviews.csv"

# SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# SUMMARY_WORD_THRESHOLD = 15
# DEFAULT_MAX_LENGTH = 111

# # DistilBART is loaded only when summarization is first requested.
# summary_tokenizer = None
# summarization_model = None


# # =========================================================
# # NLTK setup
# # =========================================================

# for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
#     nltk.download(package, quiet=True)

# stop_words = set(stopwords.words("english"))
# stop_words = stop_words.difference({"not", "no", "nor", "never"})
# lemmatizer = WordNetLemmatizer()


# # =========================================================
# # Text preprocessing
# # Must match training preprocessing
# # =========================================================

# def clean_html_tags(text):
#     text = re.sub(r"<[^>]*>", " ", str(text))
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def normalize_negations(text):
#     text = str(text).replace("’", "'")

#     contractions = {
#         "can't": "can not",
#         "cannot": "can not",
#         "won't": "will not",
#         "don't": "do not",
#         "doesn't": "does not",
#         "didn't": "did not",
#         "isn't": "is not",
#         "aren't": "are not",
#         "wasn't": "was not",
#         "weren't": "were not",
#         "hasn't": "has not",
#         "haven't": "have not",
#         "hadn't": "had not",
#         "shouldn't": "should not",
#         "wouldn't": "would not",
#         "couldn't": "could not",
#         "mustn't": "must not",
#         "needn't": "need not",
#         "mightn't": "might not",
#         "shan't": "shall not",
#     }

#     for contraction, expanded_form in contractions.items():
#         text = re.sub(
#             rf"\b{re.escape(contraction)}\b",
#             expanded_form,
#             text,
#             flags=re.IGNORECASE,
#         )

#     return text


# def preprocess_text(text):
#     text = normalize_negations(str(text).lower())
#     text = re.sub(r"http\S+|www\S+", " ", text)
#     text = re.sub(r"\S+@\S+", " ", text)
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(r"\d+", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     tokens = word_tokenize(text)

#     cleaned_tokens = [
#         lemmatizer.lemmatize(word)
#         for word in tokens
#         if word.isalpha() and word not in stop_words
#     ]

#     return " ".join(cleaned_tokens)


# # =========================================================
# # Load trained LSTM artifacts
# # =========================================================

# def require_file(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(
#             f"Missing required file: {path}. "
#             "Upload it to the root of the Hugging Face Space."
#         )


# for file_path in [
#     LSTM_MODEL_PATH,
#     TEXT_TOKENIZER_PATH,
#     LABEL_ENCODER_PATH,
# ]:
#     require_file(file_path)

# def remove_unsupported_keras_config(value):
#     """
#     Remove configuration fields written by a newer Keras version but not
#     recognized by the Keras version bundled with TensorFlow in this Space.
#     """
#     removed_count = 0

#     if isinstance(value, dict):
#         cleaned = {}

#         for key, item in value.items():
#             if key == "quantization_config" and item is None:
#                 removed_count += 1
#                 continue

#             cleaned_item, item_removed = remove_unsupported_keras_config(item)
#             cleaned[key] = cleaned_item
#             removed_count += item_removed

#         return cleaned, removed_count

#     if isinstance(value, list):
#         cleaned_list = []

#         for item in value:
#             cleaned_item, item_removed = remove_unsupported_keras_config(item)
#             cleaned_list.append(cleaned_item)
#             removed_count += item_removed

#         return cleaned_list, removed_count

#     return value, removed_count


# def prepare_compatible_keras_model(model_path):
#     """
#     A .keras model is a ZIP archive. This function copies the archive to /tmp
#     and removes only unsupported null quantization_config fields from
#     config.json. The original uploaded model is never modified.
#     """
#     if not zipfile.is_zipfile(model_path):
#         return model_path

#     patched_path = "/tmp/lstm_sentiment_model_compatible.keras"

#     with zipfile.ZipFile(model_path, "r") as source_archive:
#         if "config.json" not in source_archive.namelist():
#             return model_path

#         model_config = json.loads(
#             source_archive.read("config.json").decode("utf-8")
#         )

#         cleaned_config, removed_count = remove_unsupported_keras_config(
#             model_config
#         )

#         if removed_count == 0:
#             return model_path

#         with zipfile.ZipFile(
#             patched_path,
#             "w",
#             compression=zipfile.ZIP_DEFLATED,
#         ) as target_archive:
#             for archive_item in source_archive.infolist():
#                 file_data = source_archive.read(archive_item.filename)

#                 if archive_item.filename == "config.json":
#                     file_data = json.dumps(
#                         cleaned_config,
#                         ensure_ascii=False,
#                     ).encode("utf-8")

#                 target_archive.writestr(archive_item, file_data)

#     print(
#         "Prepared a Keras-compatible model copy. "
#         f"Removed {removed_count} unsupported config fields."
#     )

#     return patched_path


# print("Loading trained LSTM model...")

# compatible_model_path = prepare_compatible_keras_model(
#     LSTM_MODEL_PATH
# )

# sentiment_model = load_model(
#     compatible_model_path,
#     compile=False,
# )

# with open(TEXT_TOKENIZER_PATH, "rb") as file:
#     text_tokenizer = pickle.load(file)

# with open(LABEL_ENCODER_PATH, "rb") as file:
#     label_encoder = pickle.load(file)

# try:
#     MAX_LENGTH = int(sentiment_model.input_shape[1])
# except (TypeError, ValueError, IndexError):
#     MAX_LENGTH = DEFAULT_MAX_LENGTH

# print(f"LSTM model loaded. MAX_LENGTH={MAX_LENGTH}")


# # =========================================================
# # Lazy DistilBART loading
# # =========================================================

# def get_summarization_model():
#     global summary_tokenizer
#     global summarization_model

#     if summary_tokenizer is None or summarization_model is None:
#         print("Loading DistilBART on first summarization request...")

#         # Imported lazily so PyTorch/Transformers do not load at startup.
#         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#         summary_tokenizer = AutoTokenizer.from_pretrained(
#             SUMMARIZATION_MODEL_NAME
#         )

#         summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
#             SUMMARIZATION_MODEL_NAME,
#             low_cpu_mem_usage=True,
#         )

#         summarization_model.eval()
#         print("DistilBART loaded.")

#     return summary_tokenizer, summarization_model


# # =========================================================
# # Optional Reviews.csv for search/dashboard
# # =========================================================

# def map_score_to_sentiment(score):
#     if score in [1, 2]:
#         return "negative"
#     if score == 3:
#         return "neutral"
#     return "positive"


# def read_csv_safely(file_path, max_valid_rows=100000):
#     valid_rows = []
#     bad_rows = []

#     with open(
#         file_path,
#         "r",
#         encoding="utf-8",
#         errors="replace",
#         newline="",
#     ) as file:
#         reader = csv.reader(file, delimiter=",", quotechar='"', strict=True)

#         try:
#             columns = next(reader)
#         except StopIteration:
#             raise ValueError("Reviews.csv is empty.")

#         columns = [
#             column.strip().replace("\ufeff", "")
#             for column in columns
#         ]

#         expected_columns = len(columns)

#         while len(valid_rows) < max_valid_rows:
#             try:
#                 row = next(reader)
#             except StopIteration:
#                 break
#             except csv.Error as error:
#                 bad_rows.append(
#                     {"line": reader.line_num, "error": str(error)}
#                 )
#                 continue

#             if len(row) != expected_columns:
#                 bad_rows.append(
#                     {
#                         "line": reader.line_num,
#                         "error": (
#                             f"Expected {expected_columns} fields, "
#                             f"found {len(row)}."
#                         ),
#                     }
#                 )
#                 continue

#             valid_rows.append(row)

#     return pd.DataFrame(valid_rows, columns=columns), bad_rows


# def load_reviews_dataset():
#     if not os.path.exists(REVIEWS_FILE_PATH):
#         print("Reviews.csv not found. Search and dashboard remain unavailable.")
#         return None

#     try:
#         dataset, bad_rows = read_csv_safely(REVIEWS_FILE_PATH)

#         required_columns = {"Text", "Summary", "Score"}
#         if not required_columns.issubset(dataset.columns):
#             print("Reviews.csv is missing required columns.")
#             return None

#         dataset = dataset[["Text", "Summary", "Score"]].copy()
#         dataset["Score"] = pd.to_numeric(dataset["Score"], errors="coerce")
#         dataset = dataset.dropna(subset=["Text", "Summary", "Score"])
#         dataset = dataset[dataset["Score"].isin([1, 2, 3, 4, 5])].copy()
#         dataset["Score"] = dataset["Score"].astype(int)
#         dataset["Sentiment"] = dataset["Score"].apply(map_score_to_sentiment)
#         dataset["Cleaned_Text"] = dataset["Text"].apply(clean_html_tags)
#         dataset["Processed_Text"] = dataset["Cleaned_Text"].apply(preprocess_text)
#         dataset = dataset[
#             dataset["Processed_Text"].str.strip() != ""
#         ].reset_index(drop=True)

#         print(
#             f"Reviews.csv loaded: {len(dataset)} valid rows, "
#             f"{len(bad_rows)} corrupted rows skipped."
#         )

#         return dataset

#     except Exception as error:
#         print(f"Could not load Reviews.csv: {error}")
#         return None


# df = load_reviews_dataset()


# # =========================================================
# # Inference
# # =========================================================

# def predict_sentiment(raw_text):
#     raw_text = str(raw_text).strip()

#     if not raw_text:
#         raise ValueError("Please enter a review.")

#     processed_text = preprocess_text(clean_html_tags(raw_text))

#     if not processed_text:
#         raise ValueError("The text became empty after preprocessing.")

#     sequence = text_tokenizer.texts_to_sequences([processed_text])

#     padded_sequence = pad_sequences(
#         sequence,
#         maxlen=MAX_LENGTH,
#         padding="post",
#         truncating="post",
#     )

#     probabilities = sentiment_model.predict(
#         padded_sequence,
#         verbose=0,
#     )[0]

#     predicted_index = int(np.argmax(probabilities))
#     sentiment = label_encoder.inverse_transform([predicted_index])[0]
#     confidence = float(probabilities[predicted_index])

#     all_probabilities = {
#         class_name: float(probabilities[index])
#         for index, class_name in enumerate(label_encoder.classes_)
#     }

#     return sentiment, confidence, all_probabilities


# def summarize_review(raw_text):
#     raw_text = str(raw_text).strip()

#     if not raw_text:
#         raise ValueError("Please enter a review.")

#     word_count = len(raw_text.split())

#     if word_count <= SUMMARY_WORD_THRESHOLD:
#         return (
#             "The text is too short to require summarization. "
#             "/ النص قصير جداً ولا يحتاج إلى تلخيص."
#         )

#     tokenizer, model = get_summarization_model()

#     inputs = tokenizer(
#         raw_text,
#         max_length=1024,
#         return_tensors="pt",
#         truncation=True,
#     )

#     summary_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=30,
#         no_repeat_ngram_size=2,
#         num_beams=2,
#         early_stopping=True,
#     )

#     summary = tokenizer.decode(
#         summary_ids[0],
#         skip_special_tokens=True,
#     ).strip()

#     if not summary or len(summary.split()) >= word_count:
#         return raw_text

#     return summary


# def analyze_review(raw_text):
#     sentiment, confidence, probabilities = predict_sentiment(raw_text)
#     summary = summarize_review(raw_text)

#     return {
#         "summary": summary,
#         "sentiment": sentiment,
#         "confidence": confidence,
#         "probabilities": probabilities,
#     }


# # =========================================================
# # Gradio functions
# # =========================================================

# def sentiment_emoji(sentiment):
#     return {
#         "positive": "😊",
#         "neutral": "😐",
#         "negative": "😢",
#     }.get(sentiment.lower(), "🔎")


# def web_interface_fn(review_text):
#     if not review_text or not review_text.strip():
#         return "Please enter a review.", "", {}, "", "", "", ""

#     try:
#         raw_tokens = word_tokenize(review_text.lower())

#         normalized = normalize_negations(review_text.lower())
#         no_punctuation_text = normalized.translate(
#             str.maketrans("", "", string.punctuation)
#         )
#         no_punctuation_text = re.sub(r"\d+", " ", no_punctuation_text)
#         no_punctuation_tokens = word_tokenize(no_punctuation_text)

#         without_stop_words = [
#             word
#             for word in no_punctuation_tokens
#             if word.isalpha() and word not in stop_words
#         ]

#         lemmatized = [
#             lemmatizer.lemmatize(word)
#             for word in without_stop_words
#         ]

#         result = analyze_review(review_text)

#         sentiment_text = (
#             f"{result['sentiment'].upper()} "
#             f"{sentiment_emoji(result['sentiment'])}\n"
#             f"Confidence: {result['confidence'] * 100:.2f}%"
#         )

#         labels = {
#             name.capitalize(): probability
#             for name, probability in result["probabilities"].items()
#         }

#         return (
#             result["summary"],
#             sentiment_text,
#             labels,
#             " | ".join(raw_tokens),
#             " | ".join(no_punctuation_tokens),
#             " | ".join(without_stop_words),
#             " | ".join(lemmatized),
#         )

#     except Exception as error:
#         return f"Error: {error}", "", {}, "", "", "", ""


# search_history = []


# def search_in_dataset_with_history(query):
#     global search_history

#     if df is None:
#         return (
#             "Reviews.csv is not available in this Space.",
#             "\n".join(search_history),
#         )

#     if not query or not query.strip():
#         return "Please enter a search term.", "\n".join(search_history)

#     common_keywords = [
#         "battery", "screen", "camera", "price", "performance",
#         "design", "software", "quality", "money", "coffee",
#         "food", "tea", "taste",
#     ]

#     match = process.extractOne(query.lower(), common_keywords)
#     actual_query = (
#         match[0]
#         if match and match[1] > 80
#         else query.lower().strip()
#     )

#     if query not in search_history:
#         search_history.insert(0, query)
#         search_history = search_history[:5]

#     results = df[
#         df["Text"].str.contains(
#             actual_query,
#             case=False,
#             na=False,
#             regex=False,
#         )
#     ]

#     output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"

#     if results.empty:
#         output += "No matching reviews were found."

#     for _, row in results.head(5).iterrows():
#         output += (
#             f"📝 {str(row['Text'])[:180]}...\n"
#             f"📊 {str(row['Sentiment']).upper()}\n\n"
#         )

#     return output, "\n".join(search_history)


# def generate_dashboard(query):
#     if df is None:
#         fig, ax = plt.subplots(figsize=(7, 5))
#         ax.text(
#             0.5, 0.5,
#             "Reviews.csv is not available.",
#             ha="center",
#             va="center",
#             fontsize=13,
#         )
#         ax.axis("off")
#         return fig

#     if query and query.strip():
#         filtered_df = df[
#             df["Text"].str.contains(
#                 query.strip(),
#                 case=False,
#                 na=False,
#                 regex=False,
#             )
#         ]
#         title = f"Sentiment Distribution for: {query.strip().capitalize()}"
#     else:
#         filtered_df = df
#         title = "General Sentiment Distribution"

#     if filtered_df.empty:
#         fig, ax = plt.subplots(figsize=(7, 5))
#         ax.text(
#             0.5, 0.5,
#             "No data found for this feature.",
#             ha="center",
#             va="center",
#             fontsize=13,
#         )
#         ax.axis("off")
#         return fig

#     counts = (
#         filtered_df["Sentiment"]
#         .value_counts()
#         .reindex(["positive", "neutral", "negative"], fill_value=0)
#     )

#     fig, ax = plt.subplots(figsize=(6, 6))
#     counts.plot(
#         kind="pie",
#         autopct="%1.1f%%",
#         startangle=140,
#         ax=ax,
#         textprops={"fontsize": 12, "weight": "bold"},
#     )

#     ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
#     ax.set_ylabel("")
#     plt.tight_layout()

#     return fig


# # =========================================================
# # Gradio interface
# # =========================================================

# CUSTOM_CSS = """
# :root {
#     --bg-top: #eef4ff;
#     --bg-bottom: #f9fbff;
#     --surface: rgba(255, 255, 255, 0.92);
#     --surface-strong: #ffffff;
#     --surface-blue: #f2f6ff;
#     --text-main: #17213d;
#     --text-soft: #53617f;
#     --primary: #4d6ff2;
#     --primary-deep: #2f49c7;
#     --accent: #7185ff;
#     --border: rgba(111, 137, 220, 0.22);
#     --shadow-main: 0 24px 65px rgba(45, 70, 160, 0.15);
#     --shadow-soft: 0 12px 30px rgba(45, 70, 160, 0.09);
# }

# * {
#     box-sizing: border-box;
# }

# .gradio-container {
#     position: relative;
#     min-height: 100vh;
#     color: var(--text-main) !important;
#     background:
#         radial-gradient(circle at 10% 8%, rgba(101, 126, 245, 0.16), transparent 27%),
#         radial-gradient(circle at 92% 15%, rgba(154, 170, 255, 0.14), transparent 25%),
#         linear-gradient(180deg, var(--bg-top) 0%, #f6f9ff 48%, var(--bg-bottom) 100%) !important;
# }

# .gradio-container::before {
#     content: "";
#     position: fixed;
#     inset: 0;
#     pointer-events: none;
#     opacity: 0.42;
#     background-image:
#         linear-gradient(rgba(77, 111, 242, 0.025) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(77, 111, 242, 0.025) 1px, transparent 1px);
#     background-size: 34px 34px;
# }

# .app-shell {
#     position: relative;
#     z-index: 1;
#     max-width: 1180px;
#     margin: 0 auto;
#     padding: 8px 4px 24px;
# }

# .hero {
#     position: relative;
#     overflow: hidden;
#     min-height: 300px;
#     border: 1px solid rgba(255, 255, 255, 0.42);
#     border-radius: 30px;
#     padding: 42px 40px 34px;
#     margin: 8px 0 24px;
#     color: #ffffff;
#     background:
#         radial-gradient(circle at 78% 25%, rgba(255, 255, 255, 0.18), transparent 22%),
#         linear-gradient(135deg, #2f49c7 0%, #4667e8 50%, #6d78f6 100%);
#     box-shadow: 0 28px 70px rgba(47, 73, 199, 0.30);
# }

# .hero::before {
#     content: "";
#     position: absolute;
#     width: 260px;
#     height: 260px;
#     top: -120px;
#     right: -80px;
#     border-radius: 48% 52% 62% 38% / 50% 42% 58% 50%;
#     border: 1px solid rgba(255, 255, 255, 0.24);
#     background: rgba(255, 255, 255, 0.08);
#     transform: rotate(18deg);
# }

# .hero::after {
#     content: "";
#     position: absolute;
#     width: 200px;
#     height: 200px;
#     left: -55px;
#     bottom: -110px;
#     border-radius: 58% 42% 46% 54% / 44% 55% 45% 56%;
#     background: rgba(255, 255, 255, 0.09);
#     transform: rotate(-12deg);
# }

# .hero-content {
#     position: relative;
#     z-index: 2;
#     max-width: 820px;
# }

# .hero h1 {
#     margin: 0 0 14px;
#     color: #ffffff !important;
#     font-size: clamp(34px, 4.4vw, 52px);
#     line-height: 1.05;
#     letter-spacing: -0.035em;
#     text-shadow: 0 3px 18px rgba(20, 34, 100, 0.24);
# }

# .hero p {
#     max-width: 800px;
#     margin: 0;
#     color: rgba(255, 255, 255, 0.92) !important;
#     font-size: 16px;
#     line-height: 1.75;
#     font-weight: 500;
# }

# .team-grid {
#     position: relative;
#     z-index: 2;
#     display: grid;
#     grid-template-columns: repeat(3, minmax(0, 1fr));
#     gap: 14px;
#     margin-top: 30px;
# }

# .team-card {
#     position: relative;
#     min-width: 0;
#     overflow: hidden;
#     padding: 17px 18px;
#     border: 1px solid rgba(255, 255, 255, 0.30);
#     border-radius: 19px;
#     color: #ffffff !important;
#     background:
#         linear-gradient(145deg, rgba(255, 255, 255, 0.19), rgba(255, 255, 255, 0.10));
#     backdrop-filter: blur(12px);
#     box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.22);
#     font-weight: 800;
#     text-align: center;
# }

# .team-card::before {
#     content: "";
#     position: absolute;
#     width: 70px;
#     height: 70px;
#     top: -38px;
#     right: -25px;
#     border-radius: 50%;
#     background: rgba(255, 255, 255, 0.10);
# }

# .team-card span {
#     display: block;
#     margin-bottom: 5px;
#     color: #dce6ff !important;
#     font-size: 10px;
#     font-weight: 800;
#     letter-spacing: 0.11em;
#     text-transform: uppercase;
# }

# .feature-strip {
#     display: grid;
#     grid-template-columns: repeat(3, minmax(0, 1fr));
#     gap: 15px;
#     margin-bottom: 22px;
# }

# .feature-card {
#     position: relative;
#     overflow: hidden;
#     min-height: 118px;
#     padding: 20px 20px 18px;
#     border: 1px solid var(--border);
#     border-radius: 22px;
#     background:
#         linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(244, 248, 255, 0.92));
#     box-shadow: var(--shadow-soft);
# }

# .feature-card::after {
#     content: "";
#     position: absolute;
#     width: 90px;
#     height: 90px;
#     right: -38px;
#     bottom: -46px;
#     border-radius: 50%;
#     background: rgba(77, 111, 242, 0.07);
# }

# .feature-card strong {
#     position: relative;
#     z-index: 1;
#     display: block;
#     margin-bottom: 7px;
#     color: #25345f !important;
#     font-size: 15px;
# }

# .feature-card small {
#     position: relative;
#     z-index: 1;
#     color: var(--text-soft) !important;
#     font-size: 13px;
#     line-height: 1.6;
# }

# .app-tabs {
#     position: relative;
#     overflow: hidden;
#     border: 1px solid var(--border) !important;
#     border-radius: 26px !important;
#     background: rgba(255, 255, 255, 0.88) !important;
#     backdrop-filter: blur(14px);
#     box-shadow: var(--shadow-main) !important;
#     padding: 10px !important;
# }

# .app-tabs::before {
#     content: "";
#     position: absolute;
#     width: 170px;
#     height: 170px;
#     right: -70px;
#     top: -90px;
#     border-radius: 50%;
#     background: rgba(99, 123, 236, 0.06);
#     pointer-events: none;
# }

# .app-tabs button[role="tab"] {
#     min-height: 46px;
#     border-radius: 14px !important;
#     color: #415177 !important;
#     font-weight: 750 !important;
#     transition: all 0.2s ease !important;
# }

# .app-tabs button[role="tab"]:hover {
#     color: #2f49c7 !important;
#     background: #eef3ff !important;
# }

# .app-tabs button[role="tab"][aria-selected="true"] {
#     color: #ffffff !important;
#     background: linear-gradient(135deg, #2f49c7, #5576ed) !important;
#     box-shadow: 0 8px 20px rgba(47, 73, 199, 0.22) !important;
# }

# .section-card {
#     border: 1px solid rgba(115, 137, 207, 0.17) !important;
#     border-radius: 22px !important;
#     background:
#         linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.96)) !important;
#     box-shadow: 0 10px 28px rgba(52, 75, 151, 0.07) !important;
#     padding: 16px !important;
# }

# .section-card h1,
# .section-card h2,
# .section-card h3 {
#     color: #1c294b !important;
#     letter-spacing: -0.015em;
# }

# .section-card p,
# .section-card label,
# .section-card span {
#     color: #405171 !important;
# }

# .section-card textarea,
# .section-card input {
#     min-height: 48px;
#     border: 1px solid #d4ddf3 !important;
#     border-radius: 16px !important;
#     color: #17213d !important;
#     background: #ffffff !important;
#     box-shadow: inset 0 1px 2px rgba(40, 61, 135, 0.03) !important;
# }

# .section-card textarea:focus,
# .section-card input:focus {
#     border-color: #6f8af1 !important;
#     box-shadow: 0 0 0 4px rgba(77, 111, 242, 0.10) !important;
# }

# .section-card textarea::placeholder,
# .section-card input::placeholder {
#     color: #8290ab !important;
#     opacity: 1 !important;
# }

# .primary-action,
# .primary-action button {
#     min-height: 48px !important;
#     border: none !important;
#     border-radius: 16px !important;
#     color: #ffffff !important;
#     background: linear-gradient(135deg, #2f49c7 0%, #4d6ff2 60%, #7185ff 100%) !important;
#     box-shadow: 0 12px 26px rgba(47, 73, 199, 0.24) !important;
#     font-weight: 800 !important;
#     transition: transform 0.18s ease, box-shadow 0.18s ease !important;
# }

# .primary-action:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 16px 34px rgba(47, 73, 199, 0.30) !important;
# }

# .secondary-action,
# .secondary-action button {
#     min-height: 48px !important;
#     border: 1px solid #c5d1f5 !important;
#     border-radius: 16px !important;
#     color: #2f49c7 !important;
#     background: linear-gradient(180deg, #f8faff, #eaf0ff) !important;
#     box-shadow: 0 8px 18px rgba(64, 89, 170, 0.08) !important;
#     font-weight: 800 !important;
# }

# .gradio-container .block {
#     color: var(--text-main) !important;
# }

# .gradio-container .wrap,
# .gradio-container .panel,
# .gradio-container .form {
#     background: transparent !important;
# }

# .gradio-container .label-wrap {
#     color: #354462 !important;
# }

# .gradio-container table,
# .gradio-container .table-wrap {
#     border-radius: 16px !important;
#     overflow: hidden;
# }

# footer {
#     display: none !important;
# }

# .footer-note {
#     margin: 20px 0 5px;
#     padding: 15px;
#     color: #61708c !important;
#     text-align: center;
#     font-size: 13px;
# }


# /* Calm NLP-inspired motion */
# .hero-visuals {
#     position: absolute;
#     inset: 0;
#     z-index: 1;
#     overflow: hidden;
#     pointer-events: none;
# }

# .floating-node {
#     position: absolute;
#     display: grid;
#     place-items: center;
#     width: 46px;
#     height: 46px;
#     border: 1px solid rgba(255, 255, 255, 0.30);
#     border-radius: 15px;
#     color: rgba(255, 255, 255, 0.90);
#     background: rgba(255, 255, 255, 0.12);
#     backdrop-filter: blur(9px);
#     box-shadow: 0 9px 25px rgba(25, 43, 126, 0.16);
#     font-size: 20px;
#     animation: floatNode 7s ease-in-out infinite;
# }

# .floating-node.node-one {
#     top: 24px;
#     right: 26%;
#     animation-delay: 0s;
# }

# .floating-node.node-two {
#     top: 104px;
#     right: 8%;
#     width: 38px;
#     height: 38px;
#     border-radius: 13px;
#     font-size: 16px;
#     animation-delay: -2.4s;
# }

# .floating-node.node-three {
#     bottom: 32px;
#     right: 20%;
#     width: 42px;
#     height: 42px;
#     border-radius: 14px;
#     font-size: 18px;
#     animation-delay: -4.8s;
# }

# .signal-line {
#     position: absolute;
#     right: 12%;
#     top: 50%;
#     width: 180px;
#     height: 80px;
#     opacity: 0.45;
#     transform: translateY(-50%);
# }

# .signal-line span {
#     position: absolute;
#     bottom: 10px;
#     width: 5px;
#     border-radius: 999px;
#     background: rgba(255, 255, 255, 0.72);
#     animation: signalPulse 2.8s ease-in-out infinite;
# }

# .signal-line span:nth-child(1) { left: 0; height: 18px; animation-delay: 0s; }
# .signal-line span:nth-child(2) { left: 18px; height: 38px; animation-delay: -0.2s; }
# .signal-line span:nth-child(3) { left: 36px; height: 27px; animation-delay: -0.4s; }
# .signal-line span:nth-child(4) { left: 54px; height: 56px; animation-delay: -0.6s; }
# .signal-line span:nth-child(5) { left: 72px; height: 34px; animation-delay: -0.8s; }
# .signal-line span:nth-child(6) { left: 90px; height: 64px; animation-delay: -1s; }
# .signal-line span:nth-child(7) { left: 108px; height: 43px; animation-delay: -1.2s; }
# .signal-line span:nth-child(8) { left: 126px; height: 28px; animation-delay: -1.4s; }
# .signal-line span:nth-child(9) { left: 144px; height: 49px; animation-delay: -1.6s; }
# .signal-line span:nth-child(10) { left: 162px; height: 22px; animation-delay: -1.8s; }

# .feature-card {
#     transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
# }

# .feature-card:hover {
#     transform: translateY(-4px);
#     border-color: rgba(77, 111, 242, 0.32);
#     box-shadow: 0 17px 38px rgba(45, 70, 160, 0.13);
# }

# .team-card {
#     transition: transform 0.24s ease, background 0.24s ease;
# }

# .team-card:hover {
#     transform: translateY(-3px);
#     background: linear-gradient(
#         145deg,
#         rgba(255, 255, 255, 0.23),
#         rgba(255, 255, 255, 0.13)
#     );
# }

# .app-tabs {
#     animation: panelEnter 0.65s ease both;
# }

# .feature-strip {
#     animation: panelEnter 0.55s ease both;
# }

# @keyframes floatNode {
#     0%, 100% {
#         transform: translate3d(0, 0, 0) rotate(0deg);
#     }
#     50% {
#         transform: translate3d(0, -10px, 0) rotate(3deg);
#     }
# }

# @keyframes signalPulse {
#     0%, 100% {
#         transform: scaleY(0.58);
#         opacity: 0.42;
#     }
#     50% {
#         transform: scaleY(1);
#         opacity: 0.90;
#     }
# }

# @keyframes panelEnter {
#     from {
#         opacity: 0;
#         transform: translateY(10px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# @media (prefers-reduced-motion: reduce) {
#     *,
#     *::before,
#     *::after {
#         animation-duration: 0.01ms !important;
#         animation-iteration-count: 1 !important;
#         scroll-behavior: auto !important;
#         transition-duration: 0.01ms !important;
#     }
# }

# @media (max-width: 760px) {
#     .app-shell {
#         padding: 4px 2px 18px;
#     }

#     .hero {
#         min-height: auto;
#         padding: 28px 22px;
#         border-radius: 24px;
#     }

#     .hero h1 {
#         font-size: 34px;
#     }

#     .team-grid,
#     .feature-strip {
#         grid-template-columns: 1fr;
#     }

#     .team-card {
#         text-align: left;
#     }

#     .app-tabs {
#         border-radius: 21px !important;
#     }

#     .section-card {
#         padding: 10px !important;
#     }
# }
# """

# with gr.Blocks(
#     title="LSTM Sentiment Analysis",
#     theme=gr.themes.Soft(),
#     css=CUSTOM_CSS,
# ) as app:

#     with gr.Column(elem_classes=["app-shell"]):

#         gr.HTML(
#             """
#             <section class="hero">
#                 <div class="hero-visuals" aria-hidden="true">
#                     <div class="floating-node node-one">Aa</div>
#                     <div class="floating-node node-two">✦</div>
#                     <div class="floating-node node-three">⌁</div>
#                     <div class="signal-line">
#                         <span></span><span></span><span></span><span></span><span></span>
#                         <span></span><span></span><span></span><span></span><span></span>
#                     </div>
#                 </div>
#                 <div class="hero-content">
#                     <h1>Sentiment Analysis & Smart Summarization</h1>
#                     <p>
#                         Analyze English product and food reviews using a trained
#                         LSTM model, generate concise summaries with DistilBART,
#                         and explore dataset insights through a clean interactive dashboard.
#                     </p>
#                 </div>

#                 <div class="team-grid">
#                     <div class="team-card">
#                         <span>Team Member</span>
#                         SAIF KHATTAB
#                     </div>
#                     <div class="team-card">
#                         <span>Team Member</span>
#                         YAZEED AMEEN
#                     </div>
#                     <div class="team-card">
#                         <span>Team Member</span>
#                         ASEEL MANNOUN
#                     </div>
#                 </div>
#             </section>
#             """
#         )

#         gr.HTML(
#             """
#             <div class="feature-strip">
#                 <div class="feature-card">
#                     <strong>🧠 LSTM Classification</strong>
#                     <small>Predicts positive, neutral, or negative sentiment.</small>
#                 </div>
#                 <div class="feature-card">
#                     <strong>📝 Smart Summarization</strong>
#                     <small>Creates a concise summary for longer reviews.</small>
#                 </div>
#                 <div class="feature-card">
#                     <strong>📊 Dataset Insights</strong>
#                     <small>Search reviews and visualize sentiment distribution.</small>
#                 </div>
#             </div>
#             """
#         )

#         with gr.Tabs(elem_classes=["app-tabs"]):

#             with gr.Tab("📝 Analyze Review"):
#                 with gr.Column(elem_classes=["section-card"]):
#                     gr.Markdown(
#                         "### Analyze a Review\n"
#                         "Enter an English review to generate its summary, sentiment, "
#                         "confidence score, and processing stages."
#                     )

#                     input_text = gr.Textbox(
#                         lines=7,
#                         label="Review Text",
#                         placeholder="Write or paste an English food or product review...",
#                     )

#                     analyze_button = gr.Button(
#                         "Analyze Review",
#                         variant="primary",
#                         elem_classes=["primary-action"],
#                     )

#                     with gr.Row(equal_height=True):
#                         summary_output = gr.Textbox(
#                             label="Generated Summary",
#                             lines=5,
#                         )
#                         sentiment_output = gr.Textbox(
#                             label="Predicted Sentiment",
#                             lines=5,
#                         )

#                     probability_output = gr.Label(
#                         label="Class Probabilities",
#                         num_top_classes=3,
#                     )

#                     with gr.Accordion(
#                         "🔍 Text Processing Stages",
#                         open=False,
#                     ):
#                         gr.Markdown(
#                             "These fields show how the review is transformed "
#                             "before it reaches the LSTM model."
#                         )
#                         raw_tokens_output = gr.Textbox(label="1. Raw Tokens")
#                         no_punctuation_output = gr.Textbox(label="2. No Punctuation")
#                         no_stop_words_output = gr.Textbox(label="3. No Stop Words")
#                         lemmatized_output = gr.Textbox(label="4. Lemmatized")

#                     analyze_button.click(
#                         fn=web_interface_fn,
#                         inputs=input_text,
#                         outputs=[
#                             summary_output,
#                             sentiment_output,
#                             probability_output,
#                             raw_tokens_output,
#                             no_punctuation_output,
#                             no_stop_words_output,
#                             lemmatized_output,
#                         ],
#                     )

#             with gr.Tab("🔎 Search Dataset"):
#                 with gr.Column(elem_classes=["section-card"]):
#                     gr.Markdown(
#                         "### Search Reviews\n"
#                         "Search the dataset by a word or feature and view the closest results."
#                     )

#                     with gr.Row(equal_height=True):
#                         with gr.Column(scale=2):
#                             query_input = gr.Textbox(
#                                 label="Search Query",
#                                 placeholder="Examples: coffee, taste, quality, price",
#                             )
#                             search_button = gr.Button(
#                                 "Search Dataset",
#                                 elem_classes=["secondary-action"],
#                             )

#                         with gr.Column(scale=1):
#                             history_output = gr.Textbox(
#                                 label="Recent Searches",
#                                 lines=5,
#                                 interactive=False,
#                             )

#                     results_output = gr.Textbox(
#                         label="Matching Reviews",
#                         lines=12,
#                     )

#                     search_button.click(
#                         fn=search_in_dataset_with_history,
#                         inputs=query_input,
#                         outputs=[results_output, history_output],
#                     )

#             with gr.Tab("📊 Analytics Dashboard"):
#                 with gr.Column(elem_classes=["section-card"]):
#                     gr.Markdown(
#                         "### Sentiment Dashboard\n"
#                         "Filter the dataset by a feature to view its sentiment distribution."
#                     )

#                     with gr.Row(equal_height=True):
#                         dashboard_query = gr.Textbox(
#                             label="Feature Filter",
#                             placeholder="Examples: coffee, food, tea, quality",
#                             scale=3,
#                         )
#                         dashboard_button = gr.Button(
#                             "Generate Chart",
#                             elem_classes=["primary-action"],
#                             scale=1,
#                         )

#                     plot_output = gr.Plot(label="Sentiment Distribution")

#                     dashboard_button.click(
#                         fn=generate_dashboard,
#                         inputs=dashboard_query,
#                         outputs=plot_output,
#                     )

#         gr.HTML(
#             """
#             <div class="footer-note">
#                 Built with TensorFlow, Keras, Transformers, and Gradio.
#             </div>
#             """
#         )


# if __name__ == "__main__":
#     app.launch()
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


def generate_dashboard(query):
    if df is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5, 0.5,
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
        title = f"Sentiment Distribution for: {query.strip().capitalize()}"
    else:
        filtered_df = df
        title = "General Sentiment Distribution"

    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5, 0.5,
            "No data found for this feature.",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.axis("off")
        return fig

    counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=140,
        ax=ax,
        textprops={"fontsize": 12, "weight": "bold"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("")
    plt.tight_layout()

    return fig


# =========================================================
# Gradio interface
# =========================================================

CUSTOM_CSS = """
:root {
    --bg-top: #eef4ff;
    --bg-bottom: #f9fbff;
    --surface: rgba(255, 255, 255, 0.92);
    --surface-strong: #ffffff;
    --surface-blue: #f2f6ff;
    --text-main: #17213d;
    --text-soft: #53617f;
    --primary: #4d6ff2;
    --primary-deep: #2f49c7;
    --accent: #7185ff;
    --border: rgba(111, 137, 220, 0.22);
    --shadow-main: 0 24px 65px rgba(45, 70, 160, 0.15);
    --shadow-soft: 0 12px 30px rgba(45, 70, 160, 0.09);
}

* {
    box-sizing: border-box;
}

.gradio-container {
    position: relative;
    min-height: 100vh;
    color: var(--text-main) !important;
    background:
        radial-gradient(circle at 10% 8%, rgba(101, 126, 245, 0.16), transparent 27%),
        radial-gradient(circle at 92% 15%, rgba(154, 170, 255, 0.14), transparent 25%),
        linear-gradient(180deg, var(--bg-top) 0%, #f6f9ff 48%, var(--bg-bottom) 100%) !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.42;
    background-image:
        linear-gradient(rgba(77, 111, 242, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(77, 111, 242, 0.025) 1px, transparent 1px);
    background-size: 34px 34px;
}

.app-shell {
    position: relative;
    z-index: 1;
    max-width: 1180px;
    margin: 0 auto;
    padding: 8px 4px 24px;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    border: 1px solid rgba(255, 255, 255, 0.42);
    border-radius: 30px;
    padding: 42px 40px 34px;
    margin: 8px 0 24px;
    color: #ffffff;
    background:
        radial-gradient(circle at 78% 25%, rgba(255, 255, 255, 0.18), transparent 22%),
        linear-gradient(135deg, #2f49c7 0%, #4667e8 50%, #6d78f6 100%);
    box-shadow: 0 28px 70px rgba(47, 73, 199, 0.30);
}

.hero::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    top: -120px;
    right: -80px;
    border-radius: 48% 52% 62% 38% / 50% 42% 58% 50%;
    border: 1px solid rgba(255, 255, 255, 0.24);
    background: rgba(255, 255, 255, 0.08);
    transform: rotate(18deg);
}

.hero::after {
    content: "";
    position: absolute;
    width: 200px;
    height: 200px;
    left: -55px;
    bottom: -110px;
    border-radius: 58% 42% 46% 54% / 44% 55% 45% 56%;
    background: rgba(255, 255, 255, 0.09);
    transform: rotate(-12deg);
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 820px;
}

.hero h1 {
    margin: 0 0 14px;
    color: #ffffff !important;
    font-size: clamp(34px, 4.4vw, 52px);
    line-height: 1.05;
    letter-spacing: -0.035em;
    text-shadow: 0 3px 18px rgba(20, 34, 100, 0.24);
}

.hero p {
    max-width: 800px;
    margin: 0;
    color: rgba(255, 255, 255, 0.92) !important;
    font-size: 16px;
    line-height: 1.75;
    font-weight: 500;
}

.team-grid {
    position: relative;
    z-index: 2;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 30px;
}

.team-card {
    position: relative;
    min-width: 0;
    overflow: hidden;
    padding: 17px 18px;
    border: 1px solid rgba(255, 255, 255, 0.30);
    border-radius: 19px;
    color: #ffffff !important;
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.19), rgba(255, 255, 255, 0.10));
    backdrop-filter: blur(12px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.22);
    font-weight: 800;
    text-align: center;
}

.team-card::before {
    content: "";
    position: absolute;
    width: 70px;
    height: 70px;
    top: -38px;
    right: -25px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.10);
}

.team-card span {
    display: block;
    margin-bottom: 5px;
    color: #dce6ff !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.feature-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 15px;
    margin-bottom: 22px;
}

.feature-card {
    position: relative;
    overflow: hidden;
    min-height: 118px;
    padding: 20px 20px 18px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(244, 248, 255, 0.92));
    box-shadow: var(--shadow-soft);
}

.feature-card::after {
    content: "";
    position: absolute;
    width: 90px;
    height: 90px;
    right: -38px;
    bottom: -46px;
    border-radius: 50%;
    background: rgba(77, 111, 242, 0.07);
}

.feature-card strong {
    position: relative;
    z-index: 1;
    display: block;
    margin-bottom: 7px;
    color: #25345f !important;
    font-size: 15px;
}

.feature-card small {
    position: relative;
    z-index: 1;
    color: var(--text-soft) !important;
    font-size: 13px;
    line-height: 1.6;
}

.app-tabs {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    border-radius: 26px !important;
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(14px);
    box-shadow: var(--shadow-main) !important;
    padding: 10px !important;
}

.app-tabs::before {
    content: "";
    position: absolute;
    width: 170px;
    height: 170px;
    right: -70px;
    top: -90px;
    border-radius: 50%;
    background: rgba(99, 123, 236, 0.06);
    pointer-events: none;
}

.app-tabs button[role="tab"] {
    min-height: 46px;
    border-radius: 14px !important;
    color: #415177 !important;
    font-weight: 750 !important;
    transition: all 0.2s ease !important;
}

.app-tabs button[role="tab"]:hover {
    color: #2f49c7 !important;
    background: #eef3ff !important;
}

.app-tabs button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #2f49c7, #5576ed) !important;
    box-shadow: 0 8px 20px rgba(47, 73, 199, 0.22) !important;
}

.section-card {
    border: 1px solid rgba(115, 137, 207, 0.17) !important;
    border-radius: 22px !important;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.96)) !important;
    box-shadow: 0 10px 28px rgba(52, 75, 151, 0.07) !important;
    padding: 16px !important;
}

.section-card h1,
.section-card h2,
.section-card h3 {
    color: #1c294b !important;
    letter-spacing: -0.015em;
}

.section-card p,
.section-card label,
.section-card span {
    color: #405171 !important;
}

.section-card textarea,
.section-card input {
    min-height: 48px;
    border: 1px solid #d4ddf3 !important;
    border-radius: 16px !important;
    color: #17213d !important;
    background: #ffffff !important;
    box-shadow: inset 0 1px 2px rgba(40, 61, 135, 0.03) !important;
}

.section-card textarea:focus,
.section-card input:focus {
    border-color: #6f8af1 !important;
    box-shadow: 0 0 0 4px rgba(77, 111, 242, 0.10) !important;
}

.section-card textarea::placeholder,
.section-card input::placeholder {
    color: #8290ab !important;
    opacity: 1 !important;
}

.primary-action,
.primary-action button {
    min-height: 48px !important;
    border: none !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #2f49c7 0%, #4d6ff2 60%, #7185ff 100%) !important;
    box-shadow: 0 12px 26px rgba(47, 73, 199, 0.24) !important;
    font-weight: 800 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

.primary-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(47, 73, 199, 0.30) !important;
}

.secondary-action,
.secondary-action button {
    min-height: 48px !important;
    border: 1px solid #c5d1f5 !important;
    border-radius: 16px !important;
    color: #2f49c7 !important;
    background: linear-gradient(180deg, #f8faff, #eaf0ff) !important;
    box-shadow: 0 8px 18px rgba(64, 89, 170, 0.08) !important;
    font-weight: 800 !important;
}

.gradio-container .block {
    color: var(--text-main) !important;
}

.gradio-container .wrap,
.gradio-container .panel,
.gradio-container .form {
    background: transparent !important;
}

.gradio-container .label-wrap {
    color: #354462 !important;
}

.gradio-container table,
.gradio-container .table-wrap {
    border-radius: 16px !important;
    overflow: hidden;
}

footer {
    display: none !important;
}

.footer-note {
    margin: 20px 0 5px;
    padding: 15px;
    color: #61708c !important;
    text-align: center;
    font-size: 13px;
}


/* Calm NLP-inspired motion */
.hero-visuals {
    position: absolute;
    inset: 0;
    z-index: 1;
    overflow: hidden;
    pointer-events: none;
}

.floating-node {
    position: absolute;
    display: grid;
    place-items: center;
    width: 46px;
    height: 46px;
    border: 1px solid rgba(255, 255, 255, 0.30);
    border-radius: 15px;
    color: rgba(255, 255, 255, 0.90);
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(9px);
    box-shadow: 0 9px 25px rgba(25, 43, 126, 0.16);
    font-size: 20px;
    animation: floatNode 7s ease-in-out infinite;
}

.floating-node.node-one {
    top: 24px;
    right: 26%;
    animation-delay: 0s;
}

.floating-node.node-two {
    top: 104px;
    right: 8%;
    width: 38px;
    height: 38px;
    border-radius: 13px;
    font-size: 16px;
    animation-delay: -2.4s;
}

.floating-node.node-three {
    bottom: 32px;
    right: 20%;
    width: 42px;
    height: 42px;
    border-radius: 14px;
    font-size: 18px;
    animation-delay: -4.8s;
}

.signal-line {
    position: absolute;
    right: 12%;
    top: 50%;
    width: 180px;
    height: 80px;
    opacity: 0.45;
    transform: translateY(-50%);
}

.signal-line span {
    position: absolute;
    bottom: 10px;
    width: 5px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.72);
    animation: signalPulse 2.8s ease-in-out infinite;
}

.signal-line span:nth-child(1) { left: 0; height: 18px; animation-delay: 0s; }
.signal-line span:nth-child(2) { left: 18px; height: 38px; animation-delay: -0.2s; }
.signal-line span:nth-child(3) { left: 36px; height: 27px; animation-delay: -0.4s; }
.signal-line span:nth-child(4) { left: 54px; height: 56px; animation-delay: -0.6s; }
.signal-line span:nth-child(5) { left: 72px; height: 34px; animation-delay: -0.8s; }
.signal-line span:nth-child(6) { left: 90px; height: 64px; animation-delay: -1s; }
.signal-line span:nth-child(7) { left: 108px; height: 43px; animation-delay: -1.2s; }
.signal-line span:nth-child(8) { left: 126px; height: 28px; animation-delay: -1.4s; }
.signal-line span:nth-child(9) { left: 144px; height: 49px; animation-delay: -1.6s; }
.signal-line span:nth-child(10) { left: 162px; height: 22px; animation-delay: -1.8s; }

.feature-card {
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(77, 111, 242, 0.32);
    box-shadow: 0 17px 38px rgba(45, 70, 160, 0.13);
}

.team-card {
    transition: transform 0.24s ease, background 0.24s ease;
}

.team-card:hover {
    transform: translateY(-3px);
    background: linear-gradient(
        145deg,
        rgba(255, 255, 255, 0.23),
        rgba(255, 255, 255, 0.13)
    );
}

.app-tabs {
    animation: panelEnter 0.65s ease both;
}

.feature-strip {
    animation: panelEnter 0.55s ease both;
}

@keyframes floatNode {
    0%, 100% {
        transform: translate3d(0, 0, 0) rotate(0deg);
    }
    50% {
        transform: translate3d(0, -10px, 0) rotate(3deg);
    }
}

@keyframes signalPulse {
    0%, 100% {
        transform: scaleY(0.58);
        opacity: 0.42;
    }
    50% {
        transform: scaleY(1);
        opacity: 0.90;
    }
}

@keyframes panelEnter {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}

@media (max-width: 760px) {
    .app-shell {
        padding: 4px 2px 18px;
    }

    .hero {
        min-height: auto;
        padding: 28px 22px;
        border-radius: 24px;
    }

    .hero h1 {
        font-size: 34px;
    }

    .team-grid,
    .feature-strip {
        grid-template-columns: 1fr;
    }

    .team-card {
        text-align: left;
    }

    .app-tabs {
        border-radius: 21px !important;
    }

    .section-card {
        padding: 10px !important;
    }
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
            <section class="hero">
                <div class="hero-visuals" aria-hidden="true">
                    <div class="floating-node node-one">Aa</div>
                    <div class="floating-node node-two">✦</div>
                    <div class="floating-node node-three">⌁</div>
                    <div class="signal-line">
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                </div>
                <div class="hero-content">
                    <h1>Sentiment Analysis & Smart Summarization</h1>
                    <p>
                        Analyze English product and food reviews using a trained
                        LSTM model, generate concise summaries with DistilBART,
                        and explore dataset insights through a clean interactive dashboard.
                    </p>
                </div>

                <div class="team-grid">
                    <div class="team-card">
                        <span>Team Member</span>
                        SAIF KHATTAB
                    </div>
                    <div class="team-card">
                        <span>Team Member</span>
                        YAZEED AMEEN
                    </div>
                    <div class="team-card">
                        <span>Team Member</span>
                        ASEEL MANNOUN
                    </div>
                </div>
            </section>
            """
        )

        gr.HTML(
            """
            <div class="feature-strip">
                <div class="feature-card">
                    <strong>🧠 LSTM Classification</strong>
                    <small>Predicts positive, neutral, or negative sentiment.</small>
                </div>
                <div class="feature-card">
                    <strong>📝 Smart Summarization</strong>
                    <small>Creates a concise summary for longer reviews.</small>
                </div>
                <div class="feature-card">
                    <strong>📊 Dataset Insights</strong>
                    <small>Search reviews and visualize sentiment distribution.</small>
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

                    plot_output = gr.Plot(label="Sentiment Distribution")

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