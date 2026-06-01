import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # تم التعديل هنا
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as ui
import matplotlib.pyplot as plt
from thefuzz import process

# 1. تنزيل الحزم الأساسية من nltk للسيرفر السحابي
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# إعداد أدوات تنظيف النص
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "never"}
stop_words = stop_words.difference(negation_words)
lemmatizer = WordNetLemmatizer()

# 2. تحميل البيانات (تأكد أن ملف Reviews.csv مرفوع في المساحة)
print("جاري تحميل البيانات...")
df = pd.read_csv('Reviews.csv', nrows=100000)
df = df[['Text', 'Summary', 'Score']].dropna()

# 3. دوال المعالجة
def map_score_to_sentiment(score):
    if score in [1, 2]: return 'negative'
    elif score == 3: return 'neutral'
    else: return 'positive'

df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)

def clean_html_tags(text):
    cleaned_text = re.sub(r'<.*?>', ' ', str(text))
    return ' '.join(cleaned_text.split())

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("جاري تنظيف ومعالجة النصوص...")
df['Cleaned_Text'] = df['Text'].apply(clean_html_tags)
df['Processed_Text'] = df['Cleaned_Text'].apply(preprocess_text)

# 4. بناء وتدريب نموذج Logistic Regression
print("جاري تدريب نموذج تحليل المشاعر...")
X = df['Processed_Text']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

# تم التعديل هنا لتدريب الموديل الجديد
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. تحميل نموذج التلخيص (Transformers)
print("جاري تحميل نموذج التلخيص...")
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def analyze_and_summarize_review(raw_text):
#     cleaned = clean_html_tags(raw_text)
#     processed = preprocess_text(cleaned)
    
#     text_tfidf = tfidf.transform([processed])
#     sentiment_prediction = model.predict(text_tfidf)[0]
    
#     if len(raw_text.split()) > 10:
#         inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
#         summary_ids = sum_model.generate(inputs["input_ids"], max_length=30, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
#         summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     else:
#         summary_text = "النص قصير جداً ولا يحتاج إلى تلخيص."
        
#     return {
#         "الملخص الذكي (Summary)": summary_text,
#         "تحليل المشاعر (Sentiment)": sentiment_prediction
#     }
def analyze_and_summarize_review(raw_text):
    cleaned = clean_html_tags(raw_text)
    processed = preprocess_text(cleaned)
    
    # 1. تحليل المشاعر
    text_tfidf = tfidf.transform([processed])
    sentiment_prediction = model.predict(text_tfidf)[0]
    
    # 2. حساب عدد كلمات النص المدخل
    word_count = len(raw_text.split())
    
    # الشرط الحاسم الذي طلبته: إذا كان أكبر من 15 كلمة يلخص، غير ذلك يطبع الرسالة
    if word_count > 10:
        inputs = tokenizer([raw_text], max_length=1024, return_tensors="pt", truncation=True)
        
        # بارامترات صارمة تمنع النموذج من زيادة أي كلام أو تأليف
        summary_ids = sum_model.generate(
            inputs["input_ids"], 
            max_new_tokens=20,           # حد أقصى صغير جداً للملخص ليجبره على الاختصار الشديد
            no_repeat_ngram_size=2,      # يمنع تكرار العبارات نهائياً
            num_beams=2,                 # تقليل عدد الاحتمالات لمنع التخريف
            early_stopping=True
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # حماية إضافية: لو الموديل هلوَس برضه وطلع التلخيص أطول أو مساوي للنص الأصلي، يعرض النص الأصلي فقط
        if len(summary_text.split()) >= word_count:
            summary_text = raw_text.strip()
    else:
        # إذا كان 15 كلمة أو أقل، يطبع هذه الرسالة مباشرة ولا يشغل نموذج التلخيص
        summary_text = "The text is too short to require summarization. / النص قصير جداً ولا يحتاج إلى تلخيص."
        
    return {
        "الملخص الذكي (Summary)": summary_text,
        "تحليل المشاعر (Sentiment)": sentiment_prediction
    }

# 6. إعداد دوال واجهة Gradio
search_history = []

def generate_dashboard(query):
    if query and query.strip():
        filtered_df = df[df['Processed_Text'].str.contains(query.lower(), case=False, na=False)]
        title = f"Sentiment Distribution for: {query.capitalize()}"
    else:
        filtered_df = df
        title = "General Sentiment Distribution"
        
    if filtered_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data found for this feature', ha='center', fontsize=12)
        ax.axis('off')
        return fig
        
    sentiment_counts = filtered_df['Sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#50fa7b', '#ff5555', '#8be9fd']
    
    sentiment_counts.plot(
        kind='pie', 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        ax=ax,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('') 
    plt.tight_layout()
    return fig

def web_interface_fn(review_text):
    if not review_text.strip():
        return "Please enter a review.", "", "", "", "", ""
    
    tokens_raw = word_tokenize(review_text.lower())
    text_no_punct = review_text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens_no_punct = word_tokenize(text_no_punct)
    no_stop_words = [w for w in tokens_no_punct if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in no_stop_words]
    
    results = analyze_and_summarize_review(review_text) 
    
    return (
        results["الملخص الذكي (Summary)"], 
        results["تحليل المشاعر (Sentiment)"].upper() + (" 😊" if results["تحليل المشاعر (Sentiment)"] == 'positive' else " 😢"),
        " | ".join(tokens_raw),
        " | ".join(tokens_no_punct),
        " | ".join(no_stop_words),
        " | ".join(lemmatized)
    )

def search_in_dataset_with_history(query):
    global search_history
    if not query.strip(): return "Please enter a search term.", "\n".join(search_history)
    
    common_keywords = ['battery', 'screen', 'camera', 'price', 'performance', 'design', 'software', 'quality', 'money']
    match = process.extractOne(query.lower(), common_keywords)
    actual_query = match[0] if match and match[1] > 80 else query.lower().strip()

    if query not in search_history:
        search_history.insert(0, query)
        search_history = search_history[:5]
    
    results = df[df['Text'].str.contains(actual_query, case=False, na=False)]
    
    output = f"🔍 Found {len(results)} reviews for '{actual_query}'.\n\n"
    for idx, row in results.head(5).iterrows():
        output += f"📝: {str(row['Text'])[:120]}...\n📊: {str(row['Sentiment']).upper()}\n\n"
        
    return output, "\n".join(search_history)

# 7. بناء وتشغيل واجهة Gradio
with ui.Blocks(title="NLP Project", theme=ui.themes.Soft()) as app:
    ui.Markdown("""
    <div style="text-align: center; background-color: #1e1e2e; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #ff79c6;">💻 NLP Project</h1>
        <div style="display: flex; justify-content: space-around; color: white; font-size: 16px;">
            <div>👤 SAIF KHATTAB <br> 🆔 202312556</div>
            <div>👤 YAZEED AMEEN <br> 🆔 202310058</div>
        </div>
    </div>
    """)
    
    with ui.Tabs():
        with ui.TabItem("Analyze Text 📝"):
            input_text = ui.Textbox(lines=3, label="Enter Review")
            btn = ui.Button("Analyze 🚀", variant="primary")
            
            with ui.Row():
                out1 = ui.Textbox(label="Summary")
                out2 = ui.Textbox(label="Sentiment")
            
            with ui.Accordion("Text Processing Stages (Transparency Mode)", open=False):
                out_tokens = ui.Textbox(label="1. Raw Tokens")
                out_no_punct = ui.Textbox(label="1.5 No Punctuation")
                out_no_stop = ui.Textbox(label="2. No Stop Words")
                out_lemmatized = ui.Textbox(label="3. Lemmatized")
            
            btn.click(web_interface_fn, input_text, [out1, out2, out_tokens, out_no_punct, out_no_stop, out_lemmatized])
            
        with ui.TabItem("Search Dataset 🔍"):
            with ui.Row():
                with ui.Column():
                    query_input = ui.Textbox(label="Search Query")
                    search_btn = ui.Button("Search Data 🔎")
                history_output = ui.Textbox(label="Last 5 Searches", interactive=False)
            results_out = ui.Textbox(label="Results", lines=8)
            search_btn.click(search_in_dataset_with_history, query_input, [results_out, history_output])
            
        with ui.TabItem("Analytics Dashboard 📊"):
            dash_query = ui.Textbox(label="Filter stats by feature (Optional)", placeholder="e.g., food,tea")
            dashboard_btn = ui.Button("Show Chart 🚀")
            plot_output = ui.Plot(label="Sentiment Distribution")
            dashboard_btn.click(generate_dashboard, inputs=dash_query, outputs=plot_output)

# السطر الأخير بدون أي إضافات ليعمل على السحابة
app.launch()