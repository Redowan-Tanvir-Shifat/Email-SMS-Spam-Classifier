import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
# Download NLTK data (only needed once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.data.path.append("/root/nltk_data")  # Add this line
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 30px;
    }
    .text-area {
        border-radius: 10px;
        padding: 15px;
    }
    .predict-btn {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        margin: 10px 0;
        width: 100%;
    }
    .predict-btn:hover {
        background-color: #45a049;
    }
    .spam-result {
        color: #d32f2f;
        font-size: 24px;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffebee;
        margin-top: 20px;
    }
    .ham-result {
        color: #388e3c;
        font-size: 24px;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #777;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<h1 class="title">📧 Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)

# Sidebar for additional info
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses machine learning to classify messages as spam or not spam (ham).
    - Natural Language Processing (NLP) techniques
    - Uses TF-IDF vectorization
    - Predicts using Multinomial Naive Bayes model
    """)
    st.markdown("---")
    st.write("💡 Try pasting these examples:")
    st.code("WINNER! You've won a $1000 gift card!")
    st.code("Hey, can we meet tomorrow for lunch?")

# Main content
input_sms = st.text_area(
    "Enter your message below:",
    height=150,
    help="Paste or type the email/SMS message you want to check"
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button(
        '🔍 Analyze Message',
        key='predict',
        help='Click to check if the message is spam'
    )

if predict_btn:
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze")
    else:
        with st.spinner('Analyzing message...'):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # Display result with animation
            if result == 1:
                st.balloons()
                st.markdown('<div class="spam-result">🚨 SPAM ALERT!</div>', unsafe_allow_html=True)
                st.error("This message appears to be spam. Be cautious with any links or requests.")
            else:
                st.snow()
                st.markdown('<div class="ham-result">✅ Legitimate Message</div>', unsafe_allow_html=True)
                st.success("This message appears to be safe and not spam.")
        
        # Show processing details in expander
        with st.expander("Show processing details"):
            st.write("**Original message:**")
            st.write(input_sms)
            st.write("**After processing:**")
            st.write(transformed_sms)
            st.write("**Vectorized features:**")
            st.write(vector_input.shape)

# Footer
st.markdown("---")
st.markdown('<div class="footer">Spam Classifier App © 2025 | Made with Streamlit🔻</div>', unsafe_allow_html=True)