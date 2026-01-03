import streamlit as st
import joblib
import re


@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_model()


def clean_text(text):
    text = text.lower()

    # gabung negasi
    text = re.sub(r'tidak\s+bagus', 'tidak_bagus', text)
    text = re.sub(r'tidak\s+sesuai', 'tidak_sesuai', text)
    text = re.sub(r'tidak\s+puas', 'tidak_puas', text)
    text = re.sub(r'tidak\s+rekomendasi', 'tidak_rekomendasi', text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_sentiment(text):
    text_clean = clean_text(text)
    vec = tfidf.transform([text_clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "POSITIF 😊" if pred == 1 else "NEGATIF 😡"
    confidence = prob[pred]

    return label, confidence


st.set_page_config(page_title="Analisis Sentimen", page_icon="📝", layout="centered")

st.title("📝 Analisis Sentimen Ulasan Produk Lazada")
st.write("Masukkan komentar ulasan produk untuk mengetahui sentimennya.")


user_input = st.text_area(
    "✍️ Tulis komentar:",
    placeholder="Contoh: barang rusak dan tidak sesuai deskripsi",
    height=150
)


if st.button("🔍 Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Komentar tidak boleh kosong!")
    else:
        label, confidence = predict_sentiment(user_input)

        st.subheader("📊 Hasil Analisis")
        st.success(f"**Sentimen:** {label}")
        st.info(f"**Confidence:** {confidence:.2%}")


st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression | Bahasa Indonesia")

