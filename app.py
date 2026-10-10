import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Meme Kanseri Tespit Sistemi",
    page_icon="🧬",
    layout="wide"
)

MODEL_PATH = "fixed_densenet121_meme_kanseri.keras"
IMG_SIZE = 128

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

st.markdown("""
<style>
.main {
    background-color: #f8fbff;
}
.title {
    font-size: 42px;
    font-weight: 800;
    color: #0b3d91;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #444;
}
.result-box {
    padding: 25px;
    border-radius: 18px;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Meme Kanseri Histopatolojik Görüntü Sınıflandırma Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fine-tuned DenseNet121 modeli ile kanserli / kansersiz görüntü tahmini</div>', unsafe_allow_html=True)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Görüntü Yükle")
    uploaded_file = st.file_uploader(
        "Histopatolojik görüntü seçiniz",
        type=["png", "jpg", "jpeg"]
    )

    st.info("""
    Bu sistem derin öğrenme tabanlı bir karar destek prototipidir. 
    Tıbbi tanı yerine geçmez.
    """)

with right:
    st.subheader("Model Bilgileri")
    st.write("""
    **Kullanılan Model:** Fine-tuned DenseNet121  
    **Görüntü Boyutu:** 128x128  
    **Sınıflar:** Kansersiz / Kanserli  
    **Final Accuracy:** %85.5  
    **Recall:** %89.8  
    **F1-score:** %86.1  
    """)

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)
    return input_image, resized

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    input_image, resized_image = preprocess_image(image)

    prediction = model.predict(input_image)[0][0]

    predicted_class = 1 if prediction > 0.5 else 0

    if predicted_class == 1:
        result = "Kanserli"
        confidence = prediction * 100
        color = "#d62828"
    else:
        result = "Kansersiz"
        confidence = (1 - prediction) * 100
        color = "#2a9d8f"

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Yüklenen Görüntü")
        st.image(image, caption="Orijinal Görüntü", use_container_width=True)

    with col2:
        st.subheader("Tahmin Sonucu")

        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:{color}; text-align:center;">{result}</h2>
            <h3 style="text-align:center;">Güven Oranı: %{confidence:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

        if predicted_class == 1:
            st.error("Model bu görüntüyü kanserli olarak sınıflandırdı.")
        else:
            st.success("Model bu görüntüyü kansersiz olarak sınıflandırdı.")

        st.write("Ham model çıktısı:", float(prediction))

    st.divider()

    st.subheader("Tahmin Olasılıkları")

    labels = ["Kansersiz", "Kanserli"]
    values = [(1 - prediction) * 100, prediction * 100]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Olasılık (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Tahmin Olasılıkları")

    st.pyplot(fig)

else:
    st.warning("Tahmin yapmak için bir görüntü yükleyiniz.")