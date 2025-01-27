import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.markdown(
    """
    <style>
    body {
        color: white;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #001f4d;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffd700 !important;
        text-shadow: 2px 2px 4px #000000;
    }
    .css-1d391kg { /* Untuk tombol */
        background-color: white;
        color: #000080;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 2px 2px 5px #000000;
    }
    .css-1d391kg:hover {
        background-color: #0056b3;
        color: white;
    }
    .css-10trblm.e16nr0p30 {
        color: #0000;
    }
    footer {
        background-color: #ffd700;
        color: black;
        text-align: right;
        padding: 10px;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        box-shadow: 0px -1px 5px rgba(0, 0, 0, 0.1);
    }
    footer p {
        padding-right: 70px;
        margin: 0;
    }
    header {
        color: white;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

data = pd.read_csv("Classification.csv")

st.markdown("""
<header>
    <h1>Aplikasi Klasifikasi Obat Menggunakan KNN</h1>
</header>
""", unsafe_allow_html=True)

st.markdown("""
Aplikasi ini menggunakan metode **K-Nearest Neighbors (KNN)** untuk melakukan klasifikasi data. 
Dataset berisi informasi tentang pasien, termasuk usia, jenis kelamin, tekanan darah, kolesterol, rasio Sodium/Kalium, dan obat yang diresepkan.

Fitur-fitur aplikasi:
1. **Visualisasi Heatmap Korelasi**
2. **Diagram Distribusi Data**
3. **Implementasi KNN dengan Parameter Interaktif**
4. **Evaluasi Model**
5. **Prediksi Obat untuk Kasus Baru**
""")

with st.expander('Data'):
    st.write('Data Mentah')
    data = pd.read_csv('Classification.csv')
    data
    
st.subheader("Dataset Preview")
st.dataframe(data.head())

le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
data['BP'] = le_bp.fit_transform(data['BP'])
data['Cholesterol'] = le_chol.fit_transform(data['Cholesterol'])
data['Drug'] = le_drug.fit_transform(data['Drug'])

st.subheader("Heatmap Korelasi")
plt.figure(figsize=(10, 6))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

st.subheader("Distribusi Data")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data['Age'], bins=15, kde=True, ax=ax[0], color="blue")
ax[0].set_title("Distribusi Usia")
sns.histplot(data['Na_to_K'], bins=15, kde=True, ax=ax[1], color="green")
ax[1].set_title("Distribusi Rasio Na_to_K")
st.pyplot(fig)

X = data.drop('Drug', axis=1)
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def knn_model(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

st.sidebar.header("Parameter KNN")
k_value = st.sidebar.slider("Pilih K (Jumlah Tetangga):", min_value=1, max_value=20, value=5)

st.subheader("Klasifikasi dengan KNN")
y_pred, model = knn_model(k_value)

st.text("Laporan Klasifikasi")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matriks")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le_drug.classes_, yticklabels=le_drug.classes_)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(plt)

st.success("Model berhasil dijalankan! Sesuaikan nilai K untuk melihat pengaruhnya.")

st.subheader("Prediksi Obat untuk Kasus Baru")
age = st.number_input("Usia", min_value=0, max_value=120, value=30, step=1)
sex = st.selectbox("Jenis Kelamin", options=le_sex.classes_)
bp = st.selectbox("Tekanan Darah", options=le_bp.classes_)
cholesterol = st.selectbox("Kolesterol", options=le_chol.classes_)
na_to_k = st.number_input("Rasio Na_to_K (Sodium/Kalium)", min_value=0.0, value=15.0, step=0.1)

try:
    new_case = pd.DataFrame({
        'Age': [age],
        'Sex': [le_sex.transform([sex])[0]],
        'BP': [le_bp.transform([bp])[0]],
        'Cholesterol': [le_chol.transform([cholesterol])[0]],
        'Na_to_K': [na_to_k]
    })

    new_case_scaled = scaler.transform(new_case)

    if st.button("Prediksi Obat"):
        prediction = model.predict(new_case_scaled)
        predicted_drug = le_drug.inverse_transform(prediction)[0]
        st.success(f"Obat yang diprediksi untuk kasus ini adalah: {predicted_drug}")

except ValueError as e:
    st.error(f"Kesalahan pada input: {e}")

st.markdown("""
<footer>
    <p>Developed by <strong>Putri Nasywa Nabilla</strong> | 2025</p>
</footer>
""", unsafe_allow_html=True)
