# 🏙️ İstanbul Kira Fiyatı Tahmin Modeli

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

İstanbul'daki kiralık konut ilanlarından derlenen veri seti üzerinde **Derin Öğrenme (Deep Learning)** kullanılarak geliştirilmiş bir kira fiyatı tahmin projesidir. Proje aynı zamanda interaktif bir **Streamlit** web uygulaması içermektedir.

---

## 📌 Proje Özeti

| Başlık | Detay |
|---|---|
| **Veri Kaynağı** | HepsiEmlak (Selenium ile scrape edildi) |
| **Veri Boyutu** | 9.584 ilan |
| **Model** | Deep Learning (Keras / TensorFlow) |
| **R² Skoru** | 0.93 |
| **MAE** | ~8.181 TL |
| **Hata Oranı** | %13.6 |
| **Ortalama Fiyat** | ~60.000 TL |

---

## 📁 Proje Yapısı

```
Istanbul-House-Rent-Prediction/
│
├── IstanbulHouseRentPrediction.ipynb  # Ana notebook (EDA + FE + Model)
├── tugce_simulated_data_v01.py        # Veri simülasyon scripti
├── app.py                             # Streamlit web uygulaması
│
├── istanbul_kiralik_complete.csv      # Ham scrape verisi
├── istanbul_kiralik_simulated.csv     # Simüle edilmiş temiz veri
│
├── istanbul_rent_model.h5             # Eğitilmiş Keras modeli
├── feature_columns.pkl                # Model feature sırası
├── scaler.pkl                         # StandardScaler objesi
│
├── requirements.txt                   # Bağımlılıklar
└── README.md
```

---

## 🗂️ Veri Seti

Ham veri HepsiEmlak'tan Selenium ile toplanmıştır. İlanlarda fiyat, metrekare ve bina yaşı gibi kritik değişkenlerin büyük bölümü eksik olduğundan, **ilçe/mahalle bazında sosyo-ekonomik göstergeler** ve **bölgesel kademeleme** kullanılarak simülasyon ile tamamlanmıştır.

### Özellikler

| Kolon | Açıklama |
|---|---|
| `Metrekare` | Evin büyüklüğü (m²) |
| `OdaSayisi` | Oda sayısı |
| `SalonSayisi` | Salon sayısı |
| `Kat` | Bulunduğu kat |
| `Yapı Yaşı` | Binanın yaşı |
| `Isıtma` | Isıtma tipi (1: Soba, 2: Kombi, 3: Merkezi) |
| `Esya` | Eşyalı / Eşyasız (0/1) |
| `Ilce` | İstanbul ilçesi (One-Hot Encoded) |
| `Villa`, `Dubleks`, `Tripleks` | Konut tipi bayrakları |
| `BogazManzarali`, `Manzarali` | Manzara bayrakları |
| `PremiumSkor` | Toplam premium özellik sayısı |
| `UlasimSkor` | Ulaşım olanakları skoru |

---

## ⚙️ Feature Engineering

- `OdaSayisi` / `SalonSayisi` → `3+1` formatından sayısal dönüşüm
- `KatKategori` → Kat bilgisinin kategorilere ayrılması
- `BinaKategori` → Yapı yaşının kategorilere ayrılması
- `PremiumSkor` → Premium özelliklerin toplamı
- `UlasimSkor` → Ulaşım olanaklarının toplamı
- `Ilce` → One-Hot Encoding

---

## 🧠 Model Mimarisi

```python
model = Sequential([
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)   # Regresyon çıktısı
])
```

- **Optimizer:** Adam  
- **Loss:** Mean Squared Error  
- **EarlyStopping:** patience=20, restore_best_weights=True  
- **Epochs:** 100, **Batch Size:** 64  

---

## 📊 Model Sonuçları

| Metrik | Değer |
|---|---|
| R² | **0.93** |
| RMSE | ~17.958 TL |
| MAE | ~8.181 TL |
| Hata Oranı (MAE/Ort.) | **%13.6** |

---

## 🚀 Streamlit Uygulaması

### Yerel Kurulum

```bash
# Repo'yu klonlayın
git clone https://github.com/tugcesi/Istanbul-House-Rent-Prediction.git
cd Istanbul-House-Rent-Prediction

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py
```

### Uygulama Özellikleri

- 📍 39 İstanbul ilçesi seçimi
- 🛏️ Oda / salon / metrekare / kat girişi
- 🏗️ Bina yaşı ve ısıtma tipi
- 💰 Anlık kira tahmini
- 📊 m² başına fiyat, oda başına fiyat ve yıllık toplam

---

## 🛠️ Kullanılan Teknolojiler

| Kategori | Kütüphane |
|---|---|
| **Veri İşleme** | pandas, numpy |
| **Görselleştirme** | matplotlib, seaborn, plotly |
| **Makine Öğrenmesi** | scikit-learn |
| **Derin Öğrenme** | TensorFlow / Keras |
| **Web Uygulaması** | Streamlit |
| **Web Scraping** | Selenium |

---

## 👩‍💻 Geliştirici

**Tuğçe Başyiğit**  
[GitHub](https://github.com/tugcesi)

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.