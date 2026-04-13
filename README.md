# 🏙️ İstanbul Kira Tahmini — Istanbul House Rent Prediction

> Yapay zeka destekli İstanbul kira tahmin modeli · Deep Learning + TensorFlow/Keras

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Proje Hakkında / About

Bu proje, **İstanbul genelinde ~39 ilçeye** ait kiralık daire verilerinden aylık kira bedelini tahmin eden bir **derin öğrenme modeli** içermektedir. Model; konum (ilçe/mahalle), oda sayısı, metrekare, kat, yapı yaşı, eşya durumu ve ısıtma tipi gibi özellikleri kullanarak gerçekçi kira tahminleri üretmektedir.

This project contains a **deep learning model** that predicts monthly rent prices from rental apartment data across ~39 districts of Istanbul. The model uses features such as location (district/neighborhood), number of rooms, square meters, floor, building age, furnished status, and heating type.

---

## 📊 Model Özeti / Model Summary

| Başlık / Title | Detay / Detail |
|---|---|
| **Veri Kaynağı** | HepsiEmlak (Selenium scraping) + Endeksa.com Mart 2026 |
| **Veri Boyutu** | 9.584 ilan |
| **Model Türü** | Derin Sinir Ağı — Deep Neural Network (Keras / TensorFlow) |
| **Model Dosyası** | `.h5` formatı |
| **R² Skoru** | 0.93 |
| **MAE** | ~8.181 TL |
| **RMSE** | ~17.958 TL |
| **Hata Oranı** | %13.6 |
| **Ortalama Fiyat** | ~60.000 TL |

---

## 📁 Dosya Yapısı / File Structure

| Dosya | Açıklama |
|---|---|
| `IstanbulHouseRentPrediction.ipynb` | Ana notebook — EDA, feature engineering, model eğitimi |
| `tugce_simulated_data_v01.py` | Veri simülasyon scripti — mahalle katsayılarıyla eksik değer doldurma |
| `istanbul_rent_model.h5` | Eğitilmiş Keras modeli (.h5 formatı) |
| `feature_columns.pkl` | Model giriş kolonu sırası (pickle) |
| `scaler.pkl` | Eğitimde kullanılan StandardScaler objesi |
| `istanbul_kiralik_complete.csv` | Ham veri seti (9.584 kiralık ilan) |
| `istanbul_kiralik_simulated.csv` | Simüle edilmiş / eksik değerleri doldurulmuş temiz veri |
| `requirements.txt` | Python bağımlılıkları |
| `LICENSE` | MIT Lisans |
| `README.md` | Bu dosya |

---

## 🗂️ Veri Seti / Dataset

Ham veri HepsiEmlak'tan Selenium ile toplanmıştır. İlanlarda fiyat, metrekare ve bina yaşı gibi kritik değişkenlerin büyük bölümü eksik olduğundan, eksik değerler **`tugce_simulated_data_v01.py`** ile tamamlanmıştır.

**Simülasyon yöntemi:** [Endeksa.com](https://endeksa.com) Mart 2026 verilerine dayalı mahalle düzeyinde kira katsayıları (`ILCE_ORT` — TL/m²) kullanılarak oda tipine ve referans m²'ye göre kira değerleri üretilmiştir.

### Model Giriş Özellikleri / Input Features

| Özellik | Açıklama |
|---|---|
| `Metrekare` | Evin büyüklüğü (m²) |
| `OdaSayisi` | Oda sayısı |
| `SalonSayisi` | Salon sayısı |
| `Kat` | Bulunduğu kat |
| `Yapı Yaşı` | Binanın yaşı (yıl) |
| `Isıtma` | Isıtma tipi (Soba/Doğalgaz, Kombi, Merkezi) |
| `Esya` | Eşyalı / Eşyasız (0/1) |
| `Ilce` | İstanbul ilçesi (One-Hot Encoded, ~39 ilçe) |
| `Villa`, `Dubleks`, `Tripleks` | Konut tipi bayrakları |
| `BogazManzarali`, `Manzarali` | Manzara bayrakları |
| `PremiumSkor` | Toplam premium özellik sayısı |
| `UlasimSkor` | Ulaşım olanakları skoru |

---

## ⚙️ Feature Engineering

- `OdaSayisi` / `SalonSayisi` → `3+1` formatından sayısal dönüşüm
- `KatKategori` → Kat bilgisinin kategorilere ayrılması
- `BinaKategori` → Yapı yaşının kategorilere ayrılması
- `PremiumSkor` → Premium özelliklerin (villa, dubleks, Boğaz manzarası vb.) toplamı
- `UlasimSkor` → Ulaşım olanaklarının (metro, metrobüs vb.) toplamı
- `Ilce` → One-Hot Encoding

---

## 🧠 Model Mimarisi / Model Architecture

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
    Dense(1)   # Regresyon çıktısı / Regression output
])
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error
- **EarlyStopping:** patience=20, restore_best_weights=True
- **Epochs:** 100, **Batch Size:** 64

---

## 🚀 Kurulum ve Kullanım / Setup & Usage

### 1. Repoyu klonlayın / Clone the repository

```bash
git clone https://github.com/tugcesi/Istanbul-House-Rent-Prediction.git
cd Istanbul-House-Rent-Prediction
```

### 2. Sanal ortam oluşturun (önerilir) / Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Bağımlılıkları yükleyin / Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jupyter Notebook ile çalıştırın / Run with Jupyter Notebook

```bash
jupyter notebook IstanbulHouseRentPrediction.ipynb
```

Notebook sırasıyla EDA, feature engineering ve model eğitimini kapsamaktadır.

---

## 🗺️ Desteklenen İlçeler / Supported Districts (~39 ilçe)

Arnavutköy · Ataşehir · Avcılar · Bağcılar · Bahçelievler · Bakırköy · Bayrampaşa · Beşiktaş · Beykoz · Beylikdüzü · Beyoğlu · Büyükçekmece · Çatalca · Çekmeköy · Esenler · Esenyurt · Eyüpsultan · Fatih · Gaziosmanpaşa · Güngören · Kadıköy · Kağıthane · Kartal · Küçükçekmece · Maltepe · Pendik · Sancaktepe · Sarıyer · Silivri · Sultanbeyli · Sultangazi · Şile · Şişli · Tuzla · Ümraniye · Üsküdar · Zeytinburnu · Adalar · Başakşehir

---

## 🛠️ Teknoloji Stack'i / Technology Stack

| Kategori | Kütüphane |
|---|---|
| **Derin Öğrenme** | TensorFlow / Keras |
| **Veri İşleme** | pandas, numpy |
| **Makine Öğrenmesi** | scikit-learn (StandardScaler) |
| **Görselleştirme** | matplotlib, seaborn, plotly |
| **Notebook** | Jupyter |

---

## 🤝 Katkı / Contributing

Katkıda bulunmak ister misiniz? [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını inceleyin.

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 👩‍💻 Geliştirici / Developer

**Tuğçe Başyiğit**
- 🐙 [@tugcesi](https://github.com/tugcesi)

---

## 📄 Lisans / License

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.  
This project is licensed under the [MIT License](LICENSE).