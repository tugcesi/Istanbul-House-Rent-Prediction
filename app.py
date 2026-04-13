"""
Istanbul Kira Tahmin Uygulaması
Streamlit tabanlı kira tahmin arayüzü — Endeksa Mart 2026 verileriyle güncellenmiştir.
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa yapılandırması — İLK Streamlit çağrısı olmalı
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="İstanbul Kira Tahmin",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# İLÇE ORTALAMA AYLIK KİRA (TL) — Endeksa Mart 2026
# ─────────────────────────────────────────────────────────────────────────────
ILCE_KIRA = {
    "Esenyurt": 21632, "Sultangazi": 23698, "Arnavutköy": 22821,
    "Silivri": 26923, "Esenler": 22799, "Sultanbeyli": 26198,
    "Büyükçekmece": 35855, "Beylikdüzü": 36182, "Sancaktepe": 29081,
    "Çatalca": 28795, "Gaziosmanpaşa": 28211, "Bağcılar": 28803,
    "Avcılar": 31625, "Tuzla": 32842, "Bayrampaşa": 30665,
    "Güngören": 35567, "Fatih": 26374, "Pendik": 33000,
    "Beykoz": 43056, "Başakşehir": 41409, "Çekmeköy": 34951,
    "Küçükçekmece": 34160, "Ümraniye": 35507, "Şile": 42339,
    "Kağıthane": 33592, "Bahçelievler": 37347, "Kartal": 41340,
    "Eyüpsultan": 40138, "Üsküdar": 42946, "Maltepe": 39584,
    "Şişli": 40694, "Ataşehir": 41364, "Beyoğlu": 40866,
    "Adalar": 45000, "Zeytinburnu": 59079, "Sarıyer": 72021,
    "Bakırköy": 69712, "Beşiktaş": 61587, "Kadıköy": 71645,
}

# Sıralı ilçe listesi (dropdown için)
districts = sorted(ILCE_KIRA.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Model yükleme — önbelleklenmiş, yalnızca bir kez yüklenir
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_resources():
    """TensorFlow modeli, scaler ve feature sütunlarını yükle."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("istanbul_rent_model.h5", compile=False)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        return model, scaler, feature_cols, True
    except Exception:
        return None, None, None, False


model, scaler, feature_cols, model_loaded = load_model_resources()

# ─────────────────────────────────────────────────────────────────────────────
# CSS — titreşim / flash önleme ve görsel düzenleme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Titreşim / flash önleme */
    [data-testid="stAppViewContainer"] { transition: none !important; }
    [data-testid="block-container"]    { transition: none !important; }

    /* Genel font ve arka plan */
    body { font-family: 'Segoe UI', sans-serif; }

    /* Başlık */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Sonuç kartı */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .result-card .price {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .result-card .label {
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Demo rozeti */
    .demo-badge {
        background: #ff9800;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Metrik kartları */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        border: 1px solid #e9ecef;
    }

    /* Sekme görünümü */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Harita iframe */
    .map-container iframe { border-radius: 12px; }

    /* Tablo */
    .dataframe { font-size: 0.9rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Başlık
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🏠 İstanbul Kira Tahmin</div>'
    '<div class="sub-title">Endeksa Mart 2026 verileriyle güncellendi · 39 ilçe</div>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.info(
        "ℹ️ Model dosyası bulunamadı — **demo mod** aktif. "
        "Tahminler ilçe bazlı ortalama fiyatlardan hesaplanmaktadır.",
        icon="🔧",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sekmeler
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_map, tab_about = st.tabs(
    ["🏠 Kira Tahmini", "🗺️ İstanbul Kira Haritası", "ℹ️ Hakkında"]
)

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — Kira Tahmini
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("📋 Ev Bilgileri")

        # İlçe
        ilce = st.selectbox(
            "İlçe",
            districts,
            index=districts.index("Kadıköy") if "Kadıköy" in districts else 0,
            key="ilce_select",
        )

        # Metrekare + Oda
        c1, c2 = st.columns(2)
        with c1:
            m2 = st.number_input(
                "Metrekare (m²)", min_value=20, max_value=500, value=90, step=5, key="m2"
            )
        with c2:
            oda_options = ["1+0", "1+1", "2+1", "2+2", "3+1", "3+2", "4+1", "4+2", "5+1"]
            oda = st.selectbox("Oda Tipi", oda_options, index=2, key="oda")

        # Kat + Yaş
        c3, c4 = st.columns(2)
        with c3:
            kat = st.number_input(
                "Bulunduğu Kat", min_value=0, max_value=50, value=3, step=1, key="kat"
            )
        with c4:
            yas = st.number_input(
                "Yapı Yaşı", min_value=0, max_value=50, value=10, step=1, key="yas"
            )

        # Özellikler
        st.markdown("**Özellikler**")
        feat_c1, feat_c2, feat_c3 = st.columns(3)
        with feat_c1:
            esyali = st.checkbox("Eşyalı", key="esyali")
            villa = st.checkbox("Villa", key="villa")
            bahce = st.checkbox("Bahçe", key="bahce")
        with feat_c2:
            dubleks = st.checkbox("Dubleks", key="dubleks")
            havuz = st.checkbox("Havuz", key="havuz")
            manzara = st.checkbox("Manzaralı", key="manzara")
        with feat_c3:
            bogaz = st.checkbox("Boğaz Manzaralı", key="bogaz")
            sahil = st.checkbox("Sahile Yakın", key="sahil")

        predict_btn = st.button("🔍 Kira Tahmini Yap", use_container_width=True, type="primary")

    # ── Sonuç sütunu ──────────────────────────────────────────────────────────
    with col_result:
        st.subheader("💰 Tahmin Sonucu")

        if predict_btn:
            # Oda sayısı / salon ayrıştır
            try:
                oda_parts = oda.split("+")
                oda_sayisi = int(oda_parts[0])
                salon_sayisi = int(oda_parts[1]) if len(oda_parts) > 1 else 0
            except (ValueError, IndexError):
                oda_sayisi, salon_sayisi = 2, 1

            # Ulaşım skoru (ilçe bazlı basit tahmin)
            ulasim_skor_map = {
                "Kadıköy": 9, "Beşiktaş": 9, "Şişli": 9, "Fatih": 8, "Beyoğlu": 8,
                "Bakırköy": 8, "Üsküdar": 8, "Ataşehir": 7, "Maltepe": 7,
                "Kağıthane": 7, "Küçükçekmece": 7, "Bağcılar": 6, "Güngören": 6,
                "Bahçelievler": 6, "Kartal": 6, "Pendik": 6, "Ümraniye": 6,
                "Sarıyer": 6, "Zeytinburnu": 7, "Eyüpsultan": 6,
            }
            ulasim_skor = ulasim_skor_map.get(ilce, 5)

            predicted_price = None

            # ── Model modu ────────────────────────────────────────────────
            if model_loaded and feature_cols:
                try:
                    row = {col: 0 for col in feature_cols}

                    # Sayısal özellikler
                    row["Metrekare"] = m2
                    row["OdaSayisi"] = oda_sayisi
                    row["SalonSayisi"] = salon_sayisi
                    row["Kat"] = kat
                    row["Yapı Yaşı"] = yas
                    row["UlasimSkor"] = ulasim_skor

                    # Binary özellikler
                    row["Villa"] = int(villa)
                    row["Dubleks"] = int(dubleks)
                    row["Bahce"] = int(bahce)
                    row["Havuz"] = int(havuz)
                    row["Manzarali"] = int(manzara)
                    row["BogazManzarali"] = int(bogaz)
                    row["SahileYakinlik"] = int(sahil)

                    # İlçe one-hot
                    ilce_col = f"Ilce_{ilce}"
                    if ilce_col in row:
                        row[ilce_col] = 1

                    df_input = pd.DataFrame([row], columns=feature_cols)
                    X_scaled = scaler.transform(df_input)
                    pred_log = model.predict(X_scaled, verbose=0)[0][0]
                    predicted_price = int(np.expm1(pred_log))
                    predicted_price = round(predicted_price / 500) * 500
                    mode_label = "Model Tahmini"
                    mode_color = "#667eea"
                except Exception:
                    predicted_price = None

            # ── Demo modu ─────────────────────────────────────────────────
            if predicted_price is None:
                ilce_ort = ILCE_KIRA.get(ilce, 35000)
                m2_birim = ilce_ort / 90.0
                demo_price = int(m2 * m2_birim + oda_sayisi * 1500 - yas * 200 + kat * 300)
                demo_price = max(demo_price, 15000)
                predicted_price = round(demo_price / 500) * 500
                mode_label = "Demo Tahmini"
                mode_color = "#ff9800"

            # Göster
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Tahmini Aylık Kira</div>
                    <div class="price">₺{predicted_price:,}</div>
                    <div style="margin-top:0.5rem; opacity:0.85; font-size:0.85rem">
                        {ilce} · {m2} m² · {oda}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Karşılaştırma metrikleri
            ilce_ort_val = ILCE_KIRA.get(ilce, 35000)
            diff_pct = (predicted_price - ilce_ort_val) / ilce_ort_val * 100
            diff_str = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"

            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("İlçe Ortalaması", f"₺{ilce_ort_val:,}")
            with mc2:
                st.metric("Ortalamaya Fark", diff_str)

            if mode_label == "Demo Tahmini":
                st.markdown(
                    '<span class="demo-badge">⚠️ Demo Mod — Model yüklenmedi</span>',
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
                <div style="text-align:center; padding:3rem; color:#aaa;">
                    <div style="font-size:3rem">🏠</div>
                    <div style="margin-top:1rem">Sol taraftaki formu doldurun ve<br><b>Kira Tahmini Yap</b> butonuna tıklayın.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — Harita
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    map_c1, map_c2 = st.columns([1, 2], gap="large")

    with map_c1:
        st.subheader("📊 İlçe Kira Tablosu")
        st.caption("Kaynak: Endeksa Mart 2026 · Aylık ortalama kira (TL)")

        # Tabloyu azalan sırayla göster (en pahalıdan en ucuza)
        df_kira = (
            pd.DataFrame(
                list(ILCE_KIRA.items()), columns=["İlçe", "Ort. Aylık Kira (₺)"]
            )
            .sort_values("Ort. Aylık Kira (₺)", ascending=False)
            .reset_index(drop=True)
        )
        df_kira.index += 1
        df_kira["Ort. Aylık Kira (₺)"] = df_kira["Ort. Aylık Kira (₺)"].apply(
            lambda x: f"₺{x:,}"
        )
        st.dataframe(df_kira, use_container_width=True, height=520)

    with map_c2:
        st.subheader("🌍 Endeksa Canlı Kira Haritası")
        st.caption("Aşağıdaki harita Endeksa.com'dan canlı olarak yüklenmektedir.")

        # Endeksa İstanbul kiralık konut haritası
        endeksa_url = (
            "https://www.endeksa.com/tr/analiz/turkiye/istanbul/endeks/kiralik/konut"
        )

        components.iframe(
            endeksa_url,
            height=520,
            scrolling=True,
        )

        st.markdown(
            f"""
            <div style="text-align:right; margin-top:0.5rem;">
                <a href="{endeksa_url}" target="_blank"
                   style="color:#667eea; font-size:0.85rem; text-decoration:none;">
                   🔗 Endeksa'da Tam Ekran Aç ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Bölge bazlı bar grafik
    st.subheader("📈 İlçelere Göre Ortalama Kira (₺)")
    df_plot = pd.DataFrame(
        list(ILCE_KIRA.items()), columns=["İlçe", "Kira"]
    ).sort_values("Kira", ascending=True)

    fig = px.bar(
        df_plot,
        x="Kira",
        y="İlçe",
        orientation="h",
        color="Kira",
        color_continuous_scale="Viridis",
        labels={"Kira": "Ort. Aylık Kira (₺)", "İlçe": ""},
        height=900,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=20, b=20),
        plot_bgcolor="white",
        xaxis=dict(tickformat=",", gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=11)),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>₺%{x:,}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — Hakkında
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ Uygulama Hakkında")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
### 🎯 Proje Amacı
Bu uygulama, İstanbul'daki kiralık konut fiyatlarını tahmin etmek için makine öğrenmesi
kullanmaktadır.

### 📊 Veri Kaynakları
- **Endeksa.com** — Mart 2026 TL/m² kira endeksleri
- İstanbul kiralık konut ilanları simüle edilmiş veri seti

### 🤖 Model
- **Mimari:** Derin sinir ağı (TensorFlow/Keras)
- **Eğitim verisi:** 39 ilçe, 600+ mahalle
- **Hedef:** Log(kira) → ters dönüşüm ile TL tahmin

### 🏙️ Kapsam
| | |
|---|---|
| İlçe sayısı | 39 |
| Mahalle sayısı | 600+ |
| Güncelleme | Mart 2026 |
"""
        )

    with col_b:
        st.markdown(
            """
### 🔧 Teknik Detaylar
- **Framework:** Streamlit
- **ML:** TensorFlow 2.x + Scikit-learn
- **Görselleştirme:** Plotly
- **Kira verisi:** Endeksa Mart 2026

### 📌 Demo Mod
Model dosyası bulunamadığında uygulama demo modda çalışır.  
Demo modda tahmin formülü:
```
ilce_ort = ILCE_KIRA[ilce]
m2_birim = ilce_ort / 90
fiyat = m2 × m2_birim + oda×1500 - yaş×200 + kat×300
fiyat = max(fiyat, 15000)
fiyat = round(fiyat / 500) × 500
```

### 🌐 Kaynaklar
- [Endeksa İstanbul Kira Haritası](https://www.endeksa.com/tr/analiz/turkiye/istanbul/endeks/kiralik/konut)
- [GitHub Repo](https://github.com/tugcesi/Istanbul-House-Rent-Prediction)
"""
        )
