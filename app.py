import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="İstanbul Kira Tahmini",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
    }

    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #fbbf24 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }

    p, label, div {
        font-family: 'Inter', sans-serif !important;
    }

    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .kat-badge {
        background: rgba(6,182,212,0.15);
        border: 1px solid rgba(6,182,212,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #67e8f9;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .prediction-box {
        background: linear-gradient(135deg, rgba(251,191,36,0.1) 0%, rgba(251,191,36,0.05) 100%);
        border: 2px solid #fbbf24;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 0 30px rgba(251,191,36,0.3);
    }

    .price-text {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        color: #fbbf24;
        font-weight: 700;
    }

    .feature-card {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #06b6d4;
    }

    .stButton > button {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        padding: 0.75rem 2rem !important;
    }

    .stSelectbox label, .stSlider label,
    .stNumberInput label, .stCheckbox label {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── BAŞLIK ───────────────────────────────────────────────────────────────────
st.markdown("<h1>🏙️ İstanbul Kira Tahmini</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Yapay Zeka ile Evinizin Değerini Keşfedin</p>",
            unsafe_allow_html=True)

# ── BANNER — Unsplash direkt CDN linkleri ─────────────────────────────────────
# ✅ DÜZELTİLDİ: Unsplash'in images.unsplash.com CDN'i her zaman erişilebilir
BANNER_URL  = "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?w=1400&q=80"
SIDEBAR_URL = "https://images.unsplash.com/photo-1541432901042-2d8bd64b4a9b?w=800&q=80"

st.image(BANNER_URL, use_container_width=True, caption="İstanbul Boğazı")
st.markdown("<br>", unsafe_allow_html=True)

# ── MODEL YÜKLE ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model('istanbul_rent_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
    model_loaded = True
except Exception as e:
    st.warning(f"⚠️ Model yüklenemedi, demo modda çalışıyor. Hata: {e}")
    model_loaded = False

# ── SABITLER ──────────────────────────────────────────────────────────────────
districts = sorted([
    'Adalar', 'Arnavutköy', 'Ataşehir', 'Avcılar', 'Bağcılar',
    'Bahçelievler', 'Bakırköy', 'Başakşehir', 'Bayrampaşa', 'Beşiktaş',
    'Beykoz', 'Beylikdüzü', 'Beyoğlu', 'Büyükçekmece', 'Çatalca',
    'Çekmeköy', 'Esenler', 'Esenyurt', 'Eyüpsultan', 'Fatih',
    'Gaziosmanpaşa', 'Güngören', 'Kadıköy', 'Kağıthane', 'Kartal',
    'Küçükçekmece', 'Maltepe', 'Pendik', 'Sancaktepe', 'Sarıyer',
    'Silivri', 'Sultanbeyli', 'Sultangazi', 'Şile', 'Şişli',
    'Tuzla', 'Ümraniye', 'Üsküdar', 'Zeytinburnu'
])

kat_labels  = ['Bodrum', 'Zemin Kat', 'Düşük Kat', 'Orta Kat', 'Yüksek Kat', 'Çok Yüksek Kat']
isitma_opts = {'Soba/Doğalgaz': 1.0, 'Kombi': 2.0, 'Merkezi': 3.0}

# ── GİRDİ ALANLARI ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    ilce         = st.selectbox("📍 İlçe", districts, index=0)
    oda_sayisi   = st.slider("🛏️ Oda Sayısı", 1, 10, 2)
    salon_sayisi = st.slider("🛋️ Salon Sayısı", 0, 3, 1)
    m2           = st.number_input("📐 Metrekare (m²)", min_value=20,
                                   max_value=1000, value=85, step=5)

with col2:
    bina_yasi = st.slider("🏗️ Bina Yaşı", 0, 50, 5)
    kat       = st.slider("🚪 Bulunduğu Kat", -2, 30, 3)

    # ✅ DÜZELTİLDİ: st.info yerine özel HTML badge — her zaman görünür
    bins         = [-np.inf, -1, 0, 2, 7, 10, np.inf]
    kat_kategori = pd.cut([kat], bins=bins, labels=kat_labels)[0]
    st.markdown(
        f"<div class='kat-badge'>🏢 Kat Kategorisi: <strong>{kat_kategori}</strong></div>",
        unsafe_allow_html=True
    )

    isitma_sec = st.selectbox("🔥 Isıtma Tipi", list(isitma_opts.keys()))
    isitma_val = isitma_opts[isitma_sec]
    esyali     = st.checkbox("🛋️ Eşyalı mı?", value=False)

# ── TAHMİN BUTONU ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("💰 Kira Tahmini Yap", type="primary"):

    if model_loaded:
        try:
            input_dict = {
                'Metrekare'      : [float(m2)],
                'OdaSayisi'      : [float(oda_sayisi)],
                'SalonSayisi'    : [float(salon_sayisi)],
                'Kat'            : [float(kat)],
                'Yapı Yaşı'      : [float(bina_yasi)],
                'Isıtma'         : [isitma_val],
                'Esya'           : [float(int(esyali))],
                'Villa'          : [0.0],
                'Dubleks'        : [0.0],
                'Tripleks'       : [0.0],
                'Yali'           : [0.0],
                'BogazManzarali' : [0.0],
                'Manzarali'      : [0.0],
                'SahileYakinlik' : [0.0],
                'Bahce'          : [0.0],
                'Havuz'          : [0.0],
                'UlasimSkor'     : [0.0],
                f'Ilce_{ilce}'   : [1.0],
            }

            input_df = pd.DataFrame(input_dict)

            for c in feature_columns:
                if c not in input_df.columns:
                    input_df[c] = 0.0
            input_df = input_df[feature_columns]

            input_scaled = scaler.transform(input_df)
            prediction   = float(model.predict(input_scaled, verbose=0)[0][0])

            st.markdown(f"""
            <div class='prediction-box'>
                <div style='font-size:1.2rem; color:#94a3b8; margin-bottom:1rem;'>
                    Tahmini Aylık Kira
                </div>
                <div class='price-text'>₺{prediction:,.0f}</div>
                <div style='margin-top:1rem; color:#64748b;'>
                    📍 {ilce} &nbsp;•&nbsp; {m2} m² &nbsp;•&nbsp; {oda_sayisi}+{salon_sayisi}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class='feature-card'>
                    <h4 style='color:#06b6d4; margin:0;'>📊 m² Fiyatı</h4>
                    <p style='font-size:1.3rem; margin:0.5rem 0; color:white;'>
                        ₺{prediction / m2:,.0f} / m²
                    </p>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='feature-card' style='border-left-color:#fbbf24;'>
                    <h4 style='color:#fbbf24; margin:0;'>🏠 Oda Başına</h4>
                    <p style='font-size:1.3rem; margin:0.5rem 0; color:white;'>
                        ₺{prediction / (oda_sayisi + salon_sayisi):,.0f}
                    </p>
                </div>""", unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='feature-card' style='border-left-color:#f472b6;'>
                    <h4 style='color:#f472b6; margin:0;'>📅 Yıllık Toplam</h4>
                    <p style='font-size:1.3rem; margin:0.5rem 0; color:white;'>
                        ₺{prediction * 12:,.0f}
                    </p>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {str(e)}")

    else:
        ilce_katsayi = {
            'Beşiktaş': 2.2, 'Sarıyer': 2.0, 'Kadıköy': 1.9, 'Şişli': 1.8,
            'Beyoğlu': 1.7, 'Üsküdar': 1.6, 'Bakırköy': 1.5, 'Ataşehir': 1.4,
            'Maltepe': 1.2, 'Kartal': 1.1,
        }
        katsayi    = ilce_katsayi.get(ilce, 1.0)
        demo_price = max(
            (m2 * 180 + oda_sayisi * 2500 + kat * 400 - bina_yasi * 80) * katsayi,
            8000
        )
        st.markdown(f"""
        <div class='prediction-box'>
            <div style='font-size:1.2rem; color:#94a3b8; margin-bottom:1rem;'>
                Tahmini Aylık Kira
                <span style='font-size:0.75rem; background:rgba(251,191,36,0.2);
                padding:2px 8px; border-radius:20px; margin-left:8px;'>DEMO</span>
            </div>
            <div class='price-text'>₺{demo_price:,.0f}</div>
            <div style='margin-top:1rem; color:#64748b;'>
                📍 {ilce} &nbsp;•&nbsp; {m2} m² &nbsp;•&nbsp; {oda_sayisi}+{salon_sayisi}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── ALT BİLGİ ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#475569; font-size:0.85rem;
border-top:1px solid rgba(255,255,255,0.07); padding-top:1.5rem;'>
    🏙️ İstanbul Kira Tahmini &nbsp;·&nbsp;
    Model: <b style='color:#94a3b8;'>istanbul_rent_model.h5</b> &nbsp;·&nbsp;
    Geliştirici: <b style='color:#94a3b8;'>tugcesi❤️</b>
</p>
""", unsafe_allow_html=True)

# ── SİDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ✅ DÜZELTİLDİ: Unsplash CDN — Galata Kulesi
    st.image(SIDEBAR_URL, use_container_width=True, caption="İstanbul")
    st.markdown("---")
    st.markdown("### 📊 Model İstatistikleri")
    st.metric("R² Skoru", "0.93")
    st.metric("MAE", "8,181 TL")
    st.metric("Hata Oranı", "%13.6")
    st.metric("Veri Sayısı", "9,584 ilan")
    st.markdown("---")
    st.info("💡 Daha doğru tahminler için tüm bilgileri eksiksiz girin.")
    st.markdown("---")
    st.markdown("### 📌 Nasıl Kullanılır?")
    st.markdown("""
    1. **İlçeyi** seçin
    2. **Oda & m²** bilgilerini girin
    3. **Bina yaşı & kat** belirtin
    4. **Isıtma & eşya** durumunu seçin
    5. **Tahmin Yap** butonuna basın
    """)