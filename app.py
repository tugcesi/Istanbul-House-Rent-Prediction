import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="İstanbul Kira Tahmini",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%); }
.main-title {
    font-family: 'Playfair Display', serif; color: #fbbf24;
    font-size: 2.8rem; text-align: center; margin-bottom: 0.2rem;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
}
.subtitle { color: #94a3b8; text-align: center; font-size: 1.05rem; margin-bottom: 1.5rem; }
.card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.5rem; }
.kat-badge { background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.35); border-radius: 8px; padding: 0.5rem 1rem; color: #67e8f9; font-size: 0.9rem; margin: 0.4rem 0 0.8rem 0; }
.mahalle-badge { background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.30); border-radius: 8px; padding: 0.5rem 1rem; color: #fde68a; font-size: 0.85rem; margin: 0.4rem 0 0.8rem 0; }
.result-box { background: linear-gradient(135deg, rgba(251,191,36,0.12), rgba(251,191,36,0.04)); border: 2px solid #fbbf24; border-radius: 18px; padding: 2rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 0 30px rgba(251,191,36,0.25); }
.result-price { font-family: 'Playfair Display', serif; font-size: 3.5rem; color: #fbbf24; font-weight: 700; }
.stat-card { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 0.9rem 1.1rem; border-left: 3px solid #06b6d4; margin-top: 0.6rem; }
.stButton > button { background: linear-gradient(135deg, #fbbf24, #f59e0b) !important; color: #0f172a !important; font-weight: 700 !important; border-radius: 10px !important; border: none !important; font-size: 1.05rem !important; width: 100% !important; padding: 0.7rem 1.5rem !important; }
.stButton > button:hover { box-shadow: 0 8px 20px rgba(251,191,36,0.4) !important; }
label, .stCheckbox span { color: #cbd5e1 !important; font-size: 0.9rem !important; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.12) !important; color: #f1f5f9 !important; }
input[type="number"] { background: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.12) !important; color: #f1f5f9 !important; }
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255,255,255,0.12) !important;
}
div[data-testid="stNumberInput"] input::placeholder { color: #94a3b8 !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(251,191,36,0.15) !important; color: #fbbf24 !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "ilce": "Beşiktaş", "mahalle": "", "oda": 2, "salon": 1, "m2": 85,
    "yas": 5, "kat": 3, "isitma": "Kombi",
    "esyali": False, "tahmin_yapildi": False, "sonuc": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    mdl = load_model("istanbul_rent_model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scl = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        fc = pickle.load(f)
    return mdl, scl, fc

try:
    model, scaler, feature_columns = load_artifacts()
    model_loaded = True
except Exception:
    model_loaded = False

# ── SABİTLER ──────────────────────────────────────────────────────────────────
DISTRICTS = sorted([
    "Adalar","Arnavutköy","Ataşehir","Avcılar","Bağcılar","Bahçelievler",
    "Bakırköy","Başakşehir","Bayrampaşa","Beşiktaş","Beykoz","Beylikdüzü",
    "Beyoğlu","Büyükçekmece","Çatalca","Çekmeköy","Esenler","Esenyurt",
    "Eyüpsultan","Fatih","Gaziosmanpaşa","Güngören","Kadıköy","Kağıthane",
    "Kartal","Küçükçekmece","Maltepe","Pendik","Sancaktepe","Sarıyer",
    "Silivri","Sultanbeyli","Sultangazi","Şile","Şişli","Tuzla",
    "Ümraniye","Üsküdar","Zeytinburnu",
])

KAT_LABELS = ["Bodrum","Zemin Kat","Düşük Kat","Orta Kat","Yüksek Kat","Çok Yüksek Kat"]
ISITMA_MAP = {"Soba/Doğalgaz": 1.0, "Kombi": 2.0, "Merkezi": 3.0}

# ── İLÇE ORTALAMA KİRA (TL/m²) — Endeksa Mart 2026 ─────────────────────────
ILCE_ORT = {
    "Esenyurt": 228, "Sultangazi": 237, "Arnavutköy": 240, "Silivri": 247,
    "Esenler": 251, "Sultanbeyli": 298, "Büyükçekmece": 282, "Beylikdüzü": 289,
    "Sancaktepe": 312, "Çatalca": 294, "Gaziosmanpaşa": 294, "Bağcılar": 297,
    "Avcılar": 307, "Tuzla": 358, "Bayrampaşa": 368, "Güngören": 326,
    "Fatih": 330, "Pendik": 385, "Beykoz": 398, "Başakşehir": 357,
    "Çekmeköy": 342, "Küçükçekmece": 367, "Ümraniye": 422, "Şile": 285,
    "Kağıthane": 391, "Bahçelievler": 393, "Kartal": 413, "Eyüpsultan": 418,
    "Üsküdar": 425, "Maltepe": 440, "Şişli": 452, "Ataşehir": 460,
    "Beyoğlu": 492, "Adalar": 445, "Zeytinburnu": 585, "Sar��yer": 590,
    "Bakırköy": 591, "Beşiktaş": 592, "Kadıköy": 651,
}

MIN_KATSAYI = 0.65
MAX_KATSAYI = 2.20

# ── MAHALLE KATSA YI TABLOSU ──────────────────────────────────────────────────
# ilce_adi → { mahalle_adi: katsayi }  (MAHALLELER sözlüğünden düzleştirildi)
MAHALLE_KATSAYI: dict[str, dict[str, float]] = {
    "Beşiktaş": {
        "Bebek": 1.55, "Arnavutköy(Beş)": 1.45, "Etiler": 1.40,
        "Nisbetiye": 1.40, "Akat": 1.38, "Ulus": 1.38, "Levazım": 1.35,
        "Türkali": 1.45, "Konaklar": 1.35, "Kuruçeşme": 1.30,
        "Levent": 1.22, "Yıldız": 1.12, "Gayrettepe": 1.18,
        "Sinanpaşa": 1.15, "Ortaköy": 1.10, "Mecidiye": 1.10,
        "Vişnezade": 1.10, "Cihannüma": 1.08, "Abbasağa": 1.05,
        "Muradiye": 1.05, "Balmumcu": 1.00, "Beşiktaş Merkez": 0.84,
        "Dikilitaş": 0.73,
    },
    "Sarıyer": {
        "Yeniköy": 1.54, "Emirgan": 1.25, "İstinye": 1.31,
        "Ayazağa": 1.05, "Tarabya": 1.10, "Zekeriyaköy": 1.10,
        "Huzur": 1.00, "Maslak": 1.00, "Poligon": 0.95,
        "Reşitpaşa": 0.95, "Fatih Sultan Mehmet": 0.92,
        "Çayırbaşı": 0.92, "Darüşşafaka": 0.90, "Bahçeköy": 0.90,
        "Uskumruköy": 0.90, "Kireçburnu": 0.90, "Büyükdere": 0.88,
        "Ferahevler": 0.88, "Cumhuriyet(Sar)": 0.88, "Sarıyer Merkez": 0.85,
        "Çamlıtepe": 0.85, "Rumeli Kavağı": 0.75, "Rumelifeneri": 0.73,
        "Pınar(Sar)": 0.90,
    },
    "Şişli": {
        "Nişantaşı": 1.55, "Teşvikiye": 1.44, "Harbiye": 1.42,
        "Osmanbey": 1.40, "Cumhuriyet": 1.38, "Fulya": 1.24,
        "Esentepe": 1.18, "Bomonti": 1.13, "Feriköy": 1.08,
        "Halaskargazi": 1.05, "Ergenekon": 1.02, "Mecidiyeköy": 1.00,
        "Paşa": 1.00, "İnönü(Şişli)": 0.95, "Şişli Merkez": 0.86,
        "Kuştepe": 0.75, "Gülbahar": 0.73,
    },
    "Beyoğlu": {
        "Cihangir": 1.54, "Galata": 1.40, "Tomtom": 1.38,
        "Firuzağa": 1.35, "Gümüşsuyu": 1.32, "Katip Mustafa Çelebi": 1.30,
        "Çukurcuma": 1.30, "Asmalımescit": 1.28, "Karaköy": 1.18,
        "Kemankeş": 1.20, "Müeyyetzade": 1.15, "Taksim": 1.10,
        "Evliya Çelebi": 1.10, "Beyoğlu Merkez": 1.00, "Sütlüce": 0.90,
        "Pürtelaş": 0.85, "Kaptanpaşa": 0.82, "Kulaksız": 0.82,
        "Halıcıoğlu": 0.80, "Örnektepe": 0.78, "Tarlabaşı": 0.73,
    },
    "Kağıthane": {
        "Seyrantepe": 1.41, "Çağlayan": 1.20, "Kağıthane Merkez": 1.00,
        "Merkez(Kağ)": 1.00, "Talatpaşa": 0.95, "Ortabayır": 0.95,
        "Emniyet Evleri": 0.92, "Gürsel": 0.88, "Hürriyet": 0.88,
        "Harmantepe": 0.90, "Telsizler": 0.85, "Gültepe": 0.82,
        "Yahya Kemal": 0.82, "Şirintepe": 0.80, "Nurtepe": 0.78,
        "Hamidiye": 0.72,
    },
    "Eyüpsultan": {
        "Göktürk": 1.40, "Alibeyköy": 1.34, "Kemerburgaz": 1.20,
        "Eyüp Merkez": 1.10, "İslambey": 1.00, "Defterdar": 0.95,
        "Nişancı": 0.90, "Silahtarağa": 0.85, "Akşemsettin(Eyüp)": 0.88,
        "Rami": 0.81, "Düğmeciler": 0.80, "Topçular": 0.72,
    },
    "Fatih": {
        "Sultanahmet": 1.52, "Balat": 1.39, "Fener": 1.30,
        "Küçükayasofya": 1.20, "Süleymaniye": 1.15, "Zeyrek": 1.10,
        "Fatih Merkez": 1.00, "Beyazıt": 1.00, "Molla Gürani": 0.90,
        "Karagümrük": 0.90, "Samatya": 0.91, "Cerrahpaşa": 0.85,
        "Aksaray": 0.85, "Vatan": 0.85, "Kumkapı": 0.85, "Çapa": 0.88,
        "Şehremini": 0.82, "Haseki": 0.80, "Yedikule": 0.78,
        "Silivrikapı": 0.75, "Topkapı": 0.73,
    },
    "Bakırköy": {
        "Ataköy 1-4": 1.51, "Ataköy 2-5-6": 1.40, "Ataköy 7-8-9-10": 1.38,
        "Ataköy 5-11": 1.35, "Ataköy": 1.40, "Yeşilköy": 1.24,
        "Florya": 1.15, "İncirli": 0.92, "Bakırköy Merkez": 1.00,
        "Cevizlik": 0.90, "Osmaniye": 0.88, "Zuhuratbaba": 0.85,
        "Yenimahalle(Bak)": 0.85, "Sakızağacı": 0.82, "Şenlik": 0.80,
        "Kartaltepe": 0.73,
    },
    "Zeytinburnu": {
        "Yeşiltepe": 1.40, "Kazlıçeşme": 1.16, "Seyitnizam": 1.01,
        "Beştelsiz": 0.95, "Telsiz": 0.90, "Veliefendi": 0.89,
        "Çırpıcı": 0.88, "Sümer": 0.85, "Nuripaşa": 0.82,
        "Gökalp": 0.80, "Merkezefendi": 0.72,
    },
    "Bahçelievler": {
        "Yenibosna": 1.40, "Yenibosna Merkez": 1.40, "Çobançeşme": 1.05,
        "Bahçelievler Mrk.": 1.09, "Cumhuriyet(Bah)": 1.00,
        "Şirinevler": 0.99, "Hürriyet(Bah)": 0.95, "Siyavuşpaşa": 0.92,
        "Soğanlı": 0.87, "Zafer": 0.88, "Fevzi Çakmak(Bah)": 0.85,
        "Kocasinan": 0.71,
    },
    "Güngören": {
        "Güngören Merkez": 1.41, "Tozkoparan": 1.20, "Mehmetçik": 1.01,
        "Haznedar": 0.92, "Gençosman": 0.90, "Akıncılar": 0.85,
        "Mareşal Çakmak": 0.88, "Güneştepe": 0.80, "Sefaköy": 0.74,
    },
    "Küçükçekmece": {
        "Atakent": 1.42, "Halkalı": 1.17, "Küçükçekmece Mrk.": 1.01,
        "Cennet": 1.00, "Beşyol": 0.95, "İnönü": 0.84,
        "Fatih(Küçük)": 0.90, "Söğütlüçeşme": 0.88, "Kanarya": 0.82,
        "Yarımburgaz": 0.80, "Tevfikbey": 0.72,
    },
    "Bağcılar": {
        "Güneşli": 1.41, "Mahmutbey": 1.21, "Bağcılar Merkez": 1.01,
        "Barbaros(Bağ)": 0.92, "Kirazlı": 0.91, "Fevzi Çakmak(Bağ)": 0.90,
        "Yıldıztepe": 0.88, "100. Yıl": 0.88, "Demirkapı": 0.85,
        "İnönü(Bağ)": 0.85, "Yenimahalle(Bağ)": 0.82,
        "Kazım Karabekir": 0.80, "Sefaköy(Bağ)": 0.72,
    },
    "Avcılar": {
        "Denizköşkler": 1.40, "Avcılar Merkez": 1.21, "Cihangir(Avc)": 1.01,
        "Üniversite": 0.95, "Yeşilkent": 0.90, "Mustafa Kemal Paşa": 0.88,
        "Ambarlı": 0.83, "Gümüşpala": 0.82, "Firuzköy": 0.72,
    },
    "Beylikdüzü": {
        "Büyükşehir": 1.42, "Gürpınar": 1.25, "Sahil": 1.10,
        "Beylikdüzü Mrk.": 1.00, "Yakuplu": 0.90, "Barış": 0.88,
        "Kavaklı": 0.85, "Adnan Kahveci": 0.87, "Dereağzı": 0.80,
        "Cumhuriyet(Bey)": 0.73,
    },
    "Bayrampaşa": {
        "Yıldırım": 1.32, "Muratpaşa": 1.27, "Kartaltepe(Bay)": 1.22,
        "Yenidoğan(Bay)": 1.05, "Bayrampaşa Merkez": 1.00,
        "Altıntepsi": 0.98, "Orta(Bay)": 0.95, "İsmetpaşa": 0.92,
        "Terazidere": 0.88, "Vatan(Bay)": 0.88, "Kocatepe": 0.85,
        "Cevatpaşa": 0.82, "Uncubozköy": 0.82,
    },
    "Gaziosmanpaşa": {
        "Karadeniz": 1.41, "GOP Merkez": 1.12, "Karayolları": 0.95,
        "Fevzi Çakmak(GOP)": 1.00, "Hürriyet(GOP)": 0.92,
        "Barbaros(GOP)": 0.90, "Karlıtepe": 0.85, "Bağlarbaşı(GOP)": 0.85,
        "Mevlana": 0.88, "Yıldıztabya": 0.82, "Pazariçi": 0.80,
        "Sarıgöl": 0.78, "Yenidoğan(GOP)": 0.71,
    },
    "Başakşehir": {
        "Başakşehir 4.etap": 1.40, "Başakşehir 5.etap": 1.32,
        "Bahçeşehir 1.kısım": 1.25, "Bahçeşehir 2.kısım": 1.20,
        "Bahçeşehir": 1.22, "Başak": 1.05, "Başakşehir Mrk.": 1.01,
        "Ziya Gökalp(Başak)": 0.90, "Güvercintepe": 0.80,
        "Kayabaşı": 0.87, "İkitelli": 0.73,
    },
    "Esenyurt": {
        "Esenyurt Merkez": 1.40, "Fatih Mah.(Esen)": 1.14, "Pınar": 1.01,
        "Barbaros Hayrettin": 1.00, "Barbaros Hayrettin Paşa": 1.00,
        "Yenikent": 0.95, "Namık Kemal(Esen)": 0.90, "İnönü(Esen)": 0.88,
        "Saadetdere": 0.88, "Ardıçlı": 0.85, "İncirtepe": 0.82,
        "Kıraç": 0.80, "Mehterçeşme": 0.72,
    },
    "Büyükçekmece": {
        "Kumburgaz": 1.40, "Alkent": 1.30, "Büyükçekmece Mrk.": 1.10,
        "Mimaroba": 1.01, "Bahçelievler(Büy)": 0.90, "Gürpınar(Büy)": 0.85,
        "Fatih(Büy)": 0.85, "Pınartepe": 0.80, "Tepecik": 0.73,
    },
    "Esenler": {
        "Tuna": 1.41, "Esenler Merkez": 1.12, "Havaalanı": 1.00,
        "Davutpaşa(Esen)": 0.90, "Fatih(Esen)": 0.88, "Nenehatun": 0.86,
        "Oruçreis": 0.85, "Birlik": 0.82, "Kazım Karabekir(Es)": 0.82,
        "Kemer": 0.80, "Menderes": 0.72,
    },
    "Sultangazi": {
        "Uğur Mumcu(Sul)": 1.41, "Sultangazi Merkez": 1.12, "Cebeci": 1.01,
        "Cumhuriyet(Sul)": 0.90, "Habibler": 0.84, "50. Yıl": 0.88,
        "Zübeyde Hanım": 0.88, "75. Yıl": 0.85, "Esentepe(Sul)": 0.82,
        "Malkoçoğlu": 0.82, "İsmetpaşa(Sul)": 0.80, "Yayla": 0.78,
        "Gazi": 0.72,
    },
    "Arnavutköy": {
        "Hadımköy": 1.42, "Arnavutköy Merkez": 1.12, "Bolluca": 1.00,
        "Taşoluk": 0.85, "Haraçcı": 0.83, "Yunus Emre(Arn)": 0.82,
        "Dursunköy": 0.80, "İmrahor": 0.73,
    },
    "Silivri": {
        "Silivri Merkez": 1.42, "Selimpaşa": 1.17, "Silivri Çiftlikköy": 1.01,
        "Gümüşyaka": 0.90, "Piri Mehmet Paşa": 0.88, "Alipaşa": 0.82,
        "Fener(Sil)": 0.85, "Ortaköy(Sil)": 0.73,
    },
    "Çatalca": {
        "Çatalca Merkez": 1.41, "Ferhatpaşa(Çat)": 1.17, "Kaleiçi": 1.00,
        "Karacaköy": 0.83, "Elbasan": 0.73,
    },
    # ── ANADOLU YAKASI ──────────────────────────────────────────────
    "Üsküdar": {
        "Çengelköy": 1.55, "Kuzguncuk": 1.44, "Beylerbeyi": 1.32,
        "Burhaniye": 1.28, "Kandilli": 1.25, "Salacak": 1.18,
        "Altunizade": 1.15, "Acıbadem(Üsk)": 1.20, "Üsküdar Merkez": 1.00,
        "Selimiye": 1.00, "Aziz Mahmut Hüdayi": 1.00, "Murat Reis": 0.95,
        "Sultantepe": 0.95, "Ünalan": 0.92, "Bağlarbaşı": 0.92,
        "Cumhuriyet(Üsk)": 0.92, "Bahçelievler(Üsk)": 0.90, "Bulgurlu": 0.88,
        "Ferah": 0.85, "Ümraniye(Üsk)": 0.73,
    },
    "Beykoz": {
        "Anadoluhisarı": 1.95, "Kanlıca": 1.72, "Çubuklu": 1.67,
        "Kavacık": 1.32, "Paşabahçe": 1.37, "Acarkent": 1.15,
        "Beykoz Merkez": 1.00, "Akbaba": 0.95, "Çavuşbaşı": 0.92,
        "Örnekköy": 0.90, "Gümüşsuyu(Bey)": 0.90, "Göztepe(Bey)": 0.88,
        "Yalıköy": 0.85, "Dereseki": 0.85, "Riva": 0.82,
        "Anadolu Kavağı": 0.80, "Polonezköy": 0.80, "Alibeyköy(Bey)": 0.76,
        "Göllü": 0.73,
    },
    "Adalar": {
        "Büyükada": 2.10, "Heybeliada": 1.57, "Burgazada": 1.25,
        "Kınalıada": 1.10, "Adalar Merkez": 0.73,
    },
    "Kadıköy": {
        "Moda": 1.55, "Fenerbahçe": 1.43, "Bağdat Caddesi": 1.35,
        "Caddebostan": 1.27, "Suadiye": 1.25, "Erenköy": 1.15,
        "Caferağa": 1.10, "Acıbadem(Kad)": 1.04, "Osmanağa": 1.05,
        "Göztepe": 1.00, "Rasimpaşa": 1.00, "Yeldeğirmeni": 0.95,
        "Kozyatağı": 0.89, "Hasanpaşa(Kad)": 0.92, "Kadıköy Merkez": 0.83,
        "Fikirtepe": 0.85, "Sahrayıcedit": 0.88, "İçerenköy(Kad)": 0.88,
        "Merdivenköy": 0.85, "Zühtüpaşa": 0.90, "Dumlupınar": 0.82,
        "Bostancı": 0.72,
    },
    "Ataşehir": {
        "İçerenköy": 1.41, "Barbaros": 1.28, "Ataşehir Merkez": 1.13,
        "Yeni Çamlıca": 1.05, "Küçükbakkalköy": 1.00, "Yenişehir(Ata)": 0.92,
        "Esatpaşa": 0.95, "Kayışdağı": 0.87, "Mustafa Kemal(Ata)": 0.88,
        "Mimar Sinan(Ata)": 0.90, "Atatürk(Ata)": 0.85, "Ferhatpaşa": 0.73,
    },
    "Maltepe": {
        "Cevizli": 1.41, "Bağlarbaşı(Mal)": 1.27, "Maltepe Merkez": 1.11,
        "Altayçeşme": 1.00, "İdealtepe": 0.95, "Zümrütevler": 0.90,
        "Girne": 0.88, "Büyükbakkalköy": 0.84, "Feyzullah": 0.85,
        "Fındıklı(Mal)": 0.82, "Gülsuyu": 0.80, "Gülensu": 0.78,
        "Başıbüyük": 0.75, "Aydınevler": 0.73,
    },
    "Kartal": {
        "Kordonboyu": 1.40, "Uğur Mumcu(Kar)": 1.21, "Kartal Merkez": 1.00,
        "Cevizli(Kar)": 0.95, "Atalar": 0.92, "Yakacık": 0.90,
        "Topselvi": 0.88, "Cumhuriyet(Kar)": 0.88, "Orhantepe": 0.85,
        "Soğanlık": 0.85, "Esentepe(Kar)": 0.82, "Hürriyet(Kar)": 0.80,
        "Karlıktepe": 0.78, "Petrol İş": 0.73,
    },
    "Ümraniye": {
        "Ataşehir(Üm)": 1.50, "Namık Kemal(Üm)": 1.37, "Namık Kemal": 1.37,
        "Ihlamurkuyu": 1.37, "Çakmak(Üm)": 1.15, "Site(Üm)": 1.12,
        "Site": 1.12, "Ümraniye Merkez": 1.05, "Esatpaşa(Üm)": 1.00,
        "Alemdağ(Üm)": 0.98, "Alemdağ": 0.98, "Yenidoğan(Üm)": 0.95,
        "Armağanevler": 0.92, "Mustafa Kemal(Üm)": 0.92, "Aşağı Dudullu": 0.90,
        "Atatürk(Ümr)": 0.90, "Dudullu": 0.88, "Yukarı Dudullu": 0.88,
        "İstiklal(Ümr)": 0.88, "Hekimbaşı": 0.85, "Elmalıkent": 0.85,
        "Esenevler": 0.80, "Tantavi": 0.82, "Parseller": 0.82,
        "Çiftlik(Üm)": 0.80,
    },
    "Çekmeköy": {
        "Merkez(Çek)": 1.32, "Hamidiye": 1.27, "Hamidiye(Çek)": 1.27,
        "Reşadiye": 1.27, "Alemdağ(Çek)": 1.08, "Taşdelen": 1.05,
        "Çekmeköy Merkez": 1.00, "Nişantepe": 0.98, "Ekşioğlu": 0.95,
        "Soğukpınar": 0.92, "Cumhuriyet(Çek)": 0.90, "Mimar Sinan(Çek)": 0.88,
        "Çatalmeşe(Çek)": 0.88, "Mehmet Akif(Çek)": 0.85, "Ömerli(Çek)": 0.85,
        "Koçullu": 0.82, "Balaban": 0.78,
    },
    "Pendik": {
        "Kurtköy": 1.52, "Güzelyalı(Pen)": 1.40, "Yenişehir(Pen)": 1.35,
        "İçmeler(Pen)": 1.35, "Kaynarca": 1.22, "Pendik Merkez": 1.10,
        "Kavakpınar": 1.05, "Sapanbağları": 1.00, "Batı(Pen)": 0.98,
        "Batı": 0.98, "Doğu(Pen)": 0.95, "Sülüntepe": 0.90,
        "Yayalar": 0.92, "Ballıca": 0.88, "Velibaba": 0.88,
        "Çamlık": 0.85, "Çamlık(Pen)": 0.85, "Bahçelievler(Pen)": 0.85,
        "Ahmet Yesevi": 0.82, "Esenler(Pen)": 0.82, "Kışlak": 0.82,
        "Ertuğrul Gazi": 0.80, "Güllü Bağlar": 0.80, "Dumlupınar(Pen)": 0.80,
        "Dolayoba": 0.73,
    },
    "Tuzla": {
        "Aydınlı": 1.40, "İçmeler(Tuz)": 1.37, "Aydıntepe": 1.21,
        "Mimar Sinan(Tuz)": 1.15, "Mimar Sinan": 1.15, "Postane": 1.10,
        "Tuzla Merkez": 1.05, "Evliya Çelebi(Tuz)": 1.00, "Cami(Tuz)": 0.98,
        "Şifa": 0.95, "Şifa(Tuz)": 0.95, "Tepeören": 0.90, "Orhanlı": 0.88,
        "Köseköy": 0.85, "Ömerli(Tuz)": 0.82, "Yayla(Tuz)": 0.80,
        "Akfırat": 0.80,
    },
    "Sultanbeyli": {
        "Abdurrahman Gazi": 1.22, "Fatih(Sul)": 1.17, "Yavuz Selim(Sul)": 1.12,
        "Sultanbeyli Merkez": 1.00, "Sultanbeyli Mrk.": 1.00,
        "Mehmet Akif(Sul)": 0.98, "Mimarsinan(Sul)": 0.95, "Akşemsettin": 0.90,
        "Hasanpaşa(Sul)": 0.92, "Orhangazi": 0.85, "Necip Fazıl": 0.88,
        "Turgut Reis": 0.88, "Battalgazi": 0.85, "Cumhuriyet(Sul)": 0.82,
        "Hamidiye(Sul)": 0.82, "Kurna": 0.80, "Mimar Sinan(Sul)": 0.78,
    },
    "Sancaktepe": {
        "Emek": 1.35, "Emek(San)": 1.35, "Sarıgazi(San)": 1.30,
        "Yenidoğan(San)": 1.20, "Adnan Kahveci(San)": 1.08,
        "Sancaktepe Merkez": 1.00, "Sancaktepe Mrk.": 1.00,
        "Mimarsinan(San)": 0.95, "Osmangazi": 0.95, "Eyüp Sultan(San)": 0.98,
        "Hasanpaşa(San)": 0.92, "Fatih(San)": 0.90,
        "Kaynarca(San)": 0.88, "Abdurrahmanpaşa": 0.82, "Meclis": 0.85,
        "Yenişehir(San)": 0.85, "İnönü(San)": 0.85, "Paşaköy": 0.82,
        "Atatürk(San)": 0.88, "Samandıra": 0.82, "Hüseyinli": 0.80,
        "Hint": 0.78,
    },
    "Şile": {
        "Kumbaba": 1.40, "Şile Merkez": 1.35, "Uzunkum": 1.25,
        "Ağva": 1.17, "Balibey": 1.15, "Çayırbaşı(Şil)": 1.00,
        "Hacılı": 0.95, "Meşrutiyet(Şil)": 0.92, "Ağlayankaya": 0.88,
        "Değirmençayırı": 0.88, "Sahilköy": 0.85, "Balören": 0.85,
        "Yeniköy(Şile)": 0.82, "Avcıkoru": 0.82, "Çelebi(Şil)": 0.80,
        "Doğancılı": 0.78, "Kurfallı": 0.75, "Üvezli": 0.73,
    },
}

# ── İlçe → mahalle listesi (selectbox için) ───────────────────────────────────
def get_mahalleler(ilce: str) -> list[str]:
    """İlçe adına göre mahalle listesini döndür; katsayıya göre azalan sıralar."""
    mah = MAHALLE_KATSAYI.get(ilce, {})
    return ["— Mahalle seç (isteğe bağlı) —"] + sorted(
        mah.keys(), key=lambda m: mah[m], reverse=True
    )

def get_mahalle_katsayi(ilce: str, mahalle: str) -> float:
    """Seçilen mahalle için katsayıyı döndür; seçilmemişse 1.00."""
    if not mahalle or mahalle.startswith("—"):
        return 1.00
    return MAHALLE_KATSAYI.get(ilce, {}).get(mahalle, 1.00)

# ── ILCE_KIRA — Endeksa Mart 2026 ilçe ort. TL/m² × 90 m² referans ──────────
ILCE_KIRA = {
    "Beşiktaş":    53000, "Sarıyer":     53000, "Bakırköy":    53000,
    "Zeytinburnu": 52500, "Beyoğlu":     44000, "Şişli":       40500,
    "Eyüpsultan":  37500, "Kağıthane":   35000, "Fatih":       29500,
    "Bahçelievler":35500, "Küçükçekmece":33000, "Güngören":    29500,
    "Başakşehir":  32000, "Bayrampaşa":  33000, "Bağcılar":    26500,
    "Avcılar":     27500, "Beylikdüzü":  26000, "Gaziosmanpaşa":26500,
    "Esenler":     22500, "Esenyurt":    20500, "Sultangazi":  21500,
    "Büyükçekmece":25500, "Arnavutköy":  21500, "Silivri":     22000,
    "Çatalca":     26500, "Kadıköy":     58500, "Adalar":      40000,
    "Üsküdar":     38000, "Beykoz":      35500, "Ataşehir":    41500,
    "Maltepe":     39500, "Kartal":      37000, "Ümraniye":    38000,
    "Çekmeköy":    30500, "Pendik":      34500, "Tuzla":       32000,
    "Sancaktepe":  28000, "Sultanbeyli": 26500, "Şile":        25500,
}

min_k = min(ILCE_KIRA.values())
max_k = max(ILCE_KIRA.values())

DEMO_KATSAYI = {
    "Kadıköy":     1.63, "Beşiktaş":    1.48, "Bakırköy":    1.48,
    "Sarıyer":     1.48, "Zeytinburnu": 1.46, "Beyoğlu":     1.23,
    "Şişli":       1.13, "Adalar":      1.11, "Ataşehir":    1.15,
    "Maltepe":     1.10, "Üsküdar":     1.06, "Kartal":      1.03,
    "Eyüpsultan":  1.05, "Ümraniye":    1.05, "Kağıthane":   0.98,
    "Bahçelievler":0.98, "Küçükçekmece":0.92, "Beykoz":      0.99,
    "Başakşehir":  0.89, "Bayrampaşa":  0.92, "Pendik":      0.96,
    "Tuzla":       0.89, "Çekmeköy":    0.85, "Fatih":       0.82,
    "Güngören":    0.82, "Sancaktepe":  0.78, "Bağcılar":    0.74,
    "Avcılar":     0.77, "Büyükçekmece":0.70, "Gaziosmanpaşa":0.73,
    "Beylikdüzü":  0.72, "Sultanbeyli": 0.74, "Çatalca":     0.73,
    "Esenler":     0.63, "Şile":        0.71, "Sultangazi":  0.59,
    "Arnavutköy":  0.60, "Silivri":     0.62, "Esenyurt":    0.57,
}

# ── BAŞLIK ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🏙️ İstanbul Kira Tahmini</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Yapay Zeka ile Evinizin Tahmini Kirasını Öğrenin</p>", unsafe_allow_html=True)
if not model_loaded:
    st.warning("⚠️ Model yüklenemedi — demo modda çalışıyor.")

tab1, tab2 = st.tabs(["💰 Kira Tahmini", "🗺️ İstanbul Kira Haritası"])

# ══════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        # İlçe seçimi — değişince mahalle sıfırlanır
        secilen_ilce = st.selectbox(
            "📍 İlçe", DISTRICTS,
            index=DISTRICTS.index(st.session_state.ilce), key="ilce"
        )
        # Mahalle listesi ilçeye göre dinamik
        mahalle_listesi = get_mahalleler(secilen_ilce)
        # Eğer önceki mahalle yeni ilçede yoksa sıfırla
        prev_mah = st.session_state.get("mahalle", "")
        default_mah_idx = (
            mahalle_listesi.index(prev_mah)
            if prev_mah in mahalle_listesi else 0
        )
        secilen_mahalle = st.selectbox(
            "🏘️ Mahalle", mahalle_listesi,
            index=default_mah_idx, key="mahalle"
        )
        # Katsayı rozeti
        katsayi = get_mahalle_katsayi(secilen_ilce, secilen_mahalle)
        ilce_m2_fiyat = ILCE_ORT.get(secilen_ilce, 370)
        mahalle_m2 = round(ilce_m2_fiyat * max(MIN_KATSAYI, min(MAX_KATSAYI, katsayi)))
        if not secilen_mahalle.startswith("—"):
            st.markdown(
                f"<div class='mahalle-badge'>"
                f"📐 Mahalle ort. TL/m²: <b>₺{mahalle_m2:,}</b> "
                f"&nbsp;·&nbsp; katsayı: <b>{katsayi:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.slider("🛏️ Oda Sayısı", 1, 10, key="oda")
        st.slider("🛋️ Salon Sayısı", 0, 3, key="salon")
        st.number_input("📐 Metrekare (m²)", 20, 1000, step=5, key="m2")

    with c2:
        st.slider("🏗️ Bina Yaşı (yıl)", 0, 50, key="yas")
        st.slider("🚪 Bulunduğu Kat", -2, 30, key="kat")
        bins = [-np.inf, -1, 0, 2, 7, 10, np.inf]
        kat_kat = pd.cut([st.session_state.kat], bins=bins, labels=KAT_LABELS)[0]
        st.markdown(f"<div class='kat-badge'>🏢 Kat Kategorisi: <b>{kat_kat}</b></div>", unsafe_allow_html=True)
        st.selectbox("🔥 Isıtma Tipi", list(ISITMA_MAP.keys()), key="isitma")
        st.checkbox("🛋️ Eşyalı mı?", key="esyali")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💰 Kira Tahmini Yap", type="primary"):
        ilce    = st.session_state.ilce
        mahalle = st.session_state.mahalle
        oda     = st.session_state.oda
        salon   = st.session_state.salon
        m2      = st.session_state.m2
        yas     = st.session_state.yas
        kat     = st.session_state.kat
        isitma  = ISITMA_MAP[st.session_state.isitma]
        esyali  = float(int(st.session_state.esyali))

        # Mahalle katsayısını hesaba kat
        mah_katsayi = get_mahalle_katsayi(ilce, mahalle)
        mah_katsayi = max(MIN_KATSAYI, min(MAX_KATSAYI, mah_katsayi))

        if model_loaded:
            try:
                row = {
                    "Metrekare":[float(m2)],"OdaSayisi":[float(oda)],"SalonSayisi":[float(salon)],
                    "Kat":[float(kat)],"Yapı Yaşı":[float(yas)],"Isıtma":[isitma],"Esya":[esyali],
                    "Villa":[0.0],"Dubleks":[0.0],"Tripleks":[0.0],"Yali":[0.0],
                    "BogazManzarali":[0.0],"Manzarali":[0.0],"SahileYakinlik":[0.0],
                    "Bahce":[0.0],"Havuz":[0.0],"UlasimSkor":[0.0],f"Ilce_{ilce}":[1.0],
                }
                df_in = pd.DataFrame(row)
                for col in feature_columns:
                    if col not in df_in.columns:
                        df_in[col] = 0.0
                df_in = df_in[feature_columns]
                base_pred = float(model.predict(scaler.transform(df_in), verbose=0)[0][0])
                # Mahalle katsayısı model tahminine uygulanır (1.00'den sapma oranı)
                pred = base_pred * mah_katsayi
                st.session_state.sonuc = pred
                st.session_state.tahmin_yapildi = True
            except Exception as e:
                st.error(f"Hata: {e}")
        else:
            # Demo mod — ilçe TL/m² × mahalle katsayısı
            ilce_m2_ort = ILCE_ORT.get(ilce, 370)
            efektif_m2  = ilce_m2_ort * mah_katsayi
            esya_bonus  = 1.08 if esyali else 1.0
            yas_iskonto = max(0.75, 1.0 - yas * 0.005)
            pred = max(
                int(m2 * efektif_m2 * esya_bonus * yas_iskonto / 500) * 500,
                15000
            )
            st.session_state.sonuc = pred
            st.session_state.tahmin_yapildi = True

    if st.session_state.tahmin_yapildi and st.session_state.sonuc:
        pred    = st.session_state.sonuc
        ilce    = st.session_state.ilce
        mahalle = st.session_state.mahalle
        m2      = st.session_state.m2
        oda     = st.session_state.oda
        salon   = st.session_state.salon
        demo_label = "" if model_loaded else "<span style='font-size:0.7rem;background:rgba(251,191,36,0.2);padding:2px 8px;border-radius:20px;margin-left:8px;'>DEMO</span>"
        mah_label = f" / {mahalle}" if mahalle and not mahalle.startswith("—") else ""
        st.markdown(f"""
        <div class='result-box'>
            <div style='font-size:1rem;color:#94a3b8;margin-bottom:0.5rem;'>Tahmini Aylık Kira {demo_label}</div>
            <div class='result-price'>₺{pred:,.0f}</div>
            <div style='color:#64748b;margin-top:0.8rem;font-size:0.9rem;'>
                📍 {ilce}{mah_label} &nbsp;•&nbsp; {m2} m² &nbsp;•&nbsp; {oda}+{salon}
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class='stat-card'><div style='color:#8da9c4;font-size:0.75rem;text-transform:uppercase;'>📐 m² Fiyatı</div><div style='color:white;font-size:1.3rem;font-weight:600;'>₺{pred/m2:,.0f} / m²</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class='stat-card' style='border-left-color:#fbbf24;'><div style='color:#8da9c4;font-size:0.75rem;text-transform:uppercase;'>🏠 Oda Başına</div><div style='color:white;font-size:1.3rem;font-weight:600;'>₺{pred/(oda+salon):,.0f}</div></div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class='stat-card' style='border-left-color:#f472b6;'><div style='color:#8da9c4;font-size:0.75rem;text-transform:uppercase;'>📅 Yıllık Toplam</div><div style='color:white;font-size:1.3rem;font-weight:600;'>₺{pred*12:,.0f}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — CHOROPLETH HARİTA
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <p style='color:#94a3b8;font-size:0.9rem;margin-bottom:1rem;'>
    İstanbul ilçe sınırları, Endeksa Mart 2026 ortalama kira fiyatına göre renklendiriliyor.
    Açık sarı → ucuz, koyu turuncu → pahalı. İlçelere tıklayın veya üzerine gelin.
    </p>""", unsafe_allow_html=True)

    kira_json    = json.dumps(ILCE_KIRA)

    fallback_coords = {
        "Adalar":(40.8761,29.0923),"Arnavutköy":(41.1854,28.7397),
        "Ataşehir":(40.9833,29.1333),"Avcılar":(40.9794,28.7219),
        "Bağcılar":(41.0394,28.8561),"Bahçelievler":(40.9978,28.8422),
        "Bakırköy":(40.9811,28.8722),"Başakşehir":(41.0936,28.8014),
        "Bayrampaşa":(41.0531,28.9108),"Beşiktaş":(41.0436,29.0072),
        "Beykoz":(41.1289,29.1017),"Beylikdüzü":(40.9822,28.6394),
        "Beyoğlu":(41.0336,28.9775),"Büyükçekmece":(41.0214,28.5822),
        "Çatalca":(41.1436,28.4611),"Çekmeköy":(41.0353,29.1806),
        "Esenler":(41.0436,28.8758),"Esenyurt":(41.0286,28.6728),
        "Eyüpsultan":(41.0664,28.9336),"Fatih":(41.0186,28.9397),
        "Gaziosmanpaşa":(41.0664,28.9122),"Güngören":(41.0181,28.8736),
        "Kadıköy":(40.9833,29.0833),"Kağıthane":(41.0767,28.9733),
        "Kartal":(40.9081,29.1919),"Küçükçekmece":(41.0022,28.7772),
        "Maltepe":(40.9353,29.1369),"Pendik":(40.8769,29.2322),
        "Sancaktepe":(41.0014,29.2297),"Sarıyer":(41.1664,29.0522),
        "Silivri":(41.0736,28.2481),"Sultanbeyli":(40.9611,29.2694),
        "Sultangazi":(41.1069,28.8636),"Şile":(41.1769,29.6119),
        "Şişli":(41.0603,28.9872),"Tuzla":(40.8158,29.3006),
        "Ümraniye":(41.0167,29.1167),"Üsküdar":(41.0231,29.0153),
        "Zeytinburnu":(41.0003,28.9019),
    }
    fallback_json = json.dumps({k: list(v) for k, v in fallback_coords.items()})

    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#0f172a; }}
        #map {{ width:100%; height:600px; border-radius:12px; }}
        .leaflet-control-zoom a {{ background:#1e293b !important; color:#fbbf24 !important; border-color:#334155 !important; }}
        .info-box {{ background:rgba(15,23,42,0.92); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px 14px; color:#e2e8f0; font-family:Inter,sans-serif; font-size:13px; min-width:180px; }}
        .info-box h4 {{ margin:0 0 4px; color:#fbbf24; font-size:14px; }}
        .legend {{ background:rgba(15,23,42,0.92); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:12px 16px; color:#e2e8f0; font-family:Inter,sans-serif; font-size:12px; min-width:200px; }}
        .legend-bar {{ width:100%; height:14px; border-radius:7px; background:linear-gradient(to right,#fef9c3,#f59e0b,#92400e); margin:8px 0 4px; }}
        .legend-labels {{ display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; }}
    </style>
    </head>
    <body>
    <div id="map"></div>
    <script>
    var kiraData = {kira_json};
    var fallbackCoords = {fallback_json};
    var minK = {min_k};
    var maxK = {max_k};
    function getColor(kira) {{
        var t = (kira - minK) / (maxK - minK);
        var stops = [[254,249,195],[245,158,11],[146,64,14]];
        var seg, tt;
        if (t < 0.5) {{ seg=0; tt=t*2; }} else {{ seg=1; tt=(t-0.5)*2; }}
        var c1=stops[seg], c2=stops[seg+1];
        return 'rgb('+Math.round(c1[0]+(c2[0]-c1[0])*tt)+','+Math.round(c1[1]+(c2[1]-c1[1])*tt)+','+Math.round(c1[2]+(c2[2]-c1[2])*tt)+')';
    }}
    var map = L.map('map',{{center:[41.015,28.97],zoom:10}});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
        attribution:'&copy; CARTO', subdomains:'abcd', maxZoom:19
    }}).addTo(map);
    var info = L.control({{position:'topleft'}});
    info.onAdd = function() {{
        this._div = L.DomUtil.create('div','info-box');
        this.update(); return this._div;
    }};
    info.update = function(name, kira) {{
        this._div.innerHTML = name
            ? '<h4>📍 '+name+'</h4><span style="font-size:1.2rem;font-weight:700;color:#fbbf24;">₺'+kira.toLocaleString('tr-TR')+'</span><br><span style="color:#94a3b8;font-size:0.75rem;">Ort. Aylık Kira (~90 m²)</span>'
            : '<span style="color:#94a3b8;">Bir ilçenin üzerine gelin</span>';
    }};
    info.addTo(map);
    var legend = L.control({{position:'bottomright'}});
    legend.onAdd = function() {{
        var d = L.DomUtil.create('div','legend');
        d.innerHTML = '<b>🏠 Endeksa Mart 2026 — Ort. Kira</b><div class="legend-bar"></div><div class="legend-labels"><span>₺{min_k:,} (ucuz)</span><span>₺{max_k:,} (pahalı)</span></div><div style="margin-top:6px;color:#64748b;font-size:10px;text-align:center;">~90 m² referans daire • İlçeye tıkla</div>';
        return d;
    }};
    legend.addTo(map);
    var geojsonLayer;
    function styleFeature(feature) {{
        var name = feature.properties.name;
        var kira = kiraData[name];
        return {{
            fillColor  : kira ? getColor(kira) : '#1e293b',
            fillOpacity: kira ? 0.80 : 0.15,
            color      : 'rgba(255,255,255,0.25)',
            weight     : 1.5
        }};
    }}
    function onEach(feature, layer) {{
        var name = feature.properties.name;
        var kira = kiraData[name] || null;
        layer.on({{
            mouseover: function(e) {{
                e.target.setStyle({{weight:3,color:'#fbbf24',fillOpacity:0.95}});
                e.target.bringToFront();
                info.update(name, kira);
            }},
            mouseout: function(e) {{
                geojsonLayer.resetStyle(e.target);
                info.update();
            }},
            click: function(e) {{
                if (!kira) return;
                L.popup().setLatLng(e.latlng).setContent(
                    '<div style="font-family:Inter,sans-serif;min-width:160px;">' +
                    '<b style="font-size:1rem;">📍 '+name+'</b><br><br>' +
                    '<span style="color:#888;font-size:0.8rem;">Ort. Aylık Kira (~90 m²)</span><br>' +
                    '<span style="font-size:1.4rem;font-weight:700;color:#92400e;">₺'+kira.toLocaleString('tr-TR')+'</span>' +
                    '<div style="margin-top:8px;background:linear-gradient(to right,#fef9c3,#f59e0b,#92400e);height:6px;border-radius:3px;"></div>' +
                    '<div style="display:flex;justify-content:space-between;font-size:9px;color:#aaa;margin-top:2px;"><span>Ucuz</span><span>Pahalı</span></div>' +
                    '</div>'
                ).openOn(map);
            }}
        }});
    }}
    var GEOJSON_URL = 'https://raw.githubusercontent.com/alpers/istanbul-districts/master/istanbul_districts.geojson';
    fetch(GEOJSON_URL)
      .then(function(r) {{
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
      }})
      .then(function(data) {{
          geojsonLayer = L.geoJSON(data, {{
              style: styleFeature,
              onEachFeature: onEach
          }}).addTo(map);
          if (geojsonLayer.getBounds().isValid()) {{
              map.fitBounds(geojsonLayer.getBounds(), {{padding:[20,20]}});
          }}
      }})
      .catch(function(err) {{
          console.warn('İlk GeoJSON başarısız, yedek deneniyor:', err);
          var BACKUP_URL = 'https://raw.githubusercontent.com/turkeymap/istanbul/main/istanbul-districts.geojson';
          fetch(BACKUP_URL)
            .then(function(r) {{ return r.json(); }})
            .then(function(data) {{
                geojsonLayer = L.geoJSON(data, {{
                    style: styleFeature,
                    onEachFeature: onEach
                }}).addTo(map);
                if (geojsonLayer.getBounds().isValid()) {{
                    map.fitBounds(geojsonLayer.getBounds(), {{padding:[20,20]}});
                }}
            }})
            .catch(function() {{
                console.warn('GeoJSON yüklenemedi, circleMarker fallback');
                Object.entries(fallbackCoords).forEach(function(entry) {{
                    var name = entry[0], coords = entry[1];
                    var kira = kiraData[name];
                    if (!kira) return;
                    L.circleMarker(coords, {{
                        radius:18, fillColor:getColor(kira),
                        color:'rgba(255,255,255,0.3)', fillOpacity:0.88, weight:1.5
                    }}).bindTooltip(name+'<br>₺'+kira.toLocaleString('tr-TR'), {{permanent:false}})
                      .addTo(map);
                }});
            }});
      }});
    </script>
    </body>
    </html>
    """

    st.components.v1.html(map_html, height=620, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 İlçe Bazında Ortalama Kira Tablosu — Endeksa Mart 2026")
    df_kira = pd.DataFrame([
        {
            "İlçe": k,
            "Ort. TL/m²": f"₺{round(v/90):,}",
            "Ref. Aylık Kira (~90 m²)": f"₺{v:,}",
        }
        for k, v in sorted(ILCE_KIRA.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(df_kira, use_container_width=True, hide_index=True)

# ── ALT BİLGİ ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#475569;font-size:0.8rem;
border-top:1px solid rgba(255,255,255,0.06);padding-top:1.2rem;'>
    🏙️ İstanbul Kira Tahmini &nbsp;·&nbsp;
    Veri: <b style='color:#8da9c4;'>Endeksa Mart 2026</b> &nbsp;·&nbsp;
    Model: <b style='color:#8da9c4;'>istanbul_rent_model.h5</b> &nbsp;·&nbsp;
    Geliştirici: <b style='color:#8da9c4;'>Tuğçe Başyiğit</b>
</p>""", unsafe_allow_html=True)

with st.sidebar:
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
    2. **Mahalleyi** seçin (isteğe bağlı)
    3. **Oda & m²** bilgilerini girin
    4. **Bina yaşı & kat** belirtin
    5. **Tahmin Yap** butonuna basın
    6. **Harita** sekmesinde ilçe kiralarını inceleyin
    """)
    st.markdown("---")
    st.markdown("### 📌 Veri Kaynağı")
    st.markdown("""
    - **Endeksa.com** Mart 2026
    - 39 ilçe mahalle bazlı analiz
    - Kentsel gerçek kalibrasyonu
    """)
