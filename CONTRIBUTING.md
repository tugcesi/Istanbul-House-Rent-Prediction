# 🤝 Katkı Rehberi / Contributing Guide

Projeye katkıda bulunmak istediğiniz için teşekkürler!  
Thank you for your interest in contributing to this project!

---

## 🚀 Nasıl Katkı Sağlayabilirim? / How to Contribute

### 1. Fork & Clone

Repoyu fork'layın ve yerel makinenize klonlayın:

```bash
git clone https://github.com/<your-username>/Istanbul-House-Rent-Prediction.git
cd Istanbul-House-Rent-Prediction
```

### 2. Yeni Branch Oluşturun / Create a New Branch

Her özellik veya düzeltme için ayrı bir branch oluşturun:

```bash
git checkout -b feature/your-feature-name
# veya / or
git checkout -b fix/your-bug-fix
```

### 3. Değişikliklerinizi Yapın / Make Your Changes

- Kodunuzu yazın veya düzenleyin
- Mevcut testlerin geçtiğinden emin olun
- Gerekiyorsa yeni testler ekleyin

### 4. Commit & Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 5. Pull Request Açın / Open a Pull Request

GitHub üzerinden `main` branch'ine bir Pull Request (PR) açın. PR açıklamasında:

- Ne değiştirdiğinizi açıklayın
- Neden bu değişikliğin gerekli olduğunu belirtin
- İlgili issue numarasını etiketleyin (varsa): `Closes #123`

---

## 🐛 Issue Açma Rehberi / Reporting Issues

Bir hata buldunuz mu ya da yeni bir özellik mi öneriyorsunuz?

1. [Issues](https://github.com/tugcesi/Istanbul-House-Rent-Prediction/issues) sayfasına gidin
2. **"New issue"** butonuna tıklayın
3. Şablonu doldurun:
   - **Başlık:** Kısa ve açıklayıcı olsun
   - **Açıklama:** Hatayı veya özelliği detaylıca anlatın
   - **Adımlar:** Hatayı yeniden oluşturmak için adımları yazın (bug ise)
   - **Beklenen / Gerçekleşen Davranış:** Ne olmasını bekliyordunuz, ne oldu?
   - **Ortam:** Python ve kütüphane versiyonlarınızı paylaşın

---

## 📋 Kod Stili / Code Style

Bu proje **PEP 8** standartlarını takip etmektedir:

- Girinti: **4 boşluk** (tab değil)
- Maksimum satır uzunluğu: **79 karakter** (PEP 8 standardı)
- Fonksiyon ve değişken isimleri: `snake_case`
- Sınıf isimleri: `PascalCase`
- Sabit değerler: `UPPER_CASE`
- Türkçe değişken isimleri kabul edilir (proje Türkçe verilere odaklıdır)

Kod kalitesini kontrol etmek için:

```bash
pip install flake8
flake8 tugce_simulated_data_v01.py
```

---

## 🔍 İnceleme Süreci / Review Process

- PR'lar proje sahibi tarafından incelenir
- Geri bildirim verilirse, değişiklikleri yapıp commit'leyin — PR otomatik güncellenecektir
- Onaylandıktan sonra `main` branch'ine merge edilir

---

## 💡 Katkı Fikirleri / Contribution Ideas

- 🗺️ Yeni ilçe/mahalle verisi ekleme
- 📈 Model performansını artırma (hyperparameter tuning)
- 🧹 Veri temizleme ve iyileştirme
- 📊 Yeni görselleştirmeler
- 🌍 İngilizce çeviri/dokümantasyon
- 🐛 Bug düzeltmeleri

---

## 📄 Lisans / License

Katkıda bulunarak, değişikliklerinizin projenin [MIT Lisansı](LICENSE) altında yayınlanmasını kabul etmiş olursunuz.

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
