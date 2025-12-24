# 🏥 Polip Segmentasyon - UNet++

Kolonoskopi görüntülerinden **polipleri otomatik olarak tespit eden** derin öğrenme projesi.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarisi](#-model-mimarisi)
- [Sonuçlar](#-sonuçlar)
- [Web Arayüzü](#-web-arayüzü)

---

## 🎯 Proje Hakkında

Bu proje, **CVC-Clinic** veri seti kullanılarak kolonoskopi görüntülerinden polip segmentasyonu yapmak için geliştirilmiştir. Polipler, bağırsaklarda oluşan ve kansere dönüşebilecek anormal doku büyümeleridir. Erken tespit hayat kurtarır!

### Veri Seti

- **Kaynak**: CVC-Clinic Database
- **Görüntü Sayısı**: 612 frame
- **Format**: PNG ve TIFF
- **İçerik**: Kolonoskopi görüntüleri + manuel işaretlenmiş polip maskeleri

---

## ✨ Özellikler

- 🧠 **UNet++ Mimarisi** - Nested skip connections ile gelişmiş segmentasyon
- 🎯 **Deep Supervision** - Daha iyi gradient akışı ve öğrenme
- 📊 **IoU Metriği** - Intersection over Union değerlendirmesi
- 🔄 **Data Augmentation** - Albumentations ile veri çeşitlendirme
- 🖥️ **Gradio Arayüzü** - Web tabanlı kullanıcı arayüzü
- ⚡ **GPU Desteği** - CUDA ile hızlandırılmış eğitim

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilen)
- ~6GB GPU belleği

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/Ozgun1234/polip-segmantasyon.git
cd polip-segmantasyon

# 2. Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Bağımlılıkları yükle
pip install -r Code/source/requirements.txt

# 4. Git LFS dosyalarını çek (veri seti için)
git lfs pull
```

---

## 💻 Kullanım

### Model Eğitimi

```bash
cd Code/source

# Konfigürasyonu düzenle (epochs, batch_size vb.)
nano config.yaml

# Eğitimi başlat
python3 engine.py
```

### Konfigürasyon (`config.yaml`)

```yaml
extn: .png
epochs: 300 # Epoch sayısı
im_width: 384 # Görüntü genişliği
im_height: 288 # Görüntü yüksekliği
model_path: ../output/models/model.pth
```

### Tahmin Yapma

```bash
python3 engine.py --test_img ../input/PNG/Original/50.png
```

---

## 🧠 Model Mimarisi

### UNet++

```
Encoder (Daraltma)          Decoder (Genişletme)
──────────────────────────────────────────────
x0_0 ──────────────────────────────────→ x0_4 (Çıktı)
  ↓                                        ↑
x1_0 ────────→ x0_1 ────→ x0_2 ─→ x0_3 ───┘
  ↓              ↑          ↑       ↑
x2_0 ───→ x1_1 ──┴──→ x1_2 ─┴─→ x1_3─┘
  ↓          ↑           ↑        ↑
x3_0 ─→ x2_1 ┴──→ x2_2 ──┴───────┘
  ↓        ↑         ↑
x4_0 ──→ x3_1 ───────┘
```

### Neden UNet++?

| Özellik               | UNet       | UNet++          |
| --------------------- | ---------- | --------------- |
| Skip Connections      | Basit      | Nested (İç içe) |
| Feature Fusion        | Tek seviye | Çok seviyeli    |
| Deep Supervision      | ❌         | ✅              |
| Segmentasyon Kalitesi | İyi        | **Daha İyi**    |

---

## 📊 Sonuçlar

| Metrik            | Değer  |
| ----------------- | ------ |
| **IoU Score**     | ~0.85+ |
| **Training Loss** | < 0.1  |
| **Epochs**        | 300    |
| **Batch Size**    | 8      |

---

## 🖥️ Web Arayüzü

Gradio ile kullanıcı dostu bir web arayüzü:

```bash
# Gradio'yu yükle
pip install gradio>=4.0.0

# Arayüzü başlat
cd Code/source
python3 app.py
```

Tarayıcıda aç: `http://localhost:7860`

### Özellikler

- 📷 Görüntü yükleme
- 🔍 Tek tıkla analiz
- 🎯 Poliplerin yeşil renkte işaretlenmesi
- 📊 Örnek görüntüler

---

## 📁 Proje Yapısı

```
polip-segmantasyon/
├── Code/
│   ├── input/                  # Eğitim verileri
│   │   ├── PNG/
│   │   │   ├── Original/       # Kolonoskopi görüntüleri
│   │   │   └── Ground Truth/   # Polip maskeleri
│   │   └── TIF/
│   ├── output/
│   │   └── models/             # Eğitilmiş modeller
│   └── source/
│       ├── ML_Pipeline/        # Model ve yardımcı fonksiyonlar
│       │   ├── network.py      # UNet++ mimarisi
│       │   ├── dataset.py      # Veri yükleme
│       │   ├── train.py        # Eğitim
│       │   └── validate.py     # Doğrulama
│       ├── engine.py           # Ana script
│       ├── app.py              # Gradio arayüzü
│       └── config.yaml         # Konfigürasyon
├── data/                       # Ham veri
└── README.md
```

---

## 🛠️ Teknoloji Stack

| Kütüphane          | Kullanım          |
| ------------------ | ----------------- |
| **PyTorch**        | Derin öğrenme     |
| **OpenCV**         | Görüntü işleme    |
| **Albumentations** | Data augmentation |
| **Gradio**         | Web arayüzü       |
| **NumPy/Pandas**   | Veri işleme       |

---

## ⚠️ Önemli Notlar

1. **GPU Önerisi**: Eğitim GPU ile çok daha hızlı olur
2. **Bellek**: En az 6GB GPU belleği önerilir
3. **Batch Size**: GPU belleğine göre ayarlayın (4-16 arası)

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

## 🙏 Teşekkürler

- CVC-Clinic Database sağlayıcıları
- UNet++ paper yazarları
- PyTorch ekibi

---

⚠️ _Bu araç sadece eğitim ve araştırma amaçlıdır. Tıbbi teşhis için kullanılamaz._
