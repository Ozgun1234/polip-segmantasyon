# 📁 Source - Kaynak Kodlar

Bu klasör projenin ana kaynak kodlarını içerir.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Model eğitimi başlat
python3 engine.py

# 3. Web arayüzünü başlat
python3 app.py
```

---

## 📂 Dosya Yapısı

| Dosya              | Açıklama                       |
| ------------------ | ------------------------------ |
| `engine.py`        | Ana eğitim ve tahmin scripti   |
| `app.py`           | Gradio web arayüzü             |
| `config.yaml`      | Model ve eğitim konfigürasyonu |
| `requirements.txt` | Python bağımlılıkları          |

### ML_Pipeline/

| Dosya         | Açıklama                                  |
| ------------- | ----------------------------------------- |
| `network.py`  | UNet++ model mimarisi                     |
| `dataset.py`  | Veri seti yükleme sınıfı                  |
| `train.py`    | Eğitim fonksiyonu                         |
| `validate.py` | Doğrulama fonksiyonu                      |
| `predict.py`  | Tahmin fonksiyonları                      |
| `utils.py`    | Yardımcı fonksiyonlar (IoU, AverageMeter) |

---

## ⚙️ Konfigürasyon

`config.yaml` dosyasını düzenleyerek ayarları değiştirebilirsin:

```yaml
extn: .png # Görüntü uzantısı
epochs: 300 # Epoch sayısı
im_width: 384 # Görüntü genişliği
im_height: 288 # Görüntü yüksekliği
model_path: ../output/models/model.pth
log_path: ../output/models/logs/logs.csv
image_path: ../input/PNG/Original
mask_path: ../input/PNG/Ground Truth
output_path: ../output/prediction.png
```

---

## 🔧 Komutlar

### Eğitim

```bash
python3 engine.py
```

### Tahmin (Tek Görüntü)

```bash
python3 engine.py --test_img ../input/PNG/Original/50.png
```

### Web Arayüzü

```bash
python3 app.py
# Tarayıcıda: http://localhost:7860
```

---

## 📊 Eğitim Parametreleri

| Parametre     | Değer             | Açıklama                  |
| ------------- | ----------------- | ------------------------- |
| Batch Size    | 8                 | GPU belleğine göre ayarla |
| Learning Rate | 1e-3              | Adam optimizer            |
| Weight Decay  | 1e-4              | Regularization            |
| Optimizer     | Adam              | Hızlı yakınsama           |
| Loss          | BCEWithLogitsLoss | Binary segmentation için  |

---

## 💡 İpuçları

1. **GPU Bellek Hatası**: `engine.py` içinde `batch_size`'ı düşür (4 veya 2)
2. **Hızlı Test**: `config.yaml` içinde `epochs: 5` yaparak hızlıca test et
3. **Model Kaydetme**: En iyi IoU değerine sahip model otomatik kaydedilir
