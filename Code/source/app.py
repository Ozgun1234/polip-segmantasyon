"""
🏥 Polip Segmentasyon Web Arayüzü
Kolonoskopi görüntülerinden polip tespiti yapan UNet++ modeli için Gradio arayüzü
"""

import gradio as gr
import numpy as np
import torch
import cv2
import yaml
from PIL import Image
import albumentations as A

from ML_Pipeline.network import UNetPP


# Konfigürasyon yükle
with open("config.yaml") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model_path"]
IM_WIDTH = config["im_width"]
IM_HEIGHT = config["im_height"]

# Transform tanımla
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
])


def load_model():
    """Eğitilmiş modeli yükle"""
    model = UNetPP(1, 3, deep_supervision=True)
    
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.cuda()
    
    model.eval()
    return model


# Global model yükle
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


def predict(image):
    """Görüntüden polip segmentasyonu yap"""
    
    if not model_loaded:
        return None, f"❌ Model yüklenemedi: {model_error}"
    
    if image is None:
        return None, "⚠️ Lütfen bir görüntü yükleyin"
    
    try:
        # PIL Image'ı numpy array'e çevir
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Orijinal boyutu kaydet
        original_h, original_w = image_np.shape[:2]
        
        # Transform uygula
        transformed = transform(image=image_np)
        img = transformed["image"]
        
        # Model için hazırla
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Batch dimension ekle
        img_tensor = torch.from_numpy(img)
        
        # GPU varsa kullan
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Tahmin yap
        with torch.no_grad():
            output = model(img_tensor)
            mask = output[-1]  # Deep supervision'dan son çıktı
        
        # Maskeyi işle
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)  # Batch ve channel dimension'ları kaldır
        
        # Binary maskeye çevir
        mask_binary = np.zeros_like(mask)
        mask_binary[mask > -2.5] = 255
        mask_binary[mask <= -2.5] = 0
        
        # Orijinal boyuta döndür
        mask_resized = cv2.resize(mask_binary, (original_w, original_h))
        
        # Overlay oluştur (orijinal görüntü + maske)
        overlay = image_np.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = mask_resized  # Yeşil kanal
        
        # Blend
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
        
        return overlay, "✅ Segmentasyon başarılı!"
        
    except Exception as e:
        return None, f"❌ Hata: {str(e)}"


# Gradio Arayüzü
with gr.Blocks(
    title="🏥 Polip Segmentasyon",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="emerald",
    )
) as demo:
    
    gr.Markdown("""
    # 🏥 Polip Segmentasyon Sistemi
    
    Bu araç, kolonoskopi görüntülerinden **polipleri otomatik olarak tespit** eder.
    
    **Nasıl kullanılır:**
    1. Kolonoskopi görüntüsü yükleyin
    2. "Analiz Et" butonuna tıklayın
    3. Tespit edilen polipler yeşil renkte işaretlenir
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="📷 Kolonoskopi Görüntüsü",
                type="pil",
                height=350
            )
            
            analyze_btn = gr.Button(
                "🔍 Analiz Et",
                variant="primary",
                size="lg"
            )
            
        with gr.Column():
            output_image = gr.Image(
                label="🎯 Segmentasyon Sonucu",
                height=350
            )
            status_text = gr.Textbox(
                label="Durum",
                interactive=False
            )
    
    gr.Markdown("""
    ---
    
    ### 📊 Örnek Görüntüler
    """)
    
    # Örnek görüntüler
    gr.Examples(
        examples=[
            ["../input/PNG/Original/1.png"],
            ["../input/PNG/Original/50.png"],
            ["../input/PNG/Original/100.png"],
        ],
        inputs=input_image,
        label="Örnek kolonoskopi görüntüleri"
    )
    
    gr.Markdown("""
    ---
    
    **Model:** UNet++ | **Framework:** PyTorch | **Veri Seti:** CVC-Clinic Database
    
    ⚠️ *Bu araç sadece eğitim amaçlıdır. Tıbbi teşhis için kullanılamaz.*
    """)
    
    # Buton bağlantısı
    analyze_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, status_text]
    )


if __name__ == "__main__":
    print("🚀 Gradio arayüzü başlatılıyor...")
    print("📍 Tarayıcıda aç: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
