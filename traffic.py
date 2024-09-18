import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time
from PIL import Image
import numpy as np

# COCO veri kümesindeki kategorilerin isimleri
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench'
]

# Cihazı belirle (GPU varsa GPU, yoksa CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Modelin ağırlıklarını yükle ve modeli oluştur
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()  # Modeli değerlendirme moduna al

while True:
    # Kullanıcıdan fotoğraf yolunu iste
    image_path = input("Lütfen işlemek istediğiniz fotoğrafın yolunu girin (çıkmak için 'q' yazın): ")
    if image_path.lower() == 'q':
        break
    
    # Görüntüyü yükle ve RGB'den BGR'ye dönüştür
    image = Image.open(image_path)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Görüntüyü tensöre dönüştür ve cihaza yükle
    image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Nesne algılama işlemini başlat ve süreyi hesapla
    start_time = time.time()
    outputs = model(image_tensor)
    elapsed_time = time.time() - start_time
    print(f"Tespit süresi: {elapsed_time:.2f} saniye")

    # Algılanan kutuları, etiketleri ve skorları alın
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()
    
    person_count = 0
    car_count = 0
    
    # Algılanan nesnelerin üzerinde döngü
    for box, label, score in zip(boxes, labels, scores):
        # Skoru 0.7'den büyük ve etiket geçerli ise
        if score > 0.7 and label < len(COCO_INSTANCE_CATEGORY_NAMES):
            x1, y1, x2, y2 = [int(v) for v in box]
            label_text = COCO_INSTANCE_CATEGORY_NAMES[label]
            
            # Etiket kişiyse veya arabaysa sayacı artır
            if label_text == 'person':
                person_count += 1
            elif label_text == 'car':
                car_count += 1
            
            # Görüntüye dikdörtgen ve etiket metni ekle
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2.putText(image, f"{label_text}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    
    # Trafik durumu kontrolü ve bilgilendirme
    if car_count >= 10:
        print("Trafik vardır.")
    else:
        print("Trafik yoktur.")
    
    # İnsan kalabalığı durumu kontrolü ve bilgilendirme
    if person_count >= 10:
        print("İnsan kalabalığı vardır.")
    else:
        print("İnsan kalabalığı yoktur.")
    
    # Algılama sonuçlarını göster
    cv2.imshow('Nesne Algılama', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
