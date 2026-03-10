import os
import json
import xml.etree.ElementTree as ET
import shutil
from PIL import Image

# ==============================================================================
# TEKNOFEST 2026 HAVACILIKTA YAPAY ZEKA - VERİ SETİ DÖNÜŞTÜRÜCÜ (YOLO FORMATI)
# ==============================================================================
# Bu betik, Pascal VOC (XML) ve COCO (JSON) formatındaki etiketleri 
# YOLO (.txt) formatına dönüştürür.
#
# SINIF EŞLEŞTİRME STANDARDI:
# 0: Taşıt (Otomobil, otobüs, kamyon, tren, deniz taşıtı, motosiklet, bisiklet)
# 1: İnsan (Ayakta veya oturan)
# 2: UAP (Uçan Araba Park Alanı)
# 3: UAİ (Uçan Ambulans İniş Alanı)
# ==============================================================================

class TeknofestConverter:
    def __init__(self, output_dir="yolo_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        # Sınıf sayaçları (Raporlama için)
        self.stats = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Klasörleri oluştur
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Girdi sınıf isimlerini standart ID'lere eşleyen sözlük
        # Bu listeyi kendi veri setinizdeki isimlere göre genişletebilirsiniz.
        self.class_map = {
            # ID 0: Taşıtlar
            'car': 0, 'automobile': 0, 'bus': 0, 'truck': 0, 'train': 0, 
            'locomotive': 0, 'wagon': 0, 'ship': 0, 'boat': 0, 
            'motorcycle': 0, 'bicycle': 0, 'motorbike': 0, 'bike': 0,
            
            # ID 1: İnsan
            'person': 1, 'human': 1, 'pedestrian': 1,
            
            # ID 2: UAP
            'uap': 2, 'parking_area': 2,
            
            # ID 3: UAİ
            'uai': 3, 'landing_area': 3,

            # Özel Durumlar
            'scooter_no_driver': 0, # Sürücüsüz scooter -> Taşıt
            'scooter_with_driver': 1 # Sürücülü scooter -> İnsan
        }

    def normalize(self, size, box):
        """Koordinatları 0-1 arasına normalize eder (YOLO formatı)."""
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x * dw, y * dh, w * dw, h * dh)

    def process_voc(self, xml_path, img_path):
        """Pascal VOC XML dosyasını işler."""
        if not os.path.exists(xml_path):
            return False
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        yolo_data = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.lower()
            
            # Sınıf eşleştirme
            cls_id = self.class_map.get(cls_name, -1)
            if cls_id == -1: continue # Tanımlanmayan sınıfları atla
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            bb = self.normalize((w, h), b)
            yolo_data.append(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}")
            self.stats[cls_id] += 1
            
        if yolo_data:
            self.save_result(img_path, yolo_data)
            return True
        return False

    def process_coco(self, json_path, img_folder):
        """COCO JSON dosyasını işler."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Kategori ID'lerini isimlere eşle
        categories = {cat['id']: cat['name'].lower() for cat in data['categories']}
        
        # Görselleri ID'ye göre indeksle
        images = {img['id']: img for img in data['images']}
        
        # Annotasyonları grupla
        annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations: annotations[img_id] = []
            annotations[img_id].append(ann)
            
        for img_id, anns in annotations.items():
            img_info = images[img_id]
            file_name = img_info['file_name']
            w, h = img_info['width'], img_info['height']
            img_path = os.path.join(img_folder, file_name)
            
            if not os.path.exists(img_path): continue
            
            yolo_data = []
            for ann in anns:
                cls_name = categories.get(ann['category_id'], "")
                cls_id = self.class_map.get(cls_name, -1)
                if cls_id == -1: continue
                
                # COCO format: [x_min, y_min, width, height]
                x, y, bw, bh = ann['bbox']
                # YOLO normalize
                x_center = (x + bw/2) / w
                y_center = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                
                yolo_data.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")
                self.stats[cls_id] += 1
                
            if yolo_data:
                self.save_result(img_path, yolo_data)
        
        print(f"COCO dönüşümü tamamlandı.")

    def save_result(self, img_path, yolo_data):
        """Görseli kopyalar ve etiket dosyasını oluşturur."""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Görseli kopyala
        shutil.copy(img_path, os.path.join(self.images_dir, os.path.basename(img_path)))
        
        # Etiketi yaz
        with open(os.path.join(self.labels_dir, base_name + ".txt"), 'w') as f:
            f.write('\n'.join(yolo_data))

    def report(self):
        """İşlem sonucunu raporlar."""
        print("\n" + "="*30)
        print("DÖNÜŞTÜRME RAPORU")
        print("="*30)
        names = {0: "Taşıt", 1: "İnsan", 2: "UAP", 3: "UAİ"}
        for cid, count in self.stats.items():
            print(f"{names[cid]} (ID {cid}): {count} adet nesne")
        print("="*30)
        print(f"Sonuçlar '{self.output_dir}' klasörüne kaydedildi.")

# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    converter = TeknofestConverter(output_dir="teknofest_dataset")
    
    # 1. Pascal VOC Örneği (Tek tek dosyalar için)
    # converter.process_voc("data/label1.xml", "data/image1.jpg")
    
    # 2. COCO Örneği (Toplu JSON için)
    # converter.process_coco("data/annotations.json", "data/images/")
    
    converter.report()
