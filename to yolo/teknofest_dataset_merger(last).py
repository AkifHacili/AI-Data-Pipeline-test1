"""
TEKNOFEST 2026 Havacılıkta Yapay Zeka
Veri Seti Birleştirme ve YOLO Dönüşüm Betiği
=============================================
Gereksinimler:
    pip install fiftyone torch torchvision

Desteklenen veri setleri:
    - VisDrone (YOLOv5 formatı)
    - UAVDT   (MOT/CSV formatı — önceden YOLO'ya çevrilmeli)
    - DroneVehicle (YOLO formatı)
    - HIT-UAV (YOLO formatı)
    - Landing Pad (YOLO formatı)
    - COCO alt küme (COCO JSON formatı)
"""

import os
import random
import fiftyone as fo
import fiftyone.utils.random as four

# ──────────────────────────────────────────────
# YAPILANDIRMA — Yolları kendi sisteminize göre güncelleyin
# ──────────────────────────────────────────────

DATASET_NAME = "teknofest_final"   # FiftyOne içindeki dataset adı
EXPORT_DIR   = "teknofest_yolo_dataset"  # Dışa aktarma klasörü
VAL_SPLIT    = 0.2                 # Doğrulama seti oranı (%20)
RANDOM_SEED  = 42

# Her veri seti için: name, path, type, (opsiyonel) label_field
DATASETS = [
    {
        "name": "visdrone",
        # VisDrone YOLOv5 formatında gelir (images/ + labels/ klasörleri)
        "path": "C:/Veriler/VisDrone",
        "type": fo.types.YOLOv5Dataset,
    },
    {
        "name": "uavdt",
        # UAVDT'yi önce visdrone2yolo.py veya benzeri bir betikle dönüştürün
        "path": "C:/Veriler/UAVDT_YOLO",
        "type": fo.types.YOLOv5Dataset,
    },
    {
        "name": "dronevehicle",
        "path": "C:/Veriler/DroneVehicle",
        "type": fo.types.YOLOv5Dataset,
    },
    {
        "name": "hit_uav",
        "path": "C:/Veriler/HIT-UAV",
        "type": fo.types.YOLOv5Dataset,
    },
    {
        "name": "landing_pad",
        "path": "C:/Veriler/LandingPad",
        "type": fo.types.YOLOv5Dataset,
    },
    # COCO formatı kullanan veri setleri için örnek:
    # {
    #     "name": "coco_subset",
    #     "path": "C:/Veriler/COCO",
    #     "type": fo.types.COCODetectionDataset,
    #     "labels_path": "annotations/instances_train2017.json",
    # },
]

# ──────────────────────────────────────────────
# SINIF EŞLEŞTİRME (MAPPING)
# Şartname: 0=Taşıt, 1=İnsan, 2=UAP, 3=UAİ
# ──────────────────────────────────────────────

# Scooter ayrı tutulur — iniş kuralı uygulandıktan sonra etiket atanır
CLASS_MAP = {
    # Taşıt Grubu (ID 0)
    "car": "tasit", "van": "tasit", "truck": "tasit", "bus": "tasit",
    "train": "tasit", "locomotive": "tasit", "wagon": "tasit",
    "ship": "tasit", "boat": "tasit", "aircraft": "tasit", "airplane": "tasit",
    "bicycle": "tasit", "motorcycle": "tasit", "motorbike": "tasit",
    "atv": "tasit", "tractor": "tasit", "tram": "tasit",
    "vehicle": "tasit", "motor": "tasit",

    # İnsan Grubu (ID 1)
    "person": "insan", "pedestrian": "insan", "human": "insan", "people": "insan",

    # Scooter — kurallar uygulanacak, sonra tasit veya insan olacak
    "scooter": "scooter",

    # İniş Bölgeleri (ID 2 ve 3) — veri setinde varsa
    "uap": "uap", "parking_area": "uap", "parking": "uap",
    "uai": "uai", "landing_area": "uai", "helipad": "uai",
    "landing_pad": "uai",
}

FINAL_CLASSES = ["tasit", "insan", "uap", "uai"]  # Sıra = YOLO class ID'si


# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────

def compute_iou(box1, box2):
    """
    FiftyOne bounding_box formatı: [x_top_left, y_top_left, width, height]
    Tüm değerler 0-1 arasında normalize edilmiş.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 1e-6 else 0.0


def apply_teknofest_rules(detections):
    """
    Şartname kurallarını uygular:

    Bisiklet / Motosiklet + Sürücü:
        → Araç etiketi (tasit) kalır, sürücü (insan) silinir.
          (Şartname: sürücüyle birlikte tek nesne olarak etiketle)

    Scooter + Sürücü (çakışma var):
        → İnsan etiketi kalır, scooter silinir.

    Scooter (sürücüsüz / çakışma yok):
        → Taşıt (tasit) olarak etiketlenir.
    """
    to_remove = set()

    # Sadece i < j çiftlerini kontrol et — çift sayımı önler
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_i = detections[i]
            det_j = detections[j]

            iou = compute_iou(det_i.bounding_box, det_j.bounding_box)
            if iou <= 0.3:
                continue

            label_i = det_i.label
            label_j = det_j.label

            # Kural 1: Bisiklet/Motosiklet + İnsan → İnsanı sil
            if label_i == "tasit" and label_j == "insan":
                to_remove.add(j)

            elif label_i == "insan" and label_j == "tasit":
                to_remove.add(i)

            # Kural 2: Scooter + İnsan → Scooter'ı sil, İnsan kalsın
            elif label_i == "scooter" and label_j == "insan":
                to_remove.add(i)

            elif label_i == "insan" and label_j == "scooter":
                to_remove.add(j)

    # Scooter'ları tasit'e dönüştür (sürücüsüzler hayatta kaldı)
    result = []
    for k, det in enumerate(detections):
        if k in to_remove:
            continue
        if det.label == "scooter":
            det.label = "tasit"
        result.append(det)

    return result


def load_dataset(cfg):
    """Tek bir veri setini yükler; hata durumunda None döner."""
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"  [UYARI] Yol bulunamadı, atlanıyor: {path}")
        return None

    kwargs = {
        "dataset_dir": path,
        "dataset_type": cfg["type"],
        "name": cfg["name"] + "_tmp",
        "overwrite": True,
    }
    # COCO formatı için labels_path ek parametresi
    if "labels_path" in cfg:
        kwargs["labels_path"] = cfg["labels_path"]

    try:
        ds = fo.Dataset.from_dir(**kwargs)
        print(f"  [OK] {cfg['name']}: {len(ds)} örnek yüklendi.")
        return ds
    except Exception as e:
        print(f"  [HATA] {cfg['name']} yüklenemedi: {e}")
        return None


# ──────────────────────────────────────────────
# ANA İŞLEM
# ──────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    # 1. Mevcut dataset'i temizle — tekrar çalıştırmada çakışma olmaz
    if fo.dataset_exists(DATASET_NAME):
        print(f"Mevcut '{DATASET_NAME}' siliniyor...")
        fo.delete_dataset(DATASET_NAME)

    master_ds = fo.Dataset(DATASET_NAME)
    master_ds.persistent = True  # FiftyOne App'te görünür kalır

    # 2. Tüm veri setlerini yükle ve birleştir
    print("\n── Veri Setleri Yükleniyor ──")
    for cfg in DATASETS:
        print(f"→ {cfg['name']}...")
        sub_ds = load_dataset(cfg)
        if sub_ds is None:
            continue
        master_ds.add_samples(sub_ds)

    print(f"\nToplam yüklenen örnek: {len(master_ds)}")

    # 3. Sınıf eşleştirme + Şartname kuralları
    print("\n── Sınıf Eşleştirme ve Şartname Kuralları Uygulanıyor ──")
    skipped = 0
    for sample in master_ds.iter_samples(progress=True):
        if not sample.detections:
            skipped += 1
            continue

        dets = sample.detections.detections

        # Adım A: Temel etiket eşleştirme
        mapped = []
        for det in dets:
            new_label = CLASS_MAP.get(det.label.lower(), None)
            if new_label is None:
                continue  # Şartname dışı sınıfı at
            det.label = new_label
            mapped.append(det)

        # Adım B: Şartname kuralları (bisiklet/scooter mantığı)
        final_dets = apply_teknofest_rules(mapped)

        sample.detections.detections = final_dets
        sample.save()

    print(f"  Etiket yok / atlandı: {skipped} örnek")

    # 4. Train / Val bölümleme (%80 / %20)
    print("\n── Train/Val Bölümleme ──")
    all_ids = master_ds.values("id")
    random.shuffle(all_ids)
    val_count = int(len(all_ids) * VAL_SPLIT)
    val_ids   = set(all_ids[:val_count])

    for sample in master_ds.iter_samples(progress=True):
        sample.tags = ["val"] if sample.id in val_ids else ["train"]
        sample.save()

    train_count = len(master_ds) - val_count
    print(f"  Train: {train_count} | Val: {val_count}")

    # 5. YOLO formatında dışa aktarma (train ve val ayrı ayrı)
    print(f"\n── Dışa Aktarma: '{EXPORT_DIR}' ──")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    for split in ["train", "val"]:
        split_view = master_ds.match_tags(split)
        split_dir  = os.path.join(EXPORT_DIR, split)

        print(f"  {split}: {len(split_view)} örnek → {split_dir}")
        split_view.export(
            export_dir=split_dir,
            dataset_type=fo.types.YOLOv8Dataset,
            label_field="detections",
            classes=FINAL_CLASSES,
        )

    # 6. data.yaml oluştur (YOLO eğitimi için gerekli)
    yaml_path = os.path.join(EXPORT_DIR, "data.yaml")
    yaml_content = f"""# TEKNOFEST 2026 - Otomatik Oluşturuldu
path: {os.path.abspath(EXPORT_DIR)}
train: train/images
val:   val/images

nc: {len(FINAL_CLASSES)}
names: {FINAL_CLASSES}
# 0: tasit  → Taşıt (araç, gemi, bisiklet vb.)
# 1: insan  → İnsan / Yaya
# 2: uap    → UAP (iniş bölgesi — uygun değil)
# 3: uai    → UAİ (iniş bölgesi — uygun)
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n✅ İşlem tamamlandı!")
    print(f"   Dataset klasörü : {os.path.abspath(EXPORT_DIR)}/")
    print(f"   data.yaml       : {yaml_path}")
    print(f"   Train           : {train_count} görüntü")
    print(f"   Val             : {val_count} görüntü")
    print(f"\nYOLO eğitimi için:")
    print(f"   yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
