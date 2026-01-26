from doclayout_yolo import YOLOv10
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
import torch
from pathlib import Path

CUSTOM_WEIGHTS = r"C:\Users\Yoked\Desktop\DDR Processor\models\custom\best.pt"
DATA_YAML = r"C:\Users\Yoked\Desktop\DDR Processor\fine-tuning\ddr_layout.yaml"
TEST_IMAGES_DIR = r"C:\Users\Yoked\Desktop\DDR Processor\dataset\images\test"

torch.serialization.add_safe_globals([YOLOv10DetectionModel])

def main():
    print("=" * 80)
    print("YOLO Model Testing Script")
    print("=" * 80)
    
    print(f"\nLoading model from: {CUSTOM_WEIGHTS}")
    model = YOLOv10(CUSTOM_WEIGHTS)
    
    print("\n" + "=" * 80)
    print("Running validation on test dataset...")
    print("=" * 80)
    
    metrics = model.val(
        data=DATA_YAML,
        split='test',
        imgsz=1024,
        batch=4,
        device=0,
        save_json=True,
        save_hybrid=True,
        conf=0.25,
        iou=0.6,
        max_det=300,
        plots=True,
    )
    
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    
    print(f"\nmAP@0.5:0.95 (all classes): {metrics.box.map:.4f}")
    print(f"mAP@0.5 (all classes):      {metrics.box.map50:.4f}")
    print(f"mAP@0.75 (all classes):     {metrics.box.map75:.4f}")
    
    print(f"\nPrecision (all classes):    {metrics.box.mp:.4f}")
    print(f"Recall (all classes):       {metrics.box.mr:.4f}")
    
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    
    if hasattr(metrics.box, 'maps'):
        class_names = model.names
        for i, (name, map_val) in enumerate(zip(class_names.values(), metrics.box.maps)):
            print(f"Class {i} ({name}): mAP@0.5:0.95 = {map_val:.4f}")
    
    print("\n" + "=" * 80)
    print("Running inference on test images...")
    print("=" * 80)
    
    test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg")) + \
                  list(Path(TEST_IMAGES_DIR).glob("*.png"))
    
    if test_images:
        print(f"\nFound {len(test_images)} test images")
        print("Running predictions...")
        
        results = model.predict(
            source=TEST_IMAGES_DIR,
            imgsz=1024,
            conf=0.25,
            iou=0.6,
            device=0,
            save=True,
            save_txt=True,
            save_conf=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        
        print(f"\nPredictions saved to: runs/detect/predict")
        
        print("\n" + "-" * 80)
        print("SAMPLE DETECTIONS (first 5 images)")
        print("-" * 80)
        for i, result in enumerate(results[:5]):
            img_name = Path(result.path).name
            num_detections = len(result.boxes)
            print(f"\n{img_name}: {num_detections} detections")
            
            if num_detections > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    print(f"  - {cls_name}: {conf:.3f}")
    else:
        print(f"\nNo test images found in: {TEST_IMAGES_DIR}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nModel: {CUSTOM_WEIGHTS}")
    print(f"Test images: {len(test_images) if test_images else 0}")
    print(f"\nOverall Performance:")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")
    
    print("\nValidation plots saved to: runs/detect/val")
    print("Prediction images saved to: runs/detect/predict")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()