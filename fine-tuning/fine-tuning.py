from doclayout_yolo import YOLOv10
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
import torch

MODEL_WEIGHTS = r"C:\Users\Yoked\Desktop\DDR Processor\models\models--juliozhao--DocLayout-YOLO-DocStructBench\snapshots\8c3299a30b8ff29a1503c4431b035b93220f7b11\doclayout_yolo_docstructbench_imgsz1024.pt"
DATA_YAML = "ddr_layout.yaml"
torch.serialization.add_safe_globals([YOLOv10DetectionModel])

def main():
    model = YOLOv10(MODEL_WEIGHTS)

    model.train(
        data=DATA_YAML,
        name="DDR_Phase1_Frozen",
        imgsz=1024,
        epochs=50,
        batch=2,
        freeze=10,
        device=0,
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=1e-4,
        patience=15,
        pretrained=True,
        amp=True,
        workers=4,
    )

if __name__ == "__main__":
    main()