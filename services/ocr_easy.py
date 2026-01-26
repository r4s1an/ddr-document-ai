from __future__ import annotations
from typing import Dict, Any, List, Iterable
from pathlib import Path
import easyocr

class EasyOCRService:
    def __init__(self, lang_list=None, gpu: bool = False):
        self.lang_list = lang_list or ["en"]
        self.reader = easyocr.Reader(self.lang_list, gpu=gpu)

    def read_image(self, img_path: str) -> Dict[str, Any]:
        p = Path(img_path)
        if not p.exists():
            return {"text": "", "conf": 0.0, "error": f"NOT_FOUND: {img_path}"}

        res = self.reader.readtext(str(p), detail=1, paragraph=True)
        if not res:
            return {"text": "", "conf": 0.0}

        texts: List[str] = []
        confs: List[float] = []

        for r in res:
             
            if isinstance(r, (list, tuple)):
                if len(r) >= 2:
                    texts.append(str(r[1]).strip())
                if len(r) >= 3:
                    try:
                        confs.append(float(r[2]))
                    except Exception:
                        pass
            else:
                texts.append(str(r).strip())

        text = " ".join([t for t in texts if t]).strip()
        conf = (sum(confs) / len(confs)) if confs else 0.0
        return {"text": text, "conf": conf}

    def read_items(
        self,
        items: List[Dict[str, Any]],
        labels: Iterable[str],
    ) -> Dict[int, Dict[str, Any]]:
        labels = set(labels)
        out: Dict[int, Dict[str, Any]] = {}
        for it in items:
            if it.get("label") not in labels:
                continue
            crop_path = it.get("crop_path", "")
            out[it["idx"]] = self.read_image(crop_path)
        return out