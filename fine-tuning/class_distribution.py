import os
from collections import Counter

def analyze_class_distribution(labels_dir):
    class_counts = Counter()
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

# Run this on your train/val/test splits
train_dist = analyze_class_distribution('dataset/labels/train')
print(f"Class distribution: {train_dist}")

import os
from pathlib import Path
from collections import defaultdict
import numpy as np

CLASSES = [
    "table",           # 0
    "figure",          # 1
    "plain_text",      # 2
    "section_header",  # 3
    "wellbore_field",  # 4
    "period_field",    # 5
]

def yolo_to_xyxy(yolo_box):
    """Convert YOLO format (center_x, center_y, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = yolo_box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes in xyxy format"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def check_label_overlaps(labels_dir, iou_threshold=0.3):
    """
    Check for overlapping bounding boxes in YOLO label files
    
    Args:
        labels_dir: Path to directory containing .txt label files
        iou_threshold: IoU threshold to consider boxes as overlapping
    
    Returns:
        Dictionary with overlap statistics
    """
    labels_dir = Path(labels_dir)
    
    total_files = 0
    files_with_overlaps = 0
    total_overlaps = 0
    overlap_details = []
    class_pair_overlaps = defaultdict(int)
    
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"Analyzing {len(label_files)} label files...")
    print("=" * 80)
    
    for label_file in label_files:
        total_files += 1
        
        # Read all boxes from the file
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    boxes.append((class_id, coords))
        
        if len(boxes) < 2:
            continue
        
        # Check all pairs of boxes for overlap
        file_overlaps = []
        for i in range(len(boxes)):
            class_i, box_i = boxes[i]
            box_i_xyxy = yolo_to_xyxy(box_i)
            
            for j in range(i + 1, len(boxes)):
                class_j, box_j = boxes[j]
                box_j_xyxy = yolo_to_xyxy(box_j)
                
                iou = calculate_iou(box_i_xyxy, box_j_xyxy)
                
                if iou > iou_threshold:
                    total_overlaps += 1
                    
                    class_pair = tuple(sorted([class_i, class_j]))
                    class_pair_overlaps[class_pair] += 1
                    
                    file_overlaps.append({
                        'box1_class': class_i,
                        'box2_class': class_j,
                        'iou': iou,
                        'box1_coords': box_i,
                        'box2_coords': box_j
                    })
        
        if file_overlaps:
            files_with_overlaps += 1
            overlap_details.append({
                'file': label_file.name,
                'overlaps': file_overlaps
            })
    
    # Print summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total label files analyzed: {total_files}")
    print(f"Files with overlaps: {files_with_overlaps} ({files_with_overlaps/total_files*100:.1f}%)")
    print(f"Total overlapping pairs: {total_overlaps}")
    print(f"Average overlaps per file: {total_overlaps/total_files:.2f}")
    
    # Print class pair statistics
    if class_pair_overlaps:
        print(f"\n🔀 OVERLAPS BY CLASS PAIRS (IoU > {iou_threshold})")
        print("=" * 80)
        sorted_pairs = sorted(class_pair_overlaps.items(), key=lambda x: x[1], reverse=True)
        for (class1, class2), count in sorted_pairs:
            print(f"{CLASSES[class1]:20s} ↔ {CLASSES[class2]:20s}: {count:4d} overlaps")
    
    # Print top files with most overlaps
    if overlap_details:
        print(f"\n⚠️  TOP 10 FILES WITH MOST OVERLAPS")
        print("=" * 80)
        sorted_files = sorted(overlap_details, key=lambda x: len(x['overlaps']), reverse=True)
        for file_info in sorted_files[:10]:
            print(f"\n{file_info['file']}: {len(file_info['overlaps'])} overlaps")
            for i, overlap in enumerate(file_info['overlaps'][:5], 1):  # Show first 5
                print(f"  [{i}] {CLASSES[overlap['box1_class']]:15s} ↔ "
                      f"{CLASSES[overlap['box2_class']]:15s} | IoU: {overlap['iou']:.3f}")
            if len(file_info['overlaps']) > 5:
                print(f"  ... and {len(file_info['overlaps']) - 5} more")
    
    return {
        'total_files': total_files,
        'files_with_overlaps': files_with_overlaps,
        'total_overlaps': total_overlaps,
        'class_pair_overlaps': dict(class_pair_overlaps),
        'overlap_details': overlap_details
    }

def find_severe_overlaps(labels_dir, iou_threshold=0.7):
    """Find boxes with very high overlap (likely duplicates)"""
    labels_dir = Path(labels_dir)
    
    severe_overlaps = []
    
    for label_file in labels_dir.glob("*.txt"):
        boxes = []
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    boxes.append((class_id, coords, line_num))
        
        for i in range(len(boxes)):
            class_i, box_i, line_i = boxes[i]
            box_i_xyxy = yolo_to_xyxy(box_i)
            
            for j in range(i + 1, len(boxes)):
                class_j, box_j, line_j = boxes[j]
                box_j_xyxy = yolo_to_xyxy(box_j)
                
                iou = calculate_iou(box_i_xyxy, box_j_xyxy)
                
                if iou > iou_threshold:
                    severe_overlaps.append({
                        'file': label_file.name,
                        'line1': line_i,
                        'line2': line_j,
                        'class1': CLASSES[class_i],
                        'class2': CLASSES[class_j],
                        'iou': iou
                    })
    
    if severe_overlaps:
        print(f"\n🚨 SEVERE OVERLAPS (IoU > {iou_threshold}) - Likely Duplicates")
        print("=" * 80)
        print(f"Found {len(severe_overlaps)} severe overlaps\n")
        
        for overlap in severe_overlaps[:20]:  # Show first 20
            print(f"File: {overlap['file']}")
            print(f"  Lines {overlap['line1']} & {overlap['line2']}: "
                  f"{overlap['class1']} ↔ {overlap['class2']} | IoU: {overlap['iou']:.3f}")
    
    return severe_overlaps

# Main execution
if __name__ == "__main__":
    # Update this path to your train labels directory
    TRAIN_LABELS_DIR = "dataset/labels/train"
    
    # Check for overlaps with IoU > 0.3
    results = check_label_overlaps(TRAIN_LABELS_DIR, iou_threshold=0.3)
    
    # Find severe overlaps (likely duplicates)
    severe = find_severe_overlaps(TRAIN_LABELS_DIR, iou_threshold=0.7)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)