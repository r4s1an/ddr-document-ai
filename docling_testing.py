from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem, PictureItem

# 1. Setup
path = Path(r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf")
pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True 
pipeline_options.do_ocr = True 

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# 2. Convert
result = converter.convert(path)
doc = result.document

print("\n--- DETECTED ITEMS ANALYSIS ---")

for item, _ in doc.iterate_items():
    # TYPE 1: Tables
    if isinstance(item, TableItem):
        df = item.export_to_dataframe(doc)
        print(f"\n[TABLE] found with {len(df.columns)} columns.")
        print(f"Columns: {list(df.columns)}")

    # TYPE 2: Pictures / Figures (This is where your top section is!)
    elif isinstance(item, PictureItem):
        print(f"\n[PICTURE/FIGURE] detected.")
        # Pictures don't have .text, but they might have annotations
        if hasattr(item, 'annotations') and item.annotations:
            print(f"   (Contains {len(item.annotations)} internal OCR snippets)")
        else:
            print("   (Visual block with no extracted text summary)")

    # TYPE 3: Everything Else (Paragraphs, Headers, etc.)
    else:
        # Use getattr to safely get text if it exists
        text = getattr(item, 'text', "[No Text Content]")
        label = getattr(item, 'label', "Unknown")
        print(f"[{label}]: {text[:100]}")


# import os
# import json
# from docling.document_converter import DocumentConverter
# from docling_core.types.doc.document import TableItem

# pdf_path = r"C:\Users\Yoked\Desktop\EIgroup 2nd try\PDF_version_1000\15_9_F_14_2008_06_14.pdf"

# # --- Initialize ---
# converter = DocumentConverter()

# # --- Run conversion (GPU auto-detected) ---
# print(f"Processing {os.path.basename(pdf_path)}...")
# result = converter.convert(pdf_path)
# doc = result.document
# print(f"Successfully converted: {doc.name}")

# # --- Iterate tables ---
# for item, _ in doc.iterate_items():
#     if isinstance(item, TableItem):
#         print("\nTable detected:")
#         df = item.export_to_dataframe(doc=doc)  # Pass doc to remove warning
#         print(df.head())

# # --- Export to JSON ---
# json_data = doc.export_to_dict()
# with open("output_results.json", "w", encoding="utf-8") as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=4)

# print("\nDocument exported to output_results.json")