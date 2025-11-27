from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os
from pathlib import Path
from PIL import Image # Pillow is required to handle the image objects

# --- Configuration ---
input_file = Path("documents/PID603 (1).pdf")
text_output_file = "output_text.txt"
images_output_folder = "extracted_images"

# 1. Define Pipeline Options to force image data generation
pipeline_options = PdfPipelineOptions()
# THIS IS THE CRUCIAL LINE to generate the actual image pixel data
pipeline_options.generate_picture_images = True 
# Optional: Increase scale for better resolution (e.g., 2.0 = 2x resolution)
pipeline_options.images_scale = 1.5 

# 2. Initialize the Converter with the Options
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = doc_converter.convert(input_file)
doc = result.document

# --- Text Export (Your original code, kept for completeness) ---

plain_text = doc.export_to_markdown()

with open(text_output_file, "w", encoding="utf-8") as f:
    f.write(plain_text)

# --- Image Export (The corrected part) ---

if not os.path.exists(images_output_folder):
    os.makedirs(images_output_folder)

picture_count = 0

# Save all images
for i, pic in enumerate(doc.pictures):
    image_path = None 
    try:
        # 3. Use get_image(doc) to retrieve the Pillow Image object
        pil_image = pic.get_image(doc) 
        
        # Determine format for saving. Pillow saves based on the extension.
        ext = 'png' 
        image_path = os.path.join(images_output_folder, f"image_{i+1}.{ext}")
        
        if pil_image:
            # 4. Use PIL's save method to write the file
            pil_image.save(image_path)
            picture_count += 1
            print(f"Successfully saved image {i+1} to {image_path}")
        else:
             print(f"Warning: Picture {i+1} found, but image data was still not available.")

    except Exception as e:
        print(f"Error saving image {i+1} to {image_path}: {e}")

print(f"{picture_count} images saved to {images_output_folder}")