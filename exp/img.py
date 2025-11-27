# --- Additional Imports ---
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32" 
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

CHROMA_DIR = "chroma_db_images"
client = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION_NAME = "pdf_images_embeddings"

try:
    client.delete_collection(name=COLLECTION_NAME)
except:
    pass 
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")

# --- Define the Embedding Function ---
def get_image_embedding(pil_image):
    """Generates a numerical vector embedding for a PIL Image."""
    # Preprocess the image according to CLIP's requirements
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    
    # Generate the embedding vector
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
    
    # Convert the PyTorch tensor to a Python list of floats
    # and normalize it (important for good similarity search)
    embedding = image_features.cpu().numpy().tolist()[0]
    return embedding

# -------------------------------------------------------------
# --- Image Extraction and Embedding Loop (Modified Section) ---
# -------------------------------------------------------------

if not os.path.exists(images_output_folder):
    os.makedirs(images_output_folder)

picture_count = 0
image_embeddings = []
image_metadata = []
image_ids = []

# Save all images and generate embeddings
for i, pic in enumerate(doc.pictures):
    image_id = f"image_{i+1}-{input_file.stem}" # Unique ID
    
    try:
        pil_image = pic.get_image(doc) 
        
        if pil_image:
            # 1. Generate Embedding
            embedding_vector = get_image_embedding(pil_image)
            image_embeddings.append(embedding_vector)
            image_ids.append(image_id)
            
            # 2. Prepare Metadata
            # Store useful information with the vector (e.g., source file, bounding box)
            metadata = {
                "source_file": input_file.name,
                "page_number": pic.bounding_box.page,
                "bbox": str(pic.bounding_box.box), # Coordinates as a string
                "image_path": os.path.join(images_output_folder, f"image_{i+1}.png")
            }
            image_metadata.append(metadata)
            
            # 3. Save Image to Disk (optional, but good practice)
            # You can save to disk OR keep it as a PIL object and delete later
            pil_image.save(metadata["image_path"], format='PNG')
            
            picture_count += 1
            print(f"Image {i+1}: Embedded and saved.")
        else:
             print(f"Warning: Picture {i+1} found, but data was not available.")

    except Exception as e:
        print(f"Error processing image {i+1}: {e}")

# --- 3. Store Embeddings in ChromaDB ---
if image_embeddings:
    print(f"\nStoring {len(image_embeddings)} embeddings in ChromaDB...")
    collection.add(
        embeddings=image_embeddings,
        metadatas=image_metadata,
        documents=[f"Image from page {m['page_number']} of {m['source_file']}" for m in image_metadata],
        ids=image_ids
    )
    print("Insertion complete.")

print(f"\n{picture_count} images successfully processed and stored in {CHROMA_DIR}.")