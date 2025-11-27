from  utils.utils import chunk_text_and_generate_embeddings
from langchain_community.vectorstores import Chroma 
from langchain_community.vectorstores.utils import filter_complex_metadata
from scripts.doc_extractor import extract_text_images_tables


file_path = 'documents/S&T-Guidelines-MoC.pdf'


text, img_emb = extract_text_images_tables(file_path=file_path)
text_emb, e= chunk_text_and_generate_embeddings(text)


persist_directory = "./ChromaDB"
text_emb = filter_complex_metadata(text_emb)
vectorstore = Chroma.from_documents(text_emb,
                                    embedding=e,
                                    collection_name="guidelines_collection",
                                    
                                    persist_directory=persist_directory)


count = vectorstore._collection.count()
print(f"Total items in collection: {count} (Text chunks)")
image_ids = []
image_embeddings_list = []
image_metadatas = []

for item in img_emb:

    img_metadata = item['metadata']
    unique_id = f"{img_metadata['source']}_img_{img_metadata['image_index']}"
    
    image_ids.append(unique_id)
    image_embeddings_list.append(item['embedding'])
    image_metadatas.append(img_metadata)



print(f"Adding {len(image_embeddings_list)} image embeddings to the Chroma collection...")
collection = vectorstore._collection

collection.add(
    ids=image_ids,
    embeddings=image_embeddings_list,
    metadatas=image_metadatas
)

print("Image embeddings successfully added to the Chroma collection.")

count = vectorstore._collection.count()
print(f"Total items in collection: {count} (Text chunks + Images)")















# use when proposals are in pdf

# proposal_chunked = []
# for proposals in os.listdir('../documents/proposals'):
#     file_path = os.path.join('../documents/proposals', proposals)
#     load_file = file_loader(proposals)
#     chunk,e = chunk_data(load_file)
#     proposal_chunked.extend(chunk)

# use when proposals are in a csv``

# proposal_path = 'documents/naccer_proposals_100_cleaned.csv'
# proposal_load = file_loader(file_path=proposal_path)
# proposal_chunked, e = chunk_data(proposal_load)


# proposal_chunked = filter_complex_metadata(proposal_chunked)
# vectorstore_proposals = Chroma.from_documents(
#     proposal_chunked,
#     embedding=e,
#     collection_name='proposal_collection',
#     persist_directory=persist_directory
    
# )
    
# print("Database Created ")
    
    
    
    