# from  utils.utils import chunk_text_and_generate_embeddings
# from scripts.doc_extractor import extract_text_images_tables
# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores.utils import filter_complex_metadata
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # use when proposals are in pdf

# proposal_chunked = []
# image_chunk = []
# for proposals in os.listdir('../documents/proposals'):
#     file_path = os.path.join('../documents/proposals', proposals)
    
#     text, img_emb = extract_text_images_tables(file_path=file_path)
#     text_emb, e= chunk_text_and_generate_embeddings(text)
    
    
#     proposal_chunked.extend(chunk)

# # use when proposals are in a csv``

# # proposal_path = 'documents/naccer_proposals_100_cleaned.csv'
# # proposal_load = file_loader(file_path=proposal_path)
# # proposal_chunked, e = chunk_data(proposal_load)


# proposal_chunked = filter_complex_metadata(proposal_chunked)
# vectorstore_proposals = Chroma.from_documents(
#     proposal_chunked,
#     embedding=e,
#     collection_name='proposal_collection',
#     persist_directory=persist_directory
    
# )
    
# print("Database Created ")
    
    
    
    