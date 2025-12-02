from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document


pipeline_options = PdfPipelineOptions(
    generate_picture_images= True,
    images_scale=1.5,
    do_table_structure=  True
)

def extract_text_images_tables(file_path):
    doc_list = []
    doc_converter = DocumentConverter(format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    result = doc_converter.convert(file_path)
    doc = result.document
    text = doc.export_to_markdown() #str

    
    # TODO: Implement image handling later
    # image_embedings = []
    # for i, pic in enumerate(doc.pictures):
    #     pil_image = pic.get_image(doc)
    #
    #     image_metadata = {
    #         'source': file_path,
    #         'type': 'image',
    #         'image_index': i,
    #         # 'page_number': pic.page_num,
    #         # 'coordinates': pic.box.tolist(),
    #
    #     }
    #     embedding = get_image_embeddings(pil_image)
    #
    #     image_embedings.append({
    #         'embedding': embedding,
    #         'metadata': image_metadata
    #     })

    doc = Document(page_content = text,
                   metadata = {'source': file_path})
    doc_list.append(doc)
    return doc_list,None

