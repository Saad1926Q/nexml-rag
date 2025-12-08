# from langchain_community.retrievers import ArxivRetriever

# retriever = ArxivRetriever(
#     load_max_docs=20,
#     get_full_documents=True,
#     load_all_available_meta=True
# )

# docs = retriever.invoke("""Mineral Processing; Advanced; Process; Sustainable""")
# metadata = docs[0].metadata
# print(metadata['entry_id'])    


from semanticscholar import SemanticScholar

from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

query = """Mineral Processing; Advanced; Process; Sustainable"""
ss_tool = SemanticScholarQueryRun()
papers = ss_tool.run(query)

for paper in papers:
    print(paper.paperId)
    print(paper.url)