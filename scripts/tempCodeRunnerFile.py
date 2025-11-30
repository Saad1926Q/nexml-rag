 retriever_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)


    prompt = PromptTemplate(
        template=TALK2PROPOSAL_PROMPT
        input_variables = ['context', 'question']
    )
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    answer = llm.invoke(final_prompt)
    print(answer.content)