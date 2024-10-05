from langchain_core.prompts import PromptTemplate

prompt_qn = '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:'''

prompt = PromptTemplate.from_template(prompt_qn)


