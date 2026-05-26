from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5
    )
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)
parser = StrOutputParser()
text=TextLoader('cricket.txt', encoding='utf-8')

docs = text.load()

print(docs)

print(type(docs))

print(len(docs))

print(docs[0])

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))