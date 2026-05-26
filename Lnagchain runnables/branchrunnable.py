from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableBranch

os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5
    )
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain1=RunnableSequence(prompt1,llm,parser)

chain2=RunnableBranch((lambda x: len(x.split())>500,RunnableSequence(prompt2,llm,parser)),
                      RunnablePassthrough()
                      )

finalchain = RunnableSequence(chain1,chain2)

print(finalchain.invoke({'topic':'Russia vs Ukraine'}))