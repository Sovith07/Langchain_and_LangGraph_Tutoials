from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough

os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5
    )
)
model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain1=RunnableSequence(prompt1,model,parser)

chain2=RunnableParallel({'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

finalchain=RunnableSequence(chain1,chain2)

print(finalchain.invoke({'topic':'cricket'}))