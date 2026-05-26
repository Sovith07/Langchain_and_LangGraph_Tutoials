from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda

os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5
    )
)
model = ChatHuggingFace(llm=llm)

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain=RunnableSequence(prompt,llm,parser)

chain2=RunnableParallel({'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)})

finalchain=RunnableSequence(chain,chain2)

result = finalchain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
