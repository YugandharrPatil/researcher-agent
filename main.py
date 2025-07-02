#!/home/yugz/.conda/envs/python-3.12/bin/python

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

MODELS = {"llama":"llama3.2:3b", "deepseek": "deepseekk-r1:latest"}

llm = ChatOllama(model=MODELS["llama"])

res = llm.invoke("what color are bananas?")
print(res.content)

# RESPONSE TEMPLATE
class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResponseModel)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
You are a research assistant that will help generate a research paper. Answer the user query and use necessary tools. Wrap the output in this format and provide no other text
         """)
    ]
)