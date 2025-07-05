#!/home/yugz/.conda/envs/default/bin/python

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from tools import save_tool, search_tool, wiki_tool

llama3p1 = ChatOllama(model="llama3.1:latest")
llama3p2 = ChatOllama(model="llama3.2:3b")
deepseek = ChatOllama(model="deepseek-r1:latest")

# res = llama3.invoke("what color are bananas?")
# print(res.content)


# # RESPONSE TEMPLATE
class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(
    pydantic_object=ResponseModel
)  # stringify the ResponseModel schema to pass as string to LLM

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llama3p1, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("what would you like me to research? ")

raw_response = agent_executor.invoke({"query": query})
print(raw_response["output"])
