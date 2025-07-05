from datetime import datetime

import wikipedia
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="search the web for information",
)

api_wrapper = WikipediaAPIWrapper(
    wiki_client=wikipedia, top_k_results=1, doc_content_chars_max=100
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# CUSTOM TOOL


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


# to create a custom tool, just create a function and pass it as an argument to the Tool function.
save_tool = Tool(
    name="save_research", func=save_to_txt, description="save research in txt format."
)
