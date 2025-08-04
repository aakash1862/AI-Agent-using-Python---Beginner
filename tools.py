from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, filname: str = "research_paper.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}"

    with open(filname, "w") as file:
        file.write(formatted_data)

    return f"Data successfully saved to {filname}"

save_tool = Tool(
    name="save_to_txt",
    func=save_to_txt,
    description="Saves the research paper data to a text file with a timestamp."
)

search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Useful for when you need to answer questions about current events. Input should be a search query."
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

