# The flow for this code is:
# 1. Load environment variables from .env file.
# 2. Define a Pydantic model for the research response.
# 3. Initialize three different LLMs (Together, OpenAI, Anthropic) with their respective API keys.
# 4. Create a prompt template for generating research paper responses.
# 5. Invoke each LLM with a sample query and print the responses.
# 6. Create an agent using the Together LLM and execute it with a sample query.


from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_anthropic import Anthropic
from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    title: str = Field(description="The title of the research paper")
    summary: str = Field(description="A brief summary of the research paper")
    sources: list[str] = Field(description="List of sources used in the research paper")
    tools_used: list[str] = Field(description="List of tools used in the research paper")
    abstract: str = Field(description="The abstract of the research paper")
    authors: list[str] = Field(description="List of authors of the research paper")
    publication_date: str = Field(description="Publication date of the research paper")

llm = Together(
            model="meta-llama/Llama-3-70b-chat-hf",  # Or any supported Together model
            temperature=0,
            max_tokens=250,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

llm2 = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=250,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

llm3 = ChatAnthropic(
            model="claude-3-5-sonnet-20240229",
            temperature=0,
            max_tokens=250,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a helpful research assistant that will help generate a research paper. 
         You will generate a title, summary, list of sources, list of tools used, abstract, list of authors,
         and publication date.
         Wrap the output in this format and provide no other text \n{format_instructions}
         """
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# response = llm.invoke("What is the capital of France?")
# print(f"Response from Together: {response}")

# response2 = llm2.invoke("What is the capital of France?")
# print(f"Response from OpenAI: {response2}")

# response3 = llm3.invoke("What is the capital of France?")
# print(f"Response from Anthropic: {response}")

chat_history = []
agent_scratchpad = ""

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm2,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you with? ")

raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history, "agent_scratchpad": agent_scratchpad})
# print(f"Raw response: {raw_response}")

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(f"Structured response: {structured_response}")
    print("*"*20)
    print(f"Topic: {structured_response.title}\n")
    print(f"Summary: {structured_response.summary}\n")
    print(f"Sources: {structured_response.sources}\n")
    print(f"Tools Used: {structured_response.tools_used}\n")
    print(f"Abstract: {structured_response.abstract}\n")
    print(f"Authors: {structured_response.authors}\n")
    print(f"Publication Date: {structured_response.publication_date}\n")
except Exception as e:
    print(f"Error parsing response: {e}\nRaw response: {raw_response}")
