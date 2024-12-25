import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI  # Updated import
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()

# Replace with your actual ScrapingBee API key
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search
def search(query):
    print(f"Searching for: {query}")  # Debugging line
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code != 200:
            print(f"Error: Received {response.status_code} from the API.")
            return None
        return response.json()  # Return the parsed JSON response
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return None


# 2. Tool for scraping using ScrapingBee API
def scrape_website(objective: str, url: str):
    print(f"Scraping website: {url}")  # Debugging line
    scrapingbee_url = "https://app.scrapingbee.com/api/v1"
    params = {
        'api_key': SCRAPINGBEE_API_KEY,
        'url': url,
        'render': 'false',  # Change to 'true' if you need to render JavaScript
    }

    try:
        response = requests.get(scrapingbee_url, params=params)
        
        if response.status_code == 200:
            content = response.text
            if len(content) > 10000:
                output = summary(objective, content)
                return output
            else:
                return content
        else:
            print(f"Error: Received {response.status_code} from the API.")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


# 3. Summarization function
def summary(objective, content):
    print(f"Summarizing content for objective: {objective}")  # Debugging line
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = f"""
    Write a summary of the following text for {objective}:
    "{content}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    return output


# 4. Define input model for scraping
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The URL of the website to be scraped")


# 5. Create a custom ScrapeWebsiteTool
class ScrapeWebsiteTool(BaseTool):
    name: str = "scrape_website"
    description: str = "Use this tool to scrape a website for data based on a user's objective. DO NOT make up any URL."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("This method is not implemented for async.")


# 6. Create Langchain agent with search and scrape tools
tools = [
    Tool(
        name="Search",
        func=search,
        description="Use this tool to search for relevant information on the web."
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective.
            2/ If there are URL links to relevant articles, you will scrape them to gather more information.
            3/ After scraping and searching, you should think "Is there anything new I should search or scrape based on the data I collected?" If the answer is yes, continue. But do this no more than three times.
            4/ You should not make things up. Only write facts and data that you have gathered.
            5/ In the final output, you should include all reference data and links to back up your research."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# 7. Set up FastAPI for the web API
app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/")
def researchAgent(query: Query):
    try:
        print(f"Received query: {query.query}")  # Debugging line
        content = agent.invoke({"input": query.query})

        print(f"Agent Response: {content}")  # Debugging line
        
        if isinstance(content, dict) and 'output' in content:
            actual_content = content['output']
            return {"result": actual_content}
        else:
            return {"error": f"Unexpected response structure: {content}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
