from dotenv import load_dotenv
#import os
from langchain.agents import create_agent
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from datetime import datetime
import gradio as gr

load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

#llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm = ChatOllama(model="qwen2.5:3b")
system_prompt = """
You are a helpful assistant.
Answer all user's queries
Use the get-date tool if the user is asking about today's date.
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
#user_query = input("Enter a query: ")
#response = agent.invoke({"messages":[{"role": "user", "content": user_query}]})
response = agent.invoke({"messages":[{"role": "user", "content": "Hi there"}]})
#print(response["messages"][-1].content[0]["text"])
#print(response["messages"][-1].content)

def chat(message, history):
    response = agent.invoke({"messages":[{"role": "user", "content": message}]})
    last_response = response["messages"][-1].content
    return last_response

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    gr.ChatInterface(fn=chat)

demo.launch()