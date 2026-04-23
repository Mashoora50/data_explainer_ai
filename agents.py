# agents.py
# This file contains our two AI agents:
# 1. Data Analyst Agent — analyzes the data
# 2. Explainer Agent — explains results in simple language
# They work together as a multi-agent system!

import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_engine import retrieve_relevant_context, get_dataset_summary

load_dotenv()

# Set up the Groq client (Groq uses OpenAI-compatible API)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Memory: stores conversation history for the chatbot
conversation_memory = []
MAX_MEMORY = 10  # Keep last 10 exchanges

def add_to_memory(role: str, content: str):
    """Add a message to conversation history"""
    global conversation_memory
    conversation_memory.append({"role": role, "content": content})
    # Keep memory size limited (sliding window)
    if len(conversation_memory) > MAX_MEMORY * 2:
        conversation_memory = conversation_memory[-MAX_MEMORY * 2:]

def clear_memory():
    """Reset conversation when new file is uploaded"""
    global conversation_memory
    conversation_memory = []

def data_analyst_agent(user_question: str, context: str) -> str:
    """
    AGENT 1: Data Analyst
    This agent looks at the raw data and identifies facts, patterns, trends.
    It speaks like a professional data scientist.
    """
    
    system_prompt = """You are an expert Data Analyst AI Agent. Your ONLY job is to analyze datasets.

RULES:
1. Only answer questions about the provided dataset context
2. Be precise with numbers and statistics
3. Identify patterns, trends, outliers, and correlations
4. If the question cannot be answered from the data, say so clearly
5. Do NOT make up data that isn't in the context
6. Keep your analysis factual and structured

You are Agent 1 in a two-agent pipeline. Your output will be passed to the Explainer Agent."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
Dataset Context (retrieved from vector store):
{context}

User Question: {user_question}

Provide a detailed data analysis answering this question. Use specific numbers from the data.
"""}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Groq's fast Llama model
        messages=messages,
        temperature=0.1,  # Low temperature = more factual
        max_tokens=800
    )
    
    return response.choices[0].message.content


def explainer_agent(analyst_output: str, user_question: str) -> str:
    """
    AGENT 2: Explainer
    Takes the analyst's technical output and makes it simple & friendly.
    Speaks like a teacher explaining to a non-technical person.
    """
    
    system_prompt = """You are a friendly Data Explainer AI Agent. Your job is to take technical data analysis and explain it in simple, clear language that anyone can understand.

RULES:
1. Use simple everyday language (no jargon)
2. Use analogies when helpful
3. Highlight the most important insights
4. Use bullet points and short sentences
5. Be encouraging and conversational
6. Add context about WHY a finding matters
7. End with 1-2 follow-up questions the user might want to explore

You are Agent 2. You receive output from the Data Analyst Agent and simplify it."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
Original user question: {user_question}

Data Analyst's technical findings:
{analyst_output}

Now explain these findings in simple, friendly language for someone who is NOT a data expert.
"""}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,  # Slightly higher = more natural/friendly
        max_tokens=600
    )
    
    return response.choices[0].message.content


def run_multi_agent_pipeline(user_question: str) -> dict:
    """
    Runs both agents in sequence (multi-agent pipeline):
    User Question → [RAG Retrieval] → Agent 1 (Analyst) → Agent 2 (Explainer) → Response
    
    Returns dict with both outputs so we can show them in the UI
    """
    
    # Step 1: Retrieve relevant context from vector store (RAG)
    context = retrieve_relevant_context(user_question, k=4)
    
    # Step 2: Run Data Analyst Agent
    analyst_output = data_analyst_agent(user_question, context)
    
    # Step 3: Run Explainer Agent with analyst's output
    explainer_output = explainer_agent(analyst_output, user_question)
    
    # Step 4: Add to memory
    add_to_memory("user", user_question)
    add_to_memory("assistant", explainer_output)
    
    return {
        "analyst_output": analyst_output,
        "explainer_output": explainer_output,
        "context_used": context[:500] + "..." if len(context) > 500 else context
    }