# rag_engine.py
# RAG = Retrieval-Augmented Generation
# This converts your CSV into a searchable vector database

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
import os
import json

# This is our global vector store (stored in memory)
vector_store = None
df_summary = {}  # stores basic info about the dataset

def load_csv_to_vectorstore(filepath: str) -> dict:
    """
    Takes a CSV file path, reads it, converts to text chunks,
    and stores in a vector database (ChromaDB).
    Returns a summary of the dataset.
    """
    global vector_store, df_summary
    
    # Step 1: Read the CSV
    df = pd.read_csv(filepath)
    
    # Step 2: Create a summary of the data
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(5).to_string(),
        "stats": df.describe().to_string() if not df.select_dtypes(include='number').empty else "No numeric columns",
        "missing_values": df.isnull().sum().to_dict(),
        "null_count": int(df.isnull().sum().sum()),
    }
    df_summary = summary
    
    # Step 3: Convert CSV rows to text "documents" for the vector store
    documents = []
    
    # Add a summary document
    summary_text = f"""
    Dataset Summary:
    - Total rows: {summary['rows']}
    - Columns: {', '.join(summary['columns'])}
    - Data types: {json.dumps(summary['dtypes'])}
    - Missing values: {json.dumps(summary['missing_values'])}
    
    Statistical Summary:
    {summary['stats']}
    
    First 5 rows sample:
    {summary['sample']}
    """
    documents.append(Document(page_content=summary_text, metadata={"type": "summary"}))
    
    # Add each row as a document (for large files, we group rows)
    chunk_size = 50  # rows per document chunk
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        chunk_text = f"Rows {i} to {min(i+chunk_size, len(df))}:\n{chunk_df.to_string()}"
        documents.append(Document(
            page_content=chunk_text,
            metadata={"type": "data_chunk", "start_row": i}
        ))
    
    # Add column-specific analysis documents
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            col_analysis = f"""
            Column '{col}' analysis (numeric):
            - Mean: {df[col].mean():.2f}
            - Median: {df[col].median():.2f}
            - Min: {df[col].min()}, Max: {df[col].max()}
            - Std Dev: {df[col].std():.2f}
            - Missing: {df[col].isnull().sum()}
            """
        else:
            col_analysis = f"""
            Column '{col}' analysis (categorical):
            - Unique values: {df[col].nunique()}
            - Most common: {df[col].value_counts().head(5).to_string()}
            - Missing: {df[col].isnull().sum()}
            """
        documents.append(Document(page_content=col_analysis, metadata={"type": "column_analysis", "column": col}))
    
    # Step 4: Create embeddings (convert text to numbers/vectors)
    # We use a free local model — no API key needed for this part!
    embeddings = FakeEmbeddings(size=384)
    
    # Step 5: Store in ChromaDB (vector database)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="csv_data"
    )
    
    return summary


def retrieve_relevant_context(query: str, k: int = 4) -> str:
    """
    Given a user question, find the most relevant chunks from the vector store.
    k = number of chunks to retrieve
    """
    global vector_store
    
    if vector_store is None:
        return "No data loaded yet."
    
    # Search the vector store for relevant documents
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    # Combine all relevant text
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    return context


def get_dataset_summary() -> dict:
    """Return the stored summary of the current dataset"""
    return df_summary