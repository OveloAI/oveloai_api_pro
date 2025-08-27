import os
import random
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

def get_ai_response(message: str):
    """
    This is the core RAG function. It takes a user's message, retrieves
    relevant context from the FAISS vector store, and generates a response
    using the Ollama LLM.
    
    Args:
        message (str): The user's query.
    
    Returns:
        dict: A dictionary containing the generated response and confidence score.
    """
    # Load the persisted FAISS vector store using the path from our config file.
    # We now use the same embedding model that created the index.
    embeddings = HuggingFaceEmbeddings(model_name=settings.HUGGINGFACE_EMBEDDING_MODEL)

    if not os.path.exists(settings.FAISS_DB_PATH):
        return {"response": "Error: Knowledge base not found. Please ensure your FAISS vector database has been created.", "confidence": 0.0}
    
    vector_store = FAISS.load_local(
        folder_path=settings.FAISS_DB_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )

    # Set up the retriever to search the FAISS vector store for top 3 results
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize the Ollama LLM using the model name from our config
    llm = Ollama(model=settings.OLLAMA_MODEL)
    
    # Define the prompt template for the LLM. This is where we provide context.
    # The prompt now includes your updated business description.
    template = """
    You are an AI assistant for a web development, automation systems, and high intelligent AI business named OveloAI. 
    Use the following retrieved context to answer the user's question. If you don't 
    know the answer, state that you do not have enough information and can connect 
    them to a human agent. Do not make up any information.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response_text = rag_chain.invoke(message)
    
    # Add a random conversational greeting before the generated response
    greetings = ["Hello!", "Hi there!", "Hey!", "Greetings!"]
    personal_greeting = random.choice(greetings) + " " + response_text
    
    # The confidence score is a placeholder for this example
    confidence = 0.95
    
    return {"response": personal_greeting, "confidence": confidence}
