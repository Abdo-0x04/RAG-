# Universal Document RAG Chatbot
A powerful, highly flexible Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. This application allows users to upload PDF documents and chat with them using any OpenAI-compatible API. Whether you want to use OpenAI, Groq, Together AI, or a local model via LM Studio/Ollama, you can connect to it simply by changing the Base URL in the user interface.

🚀 Features
Bring Your Own LLM: Connect to any AI provider that supports the standard OpenAI API format by simply inputting your API key, Base URL, and model name.

Local Embeddings: Uses HuggingFace's all-MiniLM-L6-v2 to generate document embeddings entirely locally, saving API costs and improving privacy.

Document Chat: Upload multiple PDF files, index them into a Chroma vector database, and extract answers seamlessly.

Interactive UI: A clean, responsive sidebar for configurations and a main chat interface powered by Streamlit.

🛠️ Tech Stack
Frontend: Streamlit

Framework: LangChain

Vector Database: Chroma

Embeddings: HuggingFace (sentence-transformers)

LLM Integration: langchain-openai (Dynamic routing)
