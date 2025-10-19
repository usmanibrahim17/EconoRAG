🌍 EconoLens: Retrieval-Augmented Chatbot for Comparative Economic Analysis

"An AI-driven exploration of how developing and developed economies diverge, evolve, and adapt."

🧭 Overview

EconoLens is a Retrieval-Augmented Generation (RAG)-based chatbot designed to analyze and compare economic indicators across developing and developed countries.
It integrates structured text datasets with the reasoning power of large language models to produce context-aware, data-backed insights.

This project demonstrates how RAG pipelines can transform static economic data into interactive intelligence — making complex global trends easier to explore, question, and understand.

🎯 Objectives

To compare key economic indicators (GDP growth, FDI, trade, CO₂ emissions, etc.) across country groups.

To experiment with RAG pipelines and evaluate their retrieval quality and context fusion.

To build an accessible, interactive research tool for economics students and data enthusiasts.

🧩 Features

🧠 Intelligent Retrieval: Fetches context from a custom dataset of economic papers and stats.

💬 Conversational Interface: Chat with your dataset through a Gradio-powered UI.

📚 Cited Insights: Each response integrates references from the retrieved text files.

⚡ LLM Reasoning: Combines factual retrieval with analytical synthesis via GPT-4o-mini.

🔍 Modular Design: Built for easy expansion (add new indicators or datasets effortlessly).

🧠 Technical Architecture
Component	Description
Document Loader	Reads all .txt files in the dataset directory.
Text Splitter	Chunks documents into overlapping sections for efficient retrieval.
Embeddings	Converts text chunks into vector representations using OpenAIEmbeddings.
Vector Store (Chroma)	Stores embeddings for semantic retrieval.
Retriever + LLM (LangChain)	Retrieves relevant text and fuses it with GPT-4o-mini responses.
Frontend (Gradio)	Provides an interactive chat interface.
💬 Example Prompts

“Compare FDI trends between Southeast Asia and Western Europe.”

“How does renewable energy adoption differ across OECD and non-OECD nations?”

“What patterns are visible in CO₂ emissions per capita between developed and developing countries?”

“Discuss the relationship between GDP growth and industrial output in emerging economies.”

🚧 Future Work

Integrate live data retrieval from World Bank APIs.

Add data visualization components for comparative graphs.

Implement BERT or SentenceTransformer embeddings for open-source deployment.

Experiment with domain-specific fine-tuning for economics research.