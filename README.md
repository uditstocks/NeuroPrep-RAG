# NeuroPrep-RAG
RAG-based AI system that generates interview questions and answers grounded strictly in document context.

#Why This Matters
*Most interview generators:*
- Ask generic questions
- Ignore source material
- Hallucinate answers
*This system:*
- Uses RAG (Retrieval-Augmented Generation)
- Grounds every answer in the uploaded document
- Produces consistent, explainable outputs

# RAG Pipeline Overview
*This project follows a clean Retrieval-Augmented Generation (RAG) pipeline to generate interview questions and answers grounded in document context.*
PDF Document
   ↓
Document Loader (PyPDFLoader)
   ↓
Document Merging
   ↓
Text Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings Generation (Google Gemini Embeddings)
   ↓
Vector Store (Chroma DB)
   ↓
Question Generation (LLM)
   ↓
Prompt Refinement (LLM)
   ↓
Context Retrieval (Similarity Search)
   ↓
Answer Generation (Context-Grounded)
