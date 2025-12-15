import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAI #imports the gemini
llm = GoogleGenerativeAI(model="gemini-2.5-flash")



# DATA_LOADING (Load the data (creates ~21 docs, one per page))
from langchain_community.document_loaders import PyPDFLoader 
file_path = "data/SDG.pdf"
loader = PyPDFLoader(file_path)
raw_documents = loader.load()
print("📊 Data is loaded")
print("----------------------------------------")
# print(type(raw_documents))

# MERGE all DOC into a singal DOC
from langchain_core.documents import Document
full_data = "\n\n".join([doc.page_content for doc in raw_documents])
merged_document = Document(page_content=full_data)
print("📈 Merging of document completed")
print("----------------------------------------")
# print(type(merged_document))



# CREATING_CHUNKS_OF_LOADED_DATA
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    # separators=["\n\n", "\n", " ", ""] # Tries to split by paragraph first, then newlines, etc.
)
document_chunks = text_splitter.split_documents([merged_document])
print("📊 Chunks have been created")
print("----------------------------------------")
# print(len(document_chunks))





# PROMPT_FOR_THE_LLM_TO_GENERATE_QUESTIONS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

base_prompt = """
Generate exactly 3 different, simple, questions from the text below.
Avoid repeating the same question style from previous attempts.

TEXT:
{text}

QUESTIONS:
"""

refine_prompt = """
Here are 3 questions: {questions}

Rewrite them to be more unique and fresh.
Make them different in phrasing or angle.
Do NOT increase difficulty.
Do NOT add extra questions.
Keep them simple and related to the text.

REFINED QUESTIONS:
"""

# Base chain
BASE_PROMPT = PromptTemplate(
    template=base_prompt,
    input_variables=["text"]
)

base_chain = BASE_PROMPT | llm | StrOutputParser()

# Refine chain
REFINE_PROMPT = PromptTemplate(
    template=refine_prompt,
    input_variables=["questions"]
)

refine_chain = REFINE_PROMPT | llm | StrOutputParser()

raw_q = base_chain.invoke({"text": merged_document.page_content})
final_q = refine_chain.invoke({"questions": raw_q})
print(final_q)
print("----🤖QUESTION GENERATION COMPLEATED🤖----")
print("----------------------------------------")
print()






from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(document_chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))


print("🤖🤖DATA is embeded now generating answers for the questions...🤖🤖")
print()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

answer_genration_prompt = PromptTemplate(
        template="""
Use ONLY the context to answer the question. 
Keep the answer short, simple, and accurate.

Question: {question}

Context:
{context}

Answer:
""",
input_variables=["question", "context"]
)

retriever = vector_store.as_retriever() 
docs = retriever.invoke(final_q)
context = "\n\n".join(d.page_content for d in docs)

answer_chain = answer_genration_prompt | llm | StrOutputParser()
answer = answer_chain.invoke({"question": final_q, "context": context})
print(answer)
print()
print("🤖🤖-----------------------------🤖🤖")





  



