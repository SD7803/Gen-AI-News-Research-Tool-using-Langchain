import os
import time
import pickle
import streamlit as st

from dotenv import load_dotenv
key = os.getenv("OPENAI_API_KEY")
print(repr(key))


from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Cleaned up modern chain layout imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

from langchain_huggingface import HuggingFaceEmbeddings



# Load environment variables from .env file if it exists
load_dotenv()

st.set_page_config(page_title="RockyBot", page_icon="📈")

st.title("RockyBot: News Research Tool 📈")

# Sidebar setup
st.sidebar.title("Configuration & Setup")

# Fail-safe: Let you input the key directly in the UI if .env fails
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Guardrail: Check API key before initializing LLM
if not api_key:
    st.error("Please provide an  API Key in your .env file or the sidebar to continue.")
    st.stop()

# Initialize LLM with the verified API key

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # replace with whatever your list_models output showed
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.9,
)
if process_url_clicked:

    if len(urls) == 0:
        st.sidebar.error("Please enter at least one URL.")
        st.stop()

    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading articles... ✅")
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)
    main_placeholder.text("Creating chunks... ✅")

    # Embeddings with explicit API key mapping
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Building FAISS Index... ✅")

    time.sleep(1)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("URLs processed successfully!")

query = st.text_input("Question")

if query:

    if not os.path.exists(file_path):
        st.error("Please process the URLs first.")
        st.stop()

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    # 1. Structure the prompt cleanly for the QA layout
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question using only the provided context. 
    If you do not know the answer, say that you don't know.

    Context: {context}
    Question: {input}
    """)

    # 2. Build modern RAG architecture pipelines
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    # 3. Invoke using the standardized input dictionary format
    result = chain.invoke({"input": query})

    st.header("Answer")
    st.write(result["answer"])

    # 4. Grab metadata and output sources cleanly
    if result.get("context"):
        st.subheader("Sources")
        sources = set()
        for doc in result["context"]:
            source_url = doc.metadata.get("source")
            if source_url:
                sources.add(source_url)
        
        for source in sources:
            st.write(source)