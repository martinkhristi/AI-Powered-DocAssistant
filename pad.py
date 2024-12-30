import streamlit as st
import os
import time
import base64
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key or not google_api_key:
    st.error("Missing API keys. Please check your .env file.")
    st.stop()

# Set Google API key as an environment variable
os.environ["GOOGLE_API_KEY"] = google_api_key

# Page configuration
st.set_page_config(layout="wide", page_title="Document Assistant", page_icon="📄")
st.sidebar.title("How to Use")
st.sidebar.info(
    """
    1. Upload a PDF document.
    2. Process the document for vector embeddings.
    3. Ask questions about the document in the Query tab.
    4. View relevant passages for your query.
    """
)

st.title("📄 RAG Doc Assistant Using Groq API")

# Initialize LLM and prompt template
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a document assistant that helps users to find information in a context.
    Please provide the most accurate response based on the context and inputs.
    Only give information that is in the context, not in general.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to display PDF
def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    return pdf_display

# Function to process the uploaded PDF
def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        with open("temp_uploaded_file.pdf", "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        with st.spinner("Processing document..."):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        os.remove("temp_uploaded_file.pdf")
        st.success("Document processed successfully!")

# Tabs for UI
tab1, tab2 = st.tabs(["📂 Document Upload", "🔍 Query & Answers"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload & Process Your Document")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file:
        # Display PDF
        st.markdown("### Document Preview")
        pdf_display = display_pdf(uploaded_file)
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Embedding button
        if st.button("Process Document"):
            vector_embedding(uploaded_file)

# Tab 2: Query & Answers
with tab2:
    st.header("Ask Questions About the Document")
    if "vectors" not in st.session_state:
        st.warning("Please upload and process a document first in the **Document Upload** tab.")
    else:
        prompt1 = st.text_input("Enter Your Question")

        if prompt1:
            with st.spinner("Generating response..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start

                # Display Response
                st.markdown("### Answer")
                st.write(response['answer'])
                st.markdown(f"**Response Time:** {response_time:.2f} seconds")

                # Display Relevant Passages
                with st.expander("View Relevant Passages"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Passage {i+1}**")
                        st.write(doc.page_content)
                        st.markdown("---")
