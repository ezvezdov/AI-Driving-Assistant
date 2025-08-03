import os
from typing import List, Tuple, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

import torch

from secrets import OPENAI_API_KEY

# Configuration
PDF_PATH = "examples/pdf/psp.pdf"  # Path to your PDF file
MODEL_NAME = "ufal/robeczech-base"  # RobeCzech model
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to save/load FAISS DB
gpt_model = "gpt-4o-mini"

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pdf(pdf_path: str) -> List[Document]:
    """Load PDF file and return a list of Document objects.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def split_documents(documents: List[Document],
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of split Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    return splits


# 2. Create embeddings and vector store
def create_vector_store(splits: List[Document], save_path: str = DB_FAISS_PATH) -> FAISS:
    """Create and save vector store from document chunks.

    Args:
        splits: List of document chunks
        save_path: Path to save the vector store

    Returns:
        FAISS vector store instance
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': torch_device},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Create and save vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

# 3. Set up the RAG chain
def setup_rag_chain(vectorstore: VectorStore) -> Runnable:
    """Create RAG chain with retriever and LLM.

    Args:
        vectorstore: Vector store instance

    Returns:
        Configured RAG chain
    """
    # Initialize GPT model
    llm = ChatOpenAI(
        model_name=gpt_model,
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

    # Define prompt template
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



def main() -> None:
    """Main function to run the RAG system."""
    
    # Check if vector store exists
    if not os.path.exists(DB_FAISS_PATH):
        document = load_pdf(PDF_PATH)
        splits = split_documents(document)

        # Create and save vector store
        vectorstore = create_vector_store(splits)
    else:
        print("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Set up RAG chain
    rag_chain = setup_rag_chain(vectorstore)

    # Example usage
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer = rag_chain.invoke(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
