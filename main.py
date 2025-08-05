import os
from typing import List, Tuple, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import torch

from prompts.en import rewriter_prompt_text
from _secrets import OPENAI_API_KEY

# Configuration
PDF_PATH = "examples/pdf/psp.pdf"  # Path to your PDF file
MODEL_NAME = "ufal/robeczech-base"  # RobeCzech model
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to save/load FAISS DB
gpt_model = "gpt-4o-mini"
gpt_rewrtiter = "gpt-4.1-nano"

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


def create_bm25_retriever(splits: List[Document]) -> BM25Retriever:
    """Create a BM25 keyword retriever from document splits."""
    return BM25Retriever.from_documents(splits)


def setup_rag_chain(vectorstore: VectorStore, bm25_retriever: BM25Retriever) -> Tuple[Runnable, EnsembleRetriever]:
    """Create RAG chain with hybrid retriever and LLM."""
    llm = ChatOpenAI(
        model_name=gpt_model,
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Hybrid retriever: combine vector and keyword retrievers
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    rag_chain = (
        # {"context": hybrid_retriever, "question": RunnablePassthrough()}
        prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, hybrid_retriever

class Rewriter:
    """Class to handle query rewriting using a language model."""

    def __init__(self, rewriter_llm: str , prompt_text: str):
        self.llm  = ChatOpenAI(
            model_name=rewriter_llm,
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        self.prompt = PromptTemplate.from_template(prompt_text)

    def rewrite(self, user_query: str) -> List[str]:
        """Rewrite the user's query into multiple versions."""
        formatted_prompt = self.prompt.format(user_query=user_query)
        response = self.llm.invoke(formatted_prompt)
        versions = [v.strip() for v in response.content.split("===") if v and not v.startswith("Version")]

        return versions

def main() -> None:
    """Main function to run the RAG system."""

    document = load_pdf(PDF_PATH)
    splits = split_documents(document)

    # Check if vector store exists
    if not os.path.exists(DB_FAISS_PATH):
        # Create and save vector store
        vectorstore = create_vector_store(splits)
    else:
        # Load existing vector store
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    bm25_retriever = create_bm25_retriever(splits)
    rag_chain, hybrid_retriever = setup_rag_chain(vectorstore, bm25_retriever)

    # Setup rewriter LLM and prompt
    rewriter_llm = Rewriter(
        rewriter_llm=gpt_rewrtiter,
        prompt_text=rewriter_prompt_text
    )

    

    while True:
        user_query = input("\nAsk a question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break

        query_versions = rewriter_llm.rewrite(user_query)

        docs = []
        for q in query_versions:
            docs.extend(hybrid_retriever.invoke(q))


        # Print retrieved documents
        # Retrieve documents using the hybrid retriever
        # retrieved_docs = hybrid_retriever.invoke(question)
        # print("\nRetrieved Documents:")
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Document {i + 1}: {doc.page_content}")

        # Concatenate all retrieved documents as context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Use original user_query and all docs as context
        answer = rag_chain.invoke({"context": context, "question": user_query})

        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
