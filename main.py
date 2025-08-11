import os
import shutil
import logging
import argparse
import warnings
import importlib
from pathlib import Path

import torch
from typing import List, Tuple, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder



# Suppress noisy FutureWarnings from deps to keep console clean
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# OpenAI key taken from env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set torch device
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# CLI ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser()

parser.add_argument("--country", choices=['Czechia', 'Belarus', 'UK'], required=True, type=str, help="Country for which you want to retrieve driving regulations [Czechia, Belarus, UK].")
parser.add_argument("--language", type=str, help="Language.")
parser.add_argument("--embedding_model", type=str, help="Embedding model (for vector search)")
parser.add_argument("--rewriter_model", type=str, help="Rewriter model")
parser.add_argument("--guardrails_model", type=str, help="Guardrails model")
parser.add_argument("--reranker_model", type=str, help="Reranker model")
parser.add_argument("--conversational_llm", type=str, help="Conversational LLM")
parser.add_argument("--db_path", default="vectorstore", type=str, help="Path to save/load FAISS DB")
parser.add_argument("--documents_path", default="documents", type=str, help="Path to documents")
parser.add_argument("--vectorstore_recreate", default=False, action='store_true', help="Recreate vectorstore from documents, if it exists")
parser.add_argument("--top_k", default=5, type=int, help="Number of top documents to return after reranking")
parser.add_argument("--chunk_size", default=1000, type=int, help="Max characters per chunk after splitting")
parser.add_argument("--chunk_overlap", default=200, type=int, help="Overlap size between adjacent chunks")



class ProcessorPDF():
    def __init__(self, folder_path: Path, chunk_size: int, chunk_overlap: int) -> None:
        self.folder_path = folder_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        self.documents = self.reload_documents()


    def reload_documents(self) -> List[Document]:
        pdf_files_paths = self.get_pdf_paths()

        documents = []
        for pdf_path in pdf_files_paths:
            documents.extend(self.load_pdf(pdf_path))
        
        return documents

    def get_pdf_paths(self) -> List[str]:
        # Check if the folder exists
        if not self.folder_path.exists():
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")
        
        pdf_files = []

        # Walk through the directory and add PDF files to the list
        for pdf_path in Path(self.folder_path).rglob("*.pdf"):
            if pdf_path.suffix.lower() == ".pdf":
                pdf_files.append(pdf_path)

        return pdf_files


    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load PDF file and return a list of Document objects.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    def split_documents(self) -> List[Document]:
        """
        Split documents into smaller chunks.

        Returns:
            List of split Document objects
        """

        splits = self.text_splitter.split_documents(self.documents)

        return splits


class HybridRetriever():
    def __init__(self, vectorstore_path: Path, embedding_model: str, documents_path: Path, vectorstore_recreate: bool, top_k: int, chunk_size: int, chunk_overlap: int) -> None:

        # Set vectorstore path
        self.vectorstore_path = vectorstore_path

        # Set documents path
        self.documents_path = documents_path

        # Set top_k
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': TORCH_DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

        self.reload_retriever(vectorstore_recreate=vectorstore_recreate)
        

    def invoke(self, user_query: str) -> List[Document]:
        return self.hybrid_retriever.invoke(user_query)


    def reload_retriever(self, vectorstore_recreate: bool = True) -> None:

        # PDF processor
        processor_pdf = ProcessorPDF(self.documents_path, self.chunk_size, self.chunk_overlap)

        # Split documents
        splits = processor_pdf.split_documents()

        # Vector search
        self.vectorstore = self.create_vector_store(splits, vectorstore_recreate)

        # Keyword search
        self.bm25_retriever = BM25Retriever.from_documents(splits)

        # Combine searches
        self.hybrid_retriever = self.get_ensembled_retriever()



    def create_vector_store(self, splits: List[Document], vectorstore_recreate: bool) -> FAISS:
        """Create and save vector store from document chunks.

        Args:
            splits: List of document chunks
            save_path: Path to save the vector store

        Returns:
            FAISS vector store instance
        """

        # Create and save vector store
        if not self.vectorstore_path.exists() or vectorstore_recreate:
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vectorstore.save_local(self.vectorstore_path)

        # Load existing vector store
        else:
            vectorstore = FAISS.load_local(
                self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    def get_ensembled_retriever(self) -> EnsembleRetriever:
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        return EnsembleRetriever(
            retrievers=[faiss_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )


class ConversationalLLM():
    def __init__(self, conversational_llm: str, conversational_llm_prompt_text: str):
        self.prompt = ChatPromptTemplate.from_template(
            conversational_llm_prompt_text)

        self.llm = ChatOpenAI(
            model_name=conversational_llm,
            openai_api_key=OPENAI_API_KEY
        )

    def ask_llm(self, question: str, context: str):
        formatted_prompt = self.prompt.format(
            question=question, context=context)
        response = self.llm.invoke(formatted_prompt)
        return response.content


class Rewriter:
    """Class to handle query rewriting using a language model."""

    def __init__(self, rewriter_llm: str, prompt_text: str):
        self.llm = ChatOpenAI(
            model_name=rewriter_llm,
            openai_api_key=OPENAI_API_KEY
        )
        self.prompt = PromptTemplate.from_template(prompt_text)

    def rewrite(self, user_query: str) -> List[str]:
        """Rewrite the user's query into multiple versions."""
        formatted_prompt = self.prompt.format(user_query=user_query)
        response = self.llm.invoke(formatted_prompt)
        versions = [v.strip() for v in response.content.split(
            "===") if v and not v.startswith("Version")]

        return versions


class Reranker:
    """Class to handle reranking of retrieved documents."""

    def __init__(self, hf_model_name: str, top_k: int):
        self.model = CrossEncoder(hf_model_name, device=TORCH_DEVICE)
        self.top_k = top_k

    def rerank(self, user_query: str, docs: List[Document]) -> List[Document]:
        # Prepare pairs for scoring
        pairs = [(user_query, doc.page_content) for doc in docs]

        # Predict scores using the cross-encoder model
        scores = self.model.predict(pairs)

        # Sort docs by score, descending
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # Return only `self.top_k`
        reranked_docs = [doc for doc, score in reranked[:self.top_k]]

        return reranked_docs


class Guardrails:
    def __init__(self, guardrails_llm: str, input_prompt_text: str, output_prompt_text: str) -> None:
        self.llm = ChatOpenAI(
            model_name=guardrails_llm,
            openai_api_key=OPENAI_API_KEY
        )
        self.input_prompt = PromptTemplate.from_template(input_prompt_text)
        self.output_prompt = PromptTemplate.from_template(output_prompt_text)

    def check_input(self, user_query: str) -> bool:
        formatted_prompt = self.input_prompt.format(user_query=user_query)
        response = self.llm.invoke(formatted_prompt)
        if "yes" in response.content.lower():
            return False
        else:
            return True
            

    def check_output(self, model_output: str) -> bool:
        formatted_prompt = self.output_prompt.format(model_output=model_output)
        response = self.llm.invoke(formatted_prompt)
        if "yes" in response.content.lower():
            return False
        else:
            return True


def main(args: argparse.Namespace) -> None:
    """Main function to run the RAG system."""

    # Print welcome message
    print(config.welcome_message)

    documents_country_path = Path(args.documents_path) / args.country

    # Retrieve available languages
    available_languages = [d.name for d in documents_country_path.iterdir() if d.is_dir()]
    available_languages_str = ", ".join(available_languages)

    language = None

    # Use provided language (from --language) if it possible
    if not args.language is None:
        if args.language in available_languages:
            language = args.language
        else:
            print("Language selected by --language is not available for this country.\nAvailable languages: ", available_languages_str)
        
    
    if len(available_languages) == 1:
        language = available_languages[0]
    elif len(available_languages) == 0:
        print("Sorry, there are not documents for this country.")
    else:
        
        
        while language not in available_languages:
            language = input(f"Please, choose the language [{available_languages_str}] : ")
        

    config = importlib.import_module(f"locales.{language}.config")
    
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.rewriter_model:
        config.rewriter_model = args.rewriter_model
    if args.guardrails_model:
        config.guardrails_model = args.guardrails_model
    if args.reranker_model:
        config.reranker_model = args.reranker_model
    if args.conversational_llm:
        config.conversational_llm = args.conversational_llm
    
    # Set path to folder that contains documents
    documents_path = Path(args.documents_path) / args.country / language

    # Set path to vector database
    db_path = Path(args.db_path) / args.country / language
    for part in config.embedding_model.split("/"):
        db_path = db_path / part
    db_path = db_path / "db_faiss"


    # Setup HybridRetriever
    hybrid_retriever = HybridRetriever(db_path, config.embedding_model, documents_path, args.vectorstore_recreate, args.top_k, args.chunk_size, args.chunk_overlap)

    # Setup Conversational LLM
    conversational_llm = ConversationalLLM(
        config.conversational_llm, config.conversational_llm_prompt_text)

    # Setup Rewriter LLM and prompt
    rewriter_llm = Rewriter(
        rewriter_llm=config.rewriter_model,
        prompt_text=config.rewriter_prompt_text
    )

    # Setup Reranker
    reranker = Reranker(config.reranker_model, args.top_k)

    # Setup Guardrails
    guardrails = Guardrails(
        guardrails_llm=config.guardrails_model,
        input_prompt_text=config.guardrails_input_prompt_text,
        output_prompt_text=config.guardrails_output_prompt_text
    )

    # RAG main loop
    while True:
        # Get user query
        print("-" * shutil.get_terminal_size().columns)
        user_query = input(config.ask_message)

        # Check for special commands
        if user_query.lower() == '/quit':
            break
        elif user_query.lower() == '/help':
            print(config.help_message)
            continue
        elif user_query.lower() == '/reload':
            print(config.start_document_rescan_message)
            hybrid_retriever.reload_retriever()
            print(config.end_document_rescan_message)
            continue



        # Check input
        guardrails_check = guardrails.check_input(user_query)
        if not guardrails_check:
            print(config.question_block_message)
            continue

        # Rewrite the user query into multiple versions
        query_versions = rewriter_llm.rewrite(user_query)

        # Independently retrieve documents for each rewritten query
        docs = []
        for q in query_versions:
            docs.extend(hybrid_retriever.invoke(q))

        # Rerank documents
        reranked_docs = reranker.rerank(user_query, docs)

        # Concatenate all retrieved documents as context
        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        # Use original user_query and all docs as context
        answer = conversational_llm.ask_llm(user_query, context)

        # Output guardrails
        guardrails_check = guardrails.check_output(answer)
        if not guardrails_check:
            print(config.conversational_llm_output_block_message)
            continue
        
        # Print the answer
        print("\n" + config.answer_message, answer)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)
