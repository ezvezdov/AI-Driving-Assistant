import os
import shutil
import argparse
import warnings
import importlib

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



# Suppress FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set torch device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


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



class ProcessorPDF():
    def __init__(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
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
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")
        
        pdf_files = []

        # Walk through the directory and add PDF files to the list
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):  # Ensures the file is a PDF
                    pdf_files.append(os.path.join(root, file))

        return pdf_files


    def load_pdf(self, pdf_path: str) -> List[Document]:
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
    def __init__(self, vectorstore_path: str, embedding_model: str, documents_path: str, vectorstore_recreate: bool):

        # Set vectorstore path
        self.vectorstore_path = vectorstore_path

        # Set documents path
        self.documents_path = documents_path

        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': torch_device},
            encode_kwargs={'normalize_embeddings': False}
        )

        self.reload_retriever(vectorstore_recreate=vectorstore_recreate)
        

    def invoke(self, user_query: str) -> List[Document]:
        return self.hybrid_retriever.invoke(user_query)


    def reload_retriever(self, vectorstore_recreate: bool = True) -> None:

        # PDF processor
        proceesor_pdf = ProcessorPDF(self.documents_path)

        # Split documents
        splits = proceesor_pdf.split_documents()

        # Vector search
        self.vectorstore = self.create_vector_store(splits, vectorstore_recreate)

        # Keyword search
        self.bm25_retriever = BM25Retriever.from_documents(splits)

        # Combine searches
        self.hybrid_retriever = self.get_ensambled_retriever()



    def create_vector_store(self, splits: List[Document], vectorstore_recreate: bool) -> FAISS:
        """Create and save vector store from document chunks.

        Args:
            splits: List of document chunks
            save_path: Path to save the vector store

        Returns:
            FAISS vector store instance
        """

        # Create and save vector store
        if not os.path.exists(self.vectorstore_path) or vectorstore_recreate:
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vectorstore.save_local(self.vectorstore_path)

        # Load existing vector store
        else:
            vectorstore = FAISS.load_local(
                self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    def get_ensambled_retriever(self) -> EnsembleRetriever:
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        return EnsembleRetriever(
            retrievers=[faiss_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )


class Conversational_LLM():
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

    def __init__(self, hf_model_name: str):
        self.model = CrossEncoder(hf_model_name, device=torch_device)

    def rerank(self, user_query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        # Prepare pairs for scoring
        pairs = [(user_query, doc.page_content) for doc in docs]

        # Predict scores using the cross-encoder model
        scores = self.model.predict(pairs)

        # Sort docs by score, descending
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # Return only top_n
        reranked_docs = [doc for doc, score in reranked[:top_n]]

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
        if "no" in response.content.lower():
            return True
        else:
            return False


def main(args: argparse.Namespace) -> None:
    """Main function to run the RAG system."""

    documents_country_path = os.path.join(args.documents_path, args.country)

    # Retrieve available languages
    available_languages = [d for d in os.listdir(documents_country_path) if os.path.isdir(os.path.join(documents_country_path, d))]
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
    documents_path = os.path.join(args.documents_path, args.country, language)

    # Set path to vector database
    embedding_model_path = config.embedding_model.split("/")
    db_path = os.path.join(args.db_path, args.country, language, *embedding_model_path, "db_faiss")


    # Setup HybridRetriever
    hybrid_retriever = HybridRetriever(db_path, config.embedding_model, documents_path, args.vectorstore_recreate)

    # Setup Conversational LLM
    conversational_llm = Conversational_LLM(
        config.conversational_llm, config.conversational_llm_prompt_text)

    # Setup Rewriter LLM and prompt
    rewriter_llm = Rewriter(
        rewriter_llm=config.rewriter_model,
        prompt_text=config.rewriter_prompt_text
    )

    # Setup Reranker
    reranker = Reranker(config.reranker_model)

    # Setup Guardrails
    guardrails = Guardrails(
        guardrails_llm=config.guardrails_model,
        input_prompt_text=config.guardrails_input_prompt_text,
        output_prompt_text=config.guardrails_output_prompt_text
    )

    

    # Print welcome message
    print(config.welcome_message)

    while True:
        print("-" * shutil.get_terminal_size().columns)
        user_query = input(config.ask_message)
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
        reranked_docs = reranker.rerank(user_query, docs, top_n=5)

        # Concatenate all retrieved documents as context
        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        # Use original user_query and all docs as context
        answer = conversational_llm.ask_llm(user_query, context)

        # Output guardrails
        guardrails_check = guardrails.check_output(answer)
        if not guardrails_check:
            print(config.conversational_llm_output_block_message)
            continue

        print("\n" + config.answer_message, answer)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)
