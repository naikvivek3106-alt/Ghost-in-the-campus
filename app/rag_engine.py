import os
import shutil
from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import GOOGLE_API_KEY, DOCS_DIR, INDEX_DIR

SPECTRAL_SYSTEM_PROMPT = """You are "The Campus Spectre", the eternal ghost lingering in the university's halls for over a century.
You know every timetable, lab regulation, shortcut, and secret of this campus.

Tone & Personality:
1. Speak with eerie wit, gothic humor, and spectral sarcasm (e.g., "Ah, another mortal wandering the earthly plane...", "Beware the 3rd-floor printer; it feasts on hopes and paper jams").
2. ALWAYS provide 100% accurate factual details (timings, lab numbers, dress codes, deadlines) based on the context provided below.
3. If an answer involves campus rules, playfully warn the mortal of the dire consequences of defiance.
4. Keep the answer direct and readable—never write endless walls of text.
5. If the context does not contain the answer, do NOT hallucinate facts. Reply in character:
   "The ethereal mists cloud my vision on this matter, mortal. You must seek the living bureaucrats in the Admin Block."

Context from campus archives:
{context}
"""

class SpectralRAGEngine:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
        )

        self.llm = ChatGoogleGenerativeAI(
         model="gemini-3.5-flash",
        temperature=0.7,
         google_api_key=GOOGLE_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SPECTRAL_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.vector_store = None
        self.load_or_build_index()

    def build_index(self):
        """Indexes all documents found in DOCS_DIR and saves to INDEX_DIR."""
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR, exist_ok=True)
            # Create a placeholder document if empty
            sample_doc = os.path.join(DOCS_DIR, "general_rules.txt")
            with open(sample_doc, "w", encoding="utf-8") as f:
                f.write("Main Gate closes strictly at 10:00 PM. Identity cards are mandatory for entry.\n"
                        "Computer Lab 301 is open from 9:00 AM to 5:00 PM Monday through Friday.")

        loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            raise RuntimeError(f"No text documents found in directory: {DOCS_DIR}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = splitter.split_documents(documents)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(INDEX_DIR)

    def load_or_build_index(self):
        """Loads saved FAISS index if present; otherwise creates it."""
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            self.vector_store = FAISS.load_local(
                INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.build_index()

    def reindex(self) -> int:
        """Forces a clean rebuild of the knowledge base."""
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        self.build_index()
        # Return count of chunks indexed
        return self.vector_store.index.ntotal

    def query(self, question: str) -> Tuple[str, List[str]]:
        """Retrieves relevant context and generates a ghost response."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        matched_docs = retriever.invoke(question)

        context = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])
        sources = [doc.metadata.get("source", "unknown") for doc in matched_docs]

        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        return response, list(set(sources))