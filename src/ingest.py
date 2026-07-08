"""Ingestao do PDF no PostgreSQL + pgVector.

Passos:
  1. Le o PDF (PyPDFLoader).
  2. Divide em chunks de 1000 caracteres com overlap de 150.
  3. Converte cada chunk em embedding e armazena no PGVector.

Execucao (a partir da raiz do projeto):
    python src/ingest.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import COLLECTION_NAME, get_vector_store

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def ingest_pdf() -> None:
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise SystemExit(
            f"PDF nao encontrado: {pdf_path.resolve()}\n"
            "Coloque o arquivo na raiz do projeto como 'document.pdf' "
            "ou ajuste a variavel PDF_PATH no .env."
        )

    print(f"Lendo PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Remove chunks vazios (paginas em branco / apenas espacos).
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise SystemExit(
            "Nenhum conteudo textual foi extraido do PDF. "
            "O arquivo pode ser apenas imagem (necessitaria OCR)."
        )

    print(f"Gerando embeddings e armazenando {len(chunks)} chunks...")
    store = get_vector_store(pre_delete_collection=True)

    # Ids deterministicos evitam duplicidade caso a ingestao seja reexecutada.
    ids = [f"doc-chunk-{index}" for index in range(len(chunks))]
    store.add_documents(documents=chunks, ids=ids)

    print(
        f"Ingestao concluida: {len(chunks)} chunks armazenados na "
        f"colecao '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    ingest_pdf()
