"""
End-to-end test for vector store creation, file upload, and file_search queries
against the Azure OpenAI Responses API.

This script validates:
  1. Idempotent vector store creation (find-or-create by name)
  2. Idempotent batch file upload (skip files already in the store)
  3. Querying the vector store via the file_search tool
  4. Re-running everything is a no-op (true idempotency)
"""

import os

from dotenv import dotenv_values
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = dotenv_values(".env")

AZURE_OPENAI_ENDPOINT = config["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
MODEL_DEPLOYMENT_NAME = config.get("MODEL_DEPLOYMENT_NAME", "gpt-5.2")
VECTOR_STORE_NAME = config.get("VECTOR_STORE_NAME", "demo-vector-store")
DOCUMENTS_DIR = config.get("DOCUMENTS_DIR", "documents")
REASONING_EFFORT = config.get("REASONING_EFFORT", "medium")

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_or_create_vector_store(name: str) -> str:
    """Return the ID of an existing vector store with the given name, or create one.

    Iterates through all vector stores and matches by name so that repeated
    runs reuse the same store (idempotent).
    """
    for vs in client.vector_stores.list():
        if vs.name == name:
            print(f"  Found existing vector store '{name}' (ID: {vs.id})")
            return vs.id

    vs = client.vector_stores.create(name=name)
    print(f"  Created vector store '{name}' (ID: {vs.id})")
    return vs.id


def get_existing_filenames(vector_store_id: str) -> dict[str, str]:
    """Return a dict mapping filename -> file_id for files already in the store."""
    existing: dict[str, str] = {}
    for vs_file in client.vector_stores.files.list(vector_store_id):
        # Retrieve the underlying file object to get the filename
        file_obj = client.files.retrieve(vs_file.id)
        existing[file_obj.filename] = vs_file.id
    return existing


def upload_directory(vector_store_id: str, directory: str) -> list[str]:
    """Upload all supported files from a directory, skipping those already present.

    Returns a list of newly uploaded file IDs.
    """
    existing = get_existing_filenames(vector_store_id)
    print(f"  Files already in store: {list(existing.keys()) or '(none)'}")

    new_file_ids: list[str] = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        if filename in existing:
            print(f"  Skipping '{filename}' — already in vector store")
            continue

        print(f"  Uploading '{filename}' ...", end=" ", flush=True)
        with open(filepath, "rb") as f:
            uploaded = client.files.create(file=f, purpose="assistants")
        # Add the uploaded file to the vector store and poll until processed
        vs_file = client.vector_stores.files.create_and_poll(
            file_id=uploaded.id, vector_store_id=vector_store_id
        )
        print(f"done (status: {vs_file.status})")
        new_file_ids.append(uploaded.id)

    return new_file_ids


def query_vector_store(vector_store_id: str, question: str) -> str:
    """Ask a question using the file_search tool against the vector store."""
    response = client.responses.create(
        model=MODEL_DEPLOYMENT_NAME,
        instructions="Answer questions based only on the documents in the vector store. Be concise.",
        input=question,
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        include=["output[*].file_search_call.search_results"],
    )
    return response.output_text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_idempotent_vector_store():
    """Creating the store twice should return the same ID."""
    print("\n[TEST] Idempotent vector store creation")
    id1 = get_or_create_vector_store(VECTOR_STORE_NAME)
    id2 = get_or_create_vector_store(VECTOR_STORE_NAME)
    assert id1 == id2, f"Expected same ID, got {id1} vs {id2}"
    print("  PASS — same ID returned both times\n")
    return id1


def test_idempotent_file_upload(vector_store_id: str):
    """Uploading the same directory twice should upload zero files the second time."""
    print("[TEST] Idempotent file upload")
    print("  --- First upload ---")
    new1 = upload_directory(vector_store_id, DOCUMENTS_DIR)
    print(f"  Uploaded {len(new1)} new file(s)")

    print("  --- Second upload (should be no-ops) ---")
    new2 = upload_directory(vector_store_id, DOCUMENTS_DIR)
    assert len(new2) == 0, f"Expected 0 new uploads, got {len(new2)}"
    print("  PASS — zero files uploaded on re-run\n")


def test_file_search_query(vector_store_id: str):
    """Query the vector store and verify we get a non-empty answer."""
    print("[TEST] file_search query")
    answer = query_vector_store(vector_store_id, "Summarize the document, please.")
    assert len(answer) > 20, f"Answer too short: {answer!r}"
    print(f"  Answer preview: {answer[:200]}...")
    print("  PASS — got a substantive answer\n")


def test_specific_question(vector_store_id: str):
    """Ask a specific factual question to confirm the content is actually indexed."""
    print("[TEST] Specific factual query")
    answer = query_vector_store(
        vector_store_id, "What is the price of the Pro Plan per user per month?"
    )
    assert "79" in answer, f"Expected '$79' in answer, got: {answer!r}"
    print(f"  Answer: {answer[:300]}")
    print("  PASS — correct factual answer\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Vector Store E2E Test")
    print("=" * 60)

    vs_id = test_idempotent_vector_store()
    test_idempotent_file_upload(vs_id)
    test_file_search_query(vs_id)
    test_specific_question(vs_id)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
