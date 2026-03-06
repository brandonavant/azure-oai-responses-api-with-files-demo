# Azure OpenAI Responses API — Vector Store File Search Demo

A Jupyter notebook demonstrating document Q&A using the Azure OpenAI Responses API with GPT-5.2 and Vector Store file search.

Documents are uploaded into a **Vector Store** and queried via the `file_search` tool. Both the vector store creation and file uploads are **idempotent** — you can re-run the notebook without creating duplicates.

## Prerequisites

- Python 3.11+
- An Azure OpenAI resource with a GPT-5.2 deployment ([request access](https://aka.ms/oai/gpt5access))

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env with your Azure endpoint, API key, and deployment name

# 4. Launch the notebook
jupyter notebook
```

Open `azure_oai_responses_demo.ipynb` and run all cells top to bottom.

## Configuration

All settings live in `.env` (never committed — covered by `.gitignore`):

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Yes | Your Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes | Your API key |
| `MODEL_DEPLOYMENT_NAME` | No | Your GPT-5.2 deployment name (default: `gpt-5.2`) |
| `REASONING_EFFORT` | No | `low`, `medium`, or `high` (default: `medium`) |
| `VECTOR_STORE_NAME` | No | Name for the vector store (default: `demo-vector-store`) |
| `DOCUMENTS_DIR` | No | Directory containing `.txt` files to upload (default: `documents`) |

A sample document (`documents/sample_document.txt`) is included to get started immediately. Drop additional `.txt` files into the `documents/` directory and re-run the notebook to index them.

## Testing

An end-to-end test script validates idempotent vector store creation, idempotent file upload, and file search queries:

```bash
python test_vector_store_e2e.py
```

## License

MIT — see [LICENSE](LICENSE).
