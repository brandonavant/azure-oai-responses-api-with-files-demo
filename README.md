# Azure OpenAI Responses API — Document Q&A Demo

A Jupyter notebook demonstrating document Q&A using the Azure OpenAI Responses API with GPT-5.2.

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
| `MODEL_DEPLOYMENT_NAME` | Yes | Your GPT-5.2 deployment name |
| `REASONING_EFFORT` | No | `low`, `medium`, or `high` (default: `medium`) |
| `DOCUMENT_PATH` | No | Path to a PDF to query (default: uses built-in sample) |

A sample document (`sample_document.pdf`) is included to get started immediately.

## License

MIT — see [LICENSE](LICENSE).
