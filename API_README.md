# Local Code LLM API

FastAPI-based REST API for interacting with local Ollama models (Qwen2.5-Coder and StarCoder2) and OpenAI models (GPT-4, GPT-4o, o1).

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure your API keys (optional for OpenAI):
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY if you want to use OpenAI models
```

3. Make sure Ollama is running with models installed:
```bash
ollama list
# Should show: qwen2.5-coder:14b-instruct and starcoder2:15b
```

## Available Models

### Ollama Models (Local)
- `qwen2.5-coder:14b-instruct` - Best for code generation, explanation, and chat
- `starcoder2:15b` - Best for code completion

### OpenAI Models (Requires API Key)
- `gpt-4o` - Most advanced GPT-4 model
- `gpt-4o-mini` - Fast and affordable GPT-4 model
- `gpt-4-turbo` - GPT-4 Turbo model
- `o1` - Advanced reasoning model

## Running the API

Start the server:
```bash
uv run python api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## Endpoints

### GET `/`
API information and available endpoints.

### GET `/health`
Health check to verify the API is running.

### GET `/models`
List available models and their capabilities.

### POST `/generate`
Generate code or get explanations.

**Request:**
```json
{
  "prompt": "Write a Python function to reverse a linked list",
  "model": "qwen2.5-coder:14b-instruct",
  "temperature": 0.3,
  "max_tokens": 2000
}
```

### POST `/complete`
Complete partial code snippets.

**Request:**
```json
{
  "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
  "model": "qwen2.5-coder:14b-instruct"
}
```

### POST `/chat`
Chat interface for coding questions.

**Request:**
```json
{
  "prompt": "Explain the difference between deep copy and shallow copy",
  "model": "qwen2.5-coder:14b-instruct"
}
```

## Testing

Run the test script:
```bash
uv run python test_api.py
```

## Example cURL Commands

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Generate code with Ollama model
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to check if a number is prime",
    "model": "qwen2.5-coder:14b-instruct"
  }'

# Generate code with OpenAI model (requires API key)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to check if a number is prime",
    "model": "gpt-4o-mini",
    "temperature": 0.7
  }'

# Chat with GPT-4o
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the Big O notation of bubble sort?",
    "model": "gpt-4o"
  }'
```

## Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Write a function to find the longest common subsequence",
        "model": "qwen2.5-coder:14b-instruct",
        "temperature": 0.3
    }
)

result = response.json()
print(result["response"])
```

## Model Recommendations

### Local Models (Ollama)
- **qwen2.5-coder:14b-instruct**: Best for code generation, explanations, and chat
- **starcoder2:15b**: Best for code completion and fill-in-middle tasks

### OpenAI Models (Requires API Key)
- **gpt-4o**: Most capable, best for complex tasks and debugging
- **gpt-4o-mini**: Fast and cost-effective, good for most tasks
- **gpt-4-turbo**: Great balance of speed and capability
- **o1**: Advanced reasoning, best for algorithm design and complex problems

## Configuration

Add your OpenAI API key to `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```

Without the API key, only Ollama models will be available.
