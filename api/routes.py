"""
API routes and endpoints for the Local Code LLM API.
Defines all HTTP endpoints and delegates business logic to handlers.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from . import handlers
from logger import log

app = FastAPI(
    title="Local Code LLM API",
    description="FastAPI endpoint for Ollama models (Qwen2.5-Coder, StarCoder2) and OpenAI models (GPT-4, GPT-4o)",
    version="1.0.0"
)


# Request/Response models
class CodeRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen2.5-coder:14b-instruct"
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 2000


class CodeResponse(BaseModel):
    response: str
    model: str


@app.get("/")
async def root():
    """Root endpoint with API info."""
    import os
    available = ["qwen2.5-coder:14b-instruct", "starcoder2:15b"]
    if os.getenv('OPENAI_API_KEY'):
        available.extend(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1"])
    
    return {
        "message": "Local Code LLM API",
        "available_models": available,
        "endpoints": {
            "POST /generate": "Generate code or get explanations",
            "POST /complete": "Complete partial code",
            "POST /chat": "Chat with the model",
            "GET /models": "List available models",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return handlers.health_check()


@app.get("/models")
async def list_models():
    """List available models."""
    return {"models": handlers.get_available_models()}


@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """
    Generate code, explanations, or answer coding questions.
    
    Best with: qwen2.5-coder:14b-instruct
    
    Example:
    ```json
    {
        "prompt": "Write a Python function to reverse a linked list",
        "model": "qwen2.5-coder:14b-instruct",
        "temperature": 0.3
    }
    ```
    """
    try:
        log.info("POST /generate", model=request.model)
        response = handlers.generate_code(
            request.prompt,
            request.model,
            request.temperature
        )
        
        return CodeResponse(
            response=response,
            model=request.model
        )
    except Exception as e:
        log.error("Error generating code", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/complete", response_model=CodeResponse)
async def complete_code(request: CodeRequest):
    """
    Complete partial code snippets.
    
    Works with both models, but starcoder2:15b excels at this.
    
    Example:
    ```json
    {
        "prompt": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    ",
        "model": "qwen2.5-coder:14b-instruct"
    }
    ```
    """
    try:
        log.info("POST /complete", model=request.model)
        response = handlers.complete_code(
            request.prompt,
            request.model,
            request.temperature
        )
        
        return CodeResponse(
            response=response,
            model=request.model
        )
    except Exception as e:
        log.error("Error completing code", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat", response_model=CodeResponse)
async def chat(request: CodeRequest):
    """
    Chat interface for coding questions and explanations.
    
    Best with: qwen2.5-coder:14b-instruct
    
    Example:
    ```json
    {
        "prompt": "Explain the difference between deep copy and shallow copy in Python",
        "model": "qwen2.5-coder:14b-instruct"
    }
    ```
    """
    try:
        log.info("POST /chat", model=request.model)
        response = handlers.chat(
            request.prompt,
            request.model,
            request.temperature
        )
        
        return CodeResponse(
            response=response,
            model=request.model
        )
    except Exception as e:
        log.error("Error in chat", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

