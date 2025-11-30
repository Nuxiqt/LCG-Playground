"""Business logic handlers for the API endpoints.
Contains the actual implementation of code generation, completion, and chat functionality."""
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from typing import Dict, Union
from logger import log
import os

# Model cache to avoid recreating instances
_models: Dict[str, Union[OllamaLLM, ChatOpenAI]] = {}


def get_model(model_name: str, temperature: float = 0.3) -> Union[OllamaLLM, ChatOpenAI]:
    """
    Get or create a model instance with caching.
    Supports both Ollama and OpenAI models.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4', 'qwen2.5-coder:14b-instruct')
        temperature: Temperature for generation (0.0 to 1.0)
    
    Returns:
        OllamaLLM or ChatOpenAI instance
    """
    key = f"{model_name}_{temperature}"
    if key not in _models:
        log.info("Initializing model", model=model_name, temperature=temperature)
        
        # Check if it's an OpenAI model
        if model_name.startswith(('gpt-', 'o1-', 'o3-')):
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            _models[key] = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
            )
        else:
            # Assume it's an Ollama model
            _models[key] = OllamaLLM(
                model=model_name,
                temperature=temperature,
            )
    return _models[key]


def generate_code(prompt: str, model: str, temperature: float) -> str:
    """
    Generate code or explanations based on a prompt.
    
    Args:
        prompt: The user's request
        model: Model name to use
        temperature: Generation temperature
    
    Returns:
        Generated response
    """
    log.info("Generating code", model=model, prompt_length=len(prompt))
    llm = get_model(model, temperature)
    response = llm.invoke(prompt)
    log.info("Code generated", response_length=len(response))
    return response


def complete_code(prompt: str, model: str, temperature: float) -> str:
    """
    Complete partial code snippets.
    
    Args:
        prompt: Partial code to complete
        model: Model name to use
        temperature: Generation temperature
    
    Returns:
        Code completion
    """
    log.info("Completing code", model=model, prompt_length=len(prompt))
    llm = get_model(model, temperature)
    response = llm.invoke(prompt)
    log.info("Code completed", response_length=len(response))
    return response


def chat(prompt: str, model: str, temperature: float) -> str:
    """
    Handle chat-style coding questions and explanations.
    
    Args:
        prompt: User's question or request
        model: Model name to use
        temperature: Generation temperature
    
    Returns:
        Response to the question
    """
    log.info("Processing chat request", model=model, prompt_length=len(prompt))
    llm = get_model(model, temperature)
    response = llm.invoke(prompt)
    log.info("Chat response generated", response_length=len(response))
    return response


def get_available_models() -> list:
    """
    Get list of available models with their descriptions.
    
    Returns:
        List of model information dictionaries
    """
    models = [
        {
            "name": "qwen2.5-coder:14b-instruct",
            "description": "14B parameter instruction-tuned code model (Ollama)",
            "provider": "ollama",
            "best_for": ["code generation", "explanation", "chat"]
        },
        {
            "name": "starcoder2:15b",
            "description": "15B parameter base code model (Ollama)",
            "provider": "ollama",
            "best_for": ["code completion", "fill-in-middle"]
        }
    ]
    
    # Add OpenAI models if API key is available
    if os.getenv('OPENAI_API_KEY'):
        models.extend([
            {
                "name": "gpt-4o",
                "description": "Most advanced GPT-4 model (OpenAI)",
                "provider": "openai",
                "best_for": ["code generation", "complex reasoning", "debugging"]
            },
            {
                "name": "gpt-4o-mini",
                "description": "Fast and affordable GPT-4 model (OpenAI)",
                "provider": "openai",
                "best_for": ["code generation", "simple tasks", "chat"]
            },
            {
                "name": "gpt-4-turbo",
                "description": "GPT-4 Turbo model (OpenAI)",
                "provider": "openai",
                "best_for": ["code generation", "analysis", "refactoring"]
            },
            {
                "name": "o1",
                "description": "Advanced reasoning model (OpenAI)",
                "provider": "openai",
                "best_for": ["complex problems", "algorithm design", "debugging"]
            }
        ])
    
    return models


def health_check() -> dict:
    """
    Perform health check.
    
    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "ollama": "running"
    }
