"""Business logic handlers for the API endpoints.
Contains the actual implementation of code generation, completion, and chat functionality."""
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Dict, Union, Any
from logger import log
import os

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    log.warning("Tavily not installed. Search functionality disabled.")

# Model cache to avoid recreating instances
_models: Dict[str, Union[OllamaLLM, ChatOpenAI]] = {}

# Initialize Tavily client if available
_tavily_client = None
if TAVILY_AVAILABLE:
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if tavily_api_key:
        _tavily_client = TavilyClient(api_key=tavily_api_key)
        log.info("Tavily search enabled")
    else:
        log.warning("TAVILY_API_KEY not found. Search functionality disabled.")


@tool
def search_web(query: str) -> str:
    """Search the web for current information using Tavily.
    Use this when you need up-to-date information, documentation, or to verify facts.
    
    Args:
        query: The search query to look up
        
    Returns:
        Search results with titles, URLs and content snippets
    """
    if not _tavily_client:
        return "Search not available. Tavily API key not configured."
    
    try:
        log.info("Tool call: search_web", query=query)
        results = _tavily_client.search(query, max_results=5)
        
        if not results or 'results' not in results:
            return "No results found."
        
        # Format results
        formatted = []
        for i, result in enumerate(results['results'], 1):
            formatted.append(
                f"{i}. {result.get('title', 'Untitled')}\n"
                f"   URL: {result.get('url', 'N/A')}\n"
                f"   {result.get('content', 'No content')}"
            )
        
        output = "\n\n".join(formatted)
        log.info("Search completed", results_count=len(results['results']))
        return output
    except Exception as e:
        log.error("Search error", error=str(e))
        return f"Search error: {str(e)}"


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


def generate_code(prompt: str, model: str, temperature: float, use_search: bool = False) -> tuple[str, list]:
    """
    Generate code or explanations based on a prompt.
    
    Args:
        prompt: The user's request
        model: Model name to use
        temperature: Generation temperature
        use_search: Whether to enable web search tool for the AI
    
    Returns:
        Tuple of (generated response, list of tool calls made)
    """
    log.info("Generating code", model=model, prompt_length=len(prompt), search_enabled=use_search)
    
    llm = get_model(model, temperature)
    tool_calls = []
    
    # Add current date context
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_context = f"Current date: {current_date}\n\n"
    
    # If search is enabled and it's an OpenAI model (supports tool calling)
    if use_search and model.startswith(('gpt-', 'o1-', 'o3-')) and _tavily_client:
        # Bind the search tool to the model
        llm_with_tools = llm.bind_tools([search_web])
        
        # First call - let AI decide if it needs to search
        response = llm_with_tools.invoke([HumanMessage(content=system_context + prompt)])
        
        # Check if AI wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages = [HumanMessage(content=system_context + prompt), response]
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "tool": tool_call['name'],
                    "args": tool_call['args']
                })
                
                # Execute the tool
                tool_output = search_web.invoke(tool_call['args'])
                messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                ))
            
            # Get final response with tool results
            final_response = llm_with_tools.invoke(messages)
            response_text = final_response.content
        else:
            response_text = response.content
    else:
        # No tool calling support or not enabled
        response = llm.invoke(system_context + prompt)
        response_text = response.content if hasattr(response, 'content') else response
    
    log.info("Code generated", response_length=len(response_text), tools_used=len(tool_calls))
    return response_text, tool_calls


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
    # Extract content from AIMessage if it's an OpenAI model
    if hasattr(response, 'content'):
        response = response.content
    log.info("Code completed", response_length=len(response))
    return response


def chat(prompt: str, model: str, temperature: float, use_search: bool = False) -> tuple[str, list]:
    """
    Handle chat-style coding questions and explanations.
    
    Args:
        prompt: User's question or request
        model: Model name to use
        temperature: Generation temperature
        use_search: Whether to enable web search tool for the AI
    
    Returns:
        Tuple of (response, list of tool calls made)
    """
    log.info("Processing chat request", model=model, prompt_length=len(prompt), search_enabled=use_search)
    
    llm = get_model(model, temperature)
    tool_calls = []
    
    # Add current date context
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_context = f"Current date: {current_date}\n\n"
    
    # If search is enabled and it's an OpenAI model (supports tool calling)
    if use_search and model.startswith(('gpt-', 'o1-', 'o3-')) and _tavily_client:
        # Bind the search tool to the model
        llm_with_tools = llm.bind_tools([search_web])
        
        # First call - let AI decide if it needs to search
        response = llm_with_tools.invoke([HumanMessage(content=system_context + prompt)])
        
        # Check if AI wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages = [HumanMessage(content=system_context + prompt), response]
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "tool": tool_call['name'],
                    "args": tool_call['args']
                })
                
                # Execute the tool
                tool_output = search_web.invoke(tool_call['args'])
                messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                ))
            
            # Get final response with tool results
            final_response = llm_with_tools.invoke(messages)
            response_text = final_response.content
        else:
            response_text = response.content
    else:
        # No tool calling support or not enabled
        response = llm.invoke(system_context + prompt)
        response_text = response.content if hasattr(response, 'content') else response
    
    log.info("Chat response generated", response_length=len(response_text), tools_used=len(tool_calls))
    return response_text, tool_calls


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


def structured_output(prompt: str, schema: dict, model: str, temperature: float, use_search: bool = False) -> tuple[dict, list]:
    """
    Get structured output based on a Pydantic schema.
    
    Args:
        prompt: The user's request
        schema: Pydantic schema as dictionary
        model: Model name to use (must be OpenAI)
        temperature: Generation temperature
        use_search: Whether to enable web search tool
    
    Returns:
        Tuple of (parsed structured data, list of tool calls made)
    """
    log.info("Structured output request", model=model, search_enabled=use_search)
    
    # Only OpenAI models support structured output
    if not model.startswith(('gpt-', 'o1-', 'o3-')):
        raise ValueError("Structured output only supported with OpenAI models")
    
    from pydantic import create_model
    from datetime import datetime
    
    # Create Pydantic model from schema
    model_name = schema.get('title', 'DynamicModel')
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    # Build field definitions
    fields = {}
    for field_name, field_schema in properties.items():
        field_type = Any
        if field_schema.get('type') == 'string':
            field_type = str
        elif field_schema.get('type') == 'integer':
            field_type = int
        elif field_schema.get('type') == 'number':
            field_type = float
        elif field_schema.get('type') == 'boolean':
            field_type = bool
        elif field_schema.get('type') == 'array':
            field_type = list
        elif field_schema.get('type') == 'object':
            field_type = dict
        
        # Make optional if not in required
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type, None)
    
    DynamicModel = create_model(model_name, **fields)
    
    # Get LLM and bind schema
    llm = get_model(model, temperature)
    structured_llm = llm.with_structured_output(DynamicModel)
    
    # Add current date context
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_context = f"Current date: {current_date}\\n\\n"
    
    tool_calls = []
    
    # If search is enabled, bind tools
    if use_search and _tavily_client:
        # For structured output with tools, we need to handle it differently
        llm_with_tools = llm.bind_tools([search_web])
        
        # First call - let AI decide if it needs to search
        response = llm_with_tools.invoke([HumanMessage(content=system_context + prompt)])
        
        # Check if AI wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages = [HumanMessage(content=system_context + prompt), response]
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "tool": tool_call['name'],
                    "args": tool_call['args']
                })
                
                # Execute the tool
                tool_output = search_web.invoke(tool_call['args'])
                messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                ))
            
            # Add instruction to use gathered info
            messages.append(HumanMessage(content="Based on the search results above, " + prompt))
            
            # Get structured response
            structured_response = structured_llm.invoke(messages)
        else:
            structured_response = structured_llm.invoke([HumanMessage(content=system_context + prompt)])
    else:
        structured_response = structured_llm.invoke(system_context + prompt)
    
    # Convert to dict
    result = structured_response.model_dump() if hasattr(structured_response, 'model_dump') else structured_response.dict()
    
    log.info("Structured output completed", tools_used=len(tool_calls))
    return result, tool_calls


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
