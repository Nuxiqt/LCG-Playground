from dotenv import load_dotenv
import uvicorn
from logger import configure_logging, log
import asyncio
from agents import Agent, Runner, function_tool, set_trace_processors
from langsmith.wrappers import OpenAIAgentsTracingProcessor

load_dotenv()
configure_logging()


def main():
    """Start the FastAPI server."""
    log.info("Starting Local Code LLM API")
    log.info("API documentation available", url="http://localhost:8000/docs")
    log.info("Press CTRL+C to stop")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,  # Disable uvicorn's default logging config
    )


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())
