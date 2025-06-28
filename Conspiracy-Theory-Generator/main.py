import os
import sys
import asyncio
from typing import List

try:
    from agents import Agent, Runner, function_tool
except ImportError:
    raise SystemExit("The 'openai-agents' package is required. Install it with 'pip install openai-agents duckduckgo-search'.")

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise SystemExit("The 'duckduckgo-search' package is required. Install it with 'pip install duckduckgo-search'.")


@function_tool
def web_search(query: str, max_results: int = 5) -> List[str]:
    """Perform a web search and return up to `max_results` short snippets.

    The search intentionally uses DuckDuckGo to avoid tracking and filters
    explicit content by default. 
    """
    snippets: List[str] = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            title = result.get("title", "")
            body = result.get("body", "")
            snippet = f"{title}: {body}"
            # Truncate overly long snippets to keep token usage low
            snippets.append(snippet[:200])
    return snippets


# Define the agent responsible for generating conspiracies ethically.
conspiracy_agent = Agent(
    name="Conspiracy Theorist",
    instructions=(
        "You are an intelligent AI that invents, "
        "Make sure to give tons and i mean alot of evidence, Websites, Laws, court cases, interviews, word plays, silenced people, freedom fighters, etc."
        "conspiracy theories. You may call the `web_search` "
        "tool to collect publicly available information that you weave into the "
        "conspiracy narrative. You may also call the `web_search` tool to "
        "collect publicly available information that you weave into the "
        "conspiracy narrative. Make sure to have real information in your theory while also being cunning and smart, and make real and worth it"
    ),
    tools=[web_search],
    
    
)


def generate_conspiracy(topic: str) -> str:
    """Synchronously generate a conspiracy theory about `topic`."""
    result = Runner.run_sync(conspiracy_agent, topic)
    return result.final_output


async def main_async(topic: str):
    result = await Runner.run(conspiracy_agent, topic)
    print("\n" + result.final_output.strip())


def main():
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit(
            "Error: Please set the OPENAI_API_KEY environment variable before running."
        )

    if len(sys.argv) < 2:
        print("Usage: python main.py <topic>")
        sys.exit(1)

    topic = " ".join(sys.argv[1:]).strip()

    try:
        # Run asynchronously for non-blocking tool calls
        asyncio.run(main_async(topic))
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")


if __name__ == "__main__":
    main() 