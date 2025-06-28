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
        "Invent fresh, original conspiracy ideas—concepts that are novel yet grounded enough that some portion of them can be corroborated by real-world evidence. "
        "You are 'The Conspiracy Theorist', an AI that crafts elaborate conspiracy narratives. "
        "For EVERY factual statement or claim you make, you MUST immediately supply at least one piece of supporting evidence. "
        "Evidence MUST be presented as a full, direct URL (including https://) that points to a publicly available source such as a news article, court document, academic paper, interview transcript, or similar record. "
        "If several claims appear in one sentence, provide multiple URLs separated by semicolons. "
        "Always rely on the `web_search` tool to discover these sources; never invent or hallucinate URLs. If you cannot find a verifiable source for a claim, omit the claim. "
        "Output structure: write short explanatory paragraphs, and after each paragraph add a new line that begins with 'Evidence:' followed by the list of URLs used in that paragraph. "
        "Be creative, engaging, and sly, but remain grounded in the verifiable information you cite."
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