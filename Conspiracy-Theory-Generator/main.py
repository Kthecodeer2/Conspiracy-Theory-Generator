import os
import sys
import asyncio
from typing import List

# Needed for streaming token events
from openai.types.responses import ResponseTextDeltaEvent

try:
    from agents import Agent, Runner, WebSearchTool, ItemHelpers, ModelSettings, function_tool
except ImportError:
    raise SystemExit("The 'openai-agents' package is required. Install it with 'pip install openai-agents duckduckgo-search'.")

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise SystemExit("The 'duckduckgo-search' package is required. Install it with 'pip install duckduckgo-search'.")

try:
    import requests
except ImportError:
    raise SystemExit("The 'requests' package is required. Install it with 'pip install requests'.")


@function_tool
def web_search(query: str, max_results: int = 5) -> List[str]:
    """Perform a DuckDuckGo search and return up to `max_results` concise snippets.

    The helper first tries the DDGS() context-manager interface. If that fails
    (e.g., because of rate-limits or networking issues) it falls back to the
    simpler `duckduckgo_search.ddg` helper.  Any runtime problems are
    propagated so that the calling agent is aware a search really failed.
    """

    from duckduckgo_search import ddg  # imported here to avoid import cycles

    snippets: List[str] = []

    def _truncate(text: str, limit: int = 200) -> str:
        return text[:limit]

    try:
        # Preferred, gives richer metadata.
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                title = result.get("title", "")
                body = result.get("body", "")
                snippet = f"{title}: {body}"
                snippets.append(_truncate(snippet))

        # If DDGS returned nothing, try the lightweight fallback.
        if not snippets:
            raise ValueError("DDGS returned no results; falling back to ddg helper.")

    except Exception:
        # Fallback strategy using the simpler helper which is sometimes more reliable.
        try:
            results = ddg(query, max_results=max_results) or []
            for res in results:
                title = res.get("title", "")
                body = res.get("body", "") or res.get("snippet", "")
                snippet = f"{title}: {body}"
                snippets.append(_truncate(snippet))
        except Exception as e:
            # Bubble up an explicit error so the calling agent can react.
            raise RuntimeError(f"web_search failed: {e}")

    # Ensure we respect the requested max_results, even after fallback.
    return snippets[:max_results]


# NOTE: The @function_tool decorator wraps the function in a non-callable
# FunctionTool object. We expose a thin internal wrapper `_verify_url` for
# regular Python calls, and keep the wrapped version (`verify_url_tool`) for
# the agent.


def _verify_url_impl(url: str, timeout_seconds: int = 5) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout_seconds)
        if response.status_code < 400:
            return True
        if response.status_code in (405, 501):
            response = requests.get(url, allow_redirects=True, timeout=timeout_seconds, stream=True)
            return response.status_code < 400
        return False
    except Exception:
        return False


# Expose as tool for the agent
@function_tool
def verify_url_tool(url: str, timeout_seconds: int = 5) -> bool:
    """Agent-facing wrapper around `_verify_url_impl`."""
    return _verify_url_impl(url, timeout_seconds)


# -----------------------------------------------------------------------------
# Helper tool: search + verify in one call
# -----------------------------------------------------------------------------


@function_tool
def search_verified_links(query: str, max_results: int = 5) -> List[str]:
    """Return up to `max_results` verified URLs relevant to `query`.

    1. Perform a DuckDuckGo search for the query (via the local ddg helper).
    2. For each result, call `verify_url` to ensure the link is live (HTTP < 400).
    3. Return a list of URLs that passed verification.
    """

    from duckduckgo_search import ddg

    verified: List[str] = []
    try:
        results = ddg(query, max_results=max_results * 4) or []  # fetch more in case some fail
        for res in results:
            url = res.get("href") or res.get("url") or ""
            if url and _verify_url_impl(url):
                verified.append(url)
            if len(verified) >= max_results:
                break
    except Exception:
        pass

    return verified[:max_results]


# Define the agent responsible for generating conspiracies.
conspiracy_agent = Agent(
    name="Conspiracy Theorist",
    instructions=(
        "Invent fresh, original conspiracy ideas—concepts that are novel yet grounded enough that some portion of them can be corroborated by real-world evidence. "
        "You are 'The Conspiracy Theorist', an AI that crafts elaborate conspiracy narratives. "
        "For EVERY factual statement or claim you make, you MUST immediately supply at least one piece of supporting evidence. "
        "Evidence MUST be presented as a full, direct URL (including https://) that points to a publicly available source such as a news article, court document, academic paper, interview transcript, or similar record. "
        "Always begin by calling `search_verified_links` with a concise query about the statement you're supporting. Use the returned list of verified URLs as evidence. If additional links are needed, call `WebSearchTool` for more candidates and check each with `verify_url`. Never include a URL you haven't verified. If no verified link is available, omit the claim. "
        "Output structure: write short explanatory paragraphs, and after each paragraph add a new line that begins with 'Evidence:' followed by the list of URLs used in that paragraph. "
        "Be creative, engaging, and sly, but remain grounded in the verifiable information you cite."
        "Make it something that is not already known, and make it something that is not already out there. "
        "Only make things you can prove, and make it something that is not already out there. "
        "Dont make theories that are very obviously fake "
        "Gold Mines for some proof for a theory are Declassified documents, and court cases. "
    ),
    tools=[WebSearchTool(), search_verified_links, verify_url_tool],
    model_settings=ModelSettings(tool_choice="required"),
)


def generate_conspiracy(topic: str) -> str:
    """Synchronously generate a conspiracy theory about `topic`."""
    result = Runner.run_sync(conspiracy_agent, topic)
    return result.final_output


async def main_async(topic: str):
    """Stream the agent's output, then verify and correct links.

    We stream tokens live for user feedback but also buffer everything so we can
    scrub dead links afterward, printing a corrected version if needed.
    """

    import re

    result_stream = Runner.run_streamed(conspiracy_agent, input=topic)

    buffer: list[str] = []

    async for event in result_stream.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            token = event.data.delta
            print(token, end="", flush=True)
            buffer.append(token)
            continue

        if event.type == "run_item_stream_event" and event.item.type == "message_output_item":
            chunk = ItemHelpers.text_message_output(event.item)
            print(chunk, end="", flush=True)
            buffer.append(chunk)

    # newline after stream ends
    print()

    full_output = "".join(buffer)

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    replacements: dict[str, str] = {}
    for match in link_pattern.finditer(full_output):
        url = match.group(2)
        if not _verify_url_impl(url):
            replacements[match.group(0)] = "[INVALID LINK REMOVED]"

    if replacements:
        corrected = full_output
        for original, replacement in replacements.items():
            corrected = corrected.replace(original, replacement)

        print("\n---\nThe following links were removed after verification failure:")
        for faulty in replacements.keys():
            print(f"- {faulty}")

        print("\nCorrected output (with invalid links removed):\n")
        print(corrected.strip())


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