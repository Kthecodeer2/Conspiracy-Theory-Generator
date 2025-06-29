from __future__ import annotations

# NOTE: this is a NEW file – a lightweight Flask frontend that streams
# conspiracy-theory output to the browser in real-time.

import importlib.util
import pathlib
import textwrap
from typing import AsyncGenerator

from flask import Flask, Response, render_template, request
from openai.types.responses import ResponseTextDeltaEvent

# ---------------------------------------------------------------------------
# Dynamically load the existing `main.py` (which lives in the nested directory
# with a hyphen in its name, making it an invalid Python identifier).
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CTG_MAIN_PATH = PROJECT_ROOT / "Conspiracy-Theory-Generator" / "main.py"

_spec = importlib.util.spec_from_file_location("ctg_main", CTG_MAIN_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover – safety check
    raise ImportError(f"Unable to locate main.py at {CTG_MAIN_PATH}")

ctg_main = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(ctg_main)  # type: ignore[union-attr]

# Re-export the pieces we need from the dynamically-loaded module.
conspiracy_agent = ctg_main.conspiracy_agent
Runner = ctg_main.Runner
ItemHelpers = ctg_main.ItemHelpers

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index() -> str:  # noqa: D401 – simple wrapper
    """Render the landing page."""
    return render_template("index.html")


@app.route("/stream")
def stream() -> Response:
    """SSE endpoint that **synchronously** streams text to the browser.

    Flask's development WSGI server can't directly iterate over *async* generators,
    which caused the previous *TypeError: 'async_generator' object is not
    iterable*.  We bridge the async world (the agent) and the sync WSGI world by
    running the generation coroutine inside a **background thread** and pushing
    chunks onto a ``queue.Queue``.  The synchronous generator that Flask
    iterates over pops items from that queue and yields them as Server-Sent
    Events in real time.
    """

    import asyncio
    import queue
    import threading

    topic = (request.args.get("topic") or "").strip()
    if not topic:
        return Response("Topic query parameter 'topic' is required", status=400)

    q: "queue.Queue[str | None]" = queue.Queue()

    async def _produce() -> None:
        """Async producer that puts SSE data strings onto the queue."""

        result_stream = Runner.run_streamed(conspiracy_agent, input=topic)
        async for event in result_stream.stream_events():
            if (
                event.type == "raw_response_event"
                and isinstance(event.data, ResponseTextDeltaEvent)
            ):
                token = event.data.delta
                q.put(f"data: {token}\n\n")
            elif (
                event.type == "run_item_stream_event"
                and event.item.type == "message_output_item"
            ):
                chunk = ItemHelpers.text_message_output(event.item)
                q.put(f"data: {chunk}\n\n")

        # Signal completion
        q.put("data: [DONE]\n\n")
        q.put(None)  # sentinel for the consumer

    # Kick-off the async producer inside a background thread so it doesn't block
    threading.Thread(target=lambda: asyncio.run(_produce()), daemon=True).start()

    def _consume():
        """Synchronous generator that yields SSE strings to Flask/Werkzeug."""

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return Response(_consume(), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# Helpful CLI runner – optional, so users can ``python app.py`` directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    banner = textwrap.dedent(
        """
        =====================================================
          Conspiracy-Theory Generator – Web Frontend
        -----------------------------------------------------
        Open your browser at http://127.0.0.1:5000/ and enjoy!
        =====================================================
        """
    )
    print(banner)
    # ``threaded=True`` lets Flask keep accepting new connections while the
    # asyncio event-loop inside each request does its work.
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True) 