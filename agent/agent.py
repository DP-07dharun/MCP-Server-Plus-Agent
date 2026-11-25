"""Gemini-powered agent that answers questions about bank transactions through the MCP server.

    python agent/agent.py                         # interactive chat
    python agent/agent.py "Which city spends the most on travel?"

The agent starts server/finance_server.py as a subprocess, discovers its tools over MCP,
hands them to Gemini as function declarations and runs the call-tool / feed-result loop
until Gemini answers in plain text.

Needs GEMINI_API_KEY (in the environment or in a .env file at the repo root).
GEMINI_MODEL picks the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server" / "finance_server.py"
MAX_STEPS = 12  # tool-call rounds per question, a guard against loops

SYSTEM = """You are a financial data analyst assistant for a French retail bank.
Answer using the tools only: never invent figures. Amounts are in EUR.
Call get_schema first if you are unsure of column names or allowed values.
When you report suspicious transactions, give the transaction IDs, the rules they matched
and the risk score, and say clearly that they are leads for review, not proven fraud.
Keep answers short: a sentence or two, then a compact table or list when useful."""

DIM, CYAN, RESET = "\033[2m", "\033[36m", "\033[0m"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines) so we don't need python-dotenv."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def to_gemini_tools(mcp_tools: list[Any]) -> list[types.Tool]:
    """MCP tool list -> one Gemini Tool with a function declaration per MCP tool."""
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t.name, description=t.description or "",
                                  parameters_json_schema=t.input_schema)
        for t in mcp_tools
    ])]


def tool_output(result: Any) -> dict[str, Any]:
    """CallToolResult -> plain dict Gemini can read."""
    if result.structured_content is not None:
        payload: Any = result.structured_content
    else:
        payload = "\n".join(getattr(c, "text", "") for c in result.content)
    return {"error": payload} if result.is_error else {"result": payload}


class FinanceAgent:
    def __init__(self, mcp: Client, model: str, verbose: bool = True):
        self.mcp, self.model, self.verbose = mcp, model, verbose
        self.llm = genai.Client()  # reads GEMINI_API_KEY
        self.history: list[types.Content] = []
        self.config: types.GenerateContentConfig | None = None

    async def setup(self) -> list[str]:
        tools = (await self.mcp.list_tools()).tools
        # The server's own MCP instructions (e.g. which year the data covers) go into the prompt.
        server_notes = self.mcp.instructions or ""
        self.config = types.GenerateContentConfig(
            system_instruction=f"{SYSTEM}\n\nNotes from the data server:\n{server_notes}".strip(),
            tools=to_gemini_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        return [t.name for t in tools]

    async def ask(self, question: str) -> str:
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        for _ in range(MAX_STEPS):
            response = await self.llm.aio.models.generate_content(
                model=self.model, contents=self.history, config=self.config)
            if not response.candidates or not response.candidates[0].content:
                return "(the model returned no answer, try rephrasing)"
            self.history.append(response.candidates[0].content)
            calls = response.function_calls or []
            if not calls:
                return response.text or ""
            parts = []
            for call in calls:
                args = dict(call.args or {})
                if self.verbose:
                    print(f"{DIM}  -> {call.name}({json.dumps(args, ensure_ascii=False)}){RESET}")
                result = await self.mcp.call_tool(call.name, args)
                parts.append(types.Part.from_function_response(name=call.name, response=tool_output(result)))
            self.history.append(types.Content(role="user", parts=parts))
        return "(stopped: too many tool calls for one question)"


def explain_model_error(llm: genai.Client, model: str, exc: errors.APIError) -> str:
    msg = f"Gemini API error {exc.code}: {exc.message}"
    if exc.code == 404:
        try:
            names = [m.name.removeprefix("models/") for m in llm.models.list()
                     if "generateContent" in (m.supported_actions or []) and "flash" in m.name]
            msg += f"\nModel {model!r} is not available. Set GEMINI_MODEL to one of: {', '.join(names)}"
        except errors.APIError:
            pass
    return msg


async def main() -> int:
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and put\n"
              "GEMINI_API_KEY=your-key in the .env file at the repo root (see .env.example).")
        return 1
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(ROOT))
    async with Client(server) as mcp:
        agent = FinanceAgent(mcp, model)
        tools = await agent.setup()
        print(f"{CYAN}MCP Server + Agent | tools: {', '.join(tools)}  |  model: {model}{RESET}")
        one_shot = " ".join(sys.argv[1:]).strip()
        questions = [one_shot] if one_shot else None
        while True:
            if questions is not None:
                if not questions:
                    return 0
                question = questions.pop()
                print(f"> {question}")
            else:
                try:
                    question = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    return 0
                if question.lower() in {"exit", "quit"}:
                    return 0
                if not question:
                    continue
            try:
                print(await agent.ask(question))
            except errors.APIError as exc:
                print(explain_model_error(agent.llm, model, exc))
                agent.history.clear()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
