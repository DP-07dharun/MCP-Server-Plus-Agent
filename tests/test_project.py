"""Tests that need no API key.  Run:  python -m unittest -v"""

from __future__ import annotations

import asyncio
import csv
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "agent")]

import finance_server as fs  # noqa: E402
from google.genai import types  # noqa: E402
from mcp import Client  # noqa: E402

import agent as ag  # noqa: E402


class ToolTests(unittest.TestCase):
    def test_schema(self):
        schema = fs.get_schema()
        self.assertGreater(schema["row_count"], 5000)
        self.assertIn("Paris", schema["values"]["city"])

    def test_monthly_summary_in_calendar_order(self):
        months = [r["month"] for r in fs.spending_summary("month", top_n=12)["rows"]]
        self.assertEqual(months, sorted(months))
        self.assertEqual(len(months), 12)

    def test_summary_filters(self):
        r = fs.spending_summary("city", category="Travel", start_date="2025-07-01", end_date="2025-07-31")
        total = fs.query_transactions(
            "SELECT sum(transaction_amount) s FROM transactions WHERE category='Travel' "
            "AND date >= '2025-07-01' AND date < '2025-08-01'")["rows"][0]["s"]
        self.assertAlmostEqual(sum(x["total_eur"] for x in r["rows"]), total, delta=0.1)

    def test_unknown_user(self):
        self.assertIn("error", fs.user_profile("U9999"))


class SqlSafetyTests(unittest.TestCase):
    def test_rejects_writes_and_multiple_statements(self):
        for sql in ("DROP TABLE transactions", "DELETE FROM transactions",
                    "SET enable_external_access = true", "SELECT 1; SELECT 2"):
            with self.subTest(sql=sql):
                self.assertIn("error", fs.query_transactions(sql))
        self.assertEqual(fs.query_transactions("SELECT count(*) n FROM transactions")["rows"][0]["n"],
                         fs.get_schema()["row_count"])

    def test_no_file_access_from_select(self):
        out = fs.query_transactions(f"SELECT * FROM read_csv('{ROOT / 'data' / 'anomalies_truth.csv'}')")
        self.assertIn("error", out)

    def test_row_cap(self):
        out = fs.query_transactions("SELECT * FROM transactions")
        self.assertEqual(out["row_count"], fs.MAX_ROWS)
        self.assertTrue(out["truncated"])


class DetectionTests(unittest.TestCase):
    """Score the fraud rules against the labels the generator wrote (never shown to the agent)."""

    @classmethod
    def setUpClass(cls):
        with (ROOT / "data" / "anomalies_truth.csv").open(encoding="utf-8") as fh:
            cls.truth = {r["transaction_id"]: r["scenario"].split("#")[0] for r in csv.DictReader(fh)}
        cls.flagged = {r["transaction_id"] for r in fs.find_suspicious_transactions(limit=fs.MAX_ROWS)["rows"]}

    def test_recall_per_scenario(self):
        for scenario in sorted(set(self.truth.values())):
            ids = {t for t, s in self.truth.items() if s == scenario}
            with self.subTest(scenario=scenario):
                self.assertGreaterEqual(len(ids & self.flagged) / len(ids), 0.9)

    def test_precision(self):
        self.assertGreaterEqual(len(self.flagged & set(self.truth)) / len(self.flagged), 0.75)


class McpProtocolTests(unittest.TestCase):
    def test_tools_over_mcp(self):
        async def run():
            async with Client(fs.mcp) as client:  # in-process transport
                names = {t.name for t in (await client.list_tools()).tools}
                result = await client.call_tool("spending_summary", {"group_by": "category", "top_n": 3})
                return names, result
        names, result = asyncio.run(run())
        self.assertEqual(names, {"get_schema", "spending_summary", "user_profile",
                                 "find_suspicious_transactions", "query_transactions"})
        self.assertFalse(result.is_error)
        self.assertEqual(len(ag.tool_output(result)["result"]["rows"]), 3)


class AgentLoopTests(unittest.TestCase):
    """Drive the agent loop with a scripted fake Gemini: one tool call, then an answer."""

    def test_tool_call_then_answer(self):
        replies = [
            SimpleNamespace(
                candidates=[SimpleNamespace(content=types.Content(role="model", parts=[
                    types.Part.from_function_call(name="user_profile", args={"user_id": "U0042"})]))],
                function_calls=[types.FunctionCall(name="user_profile", args={"user_id": "U0042"})], text=None),
            SimpleNamespace(
                candidates=[SimpleNamespace(content=types.Content(role="model", parts=[
                    types.Part.from_text(text="U0042 lives in Strasbourg.")]))],
                function_calls=None, text="U0042 lives in Strasbourg."),
        ]

        async def fake_generate(**kwargs):
            return replies.pop(0)

        async def run():
            async with Client(fs.mcp) as client:
                agent = ag.FinanceAgent.__new__(ag.FinanceAgent)
                agent.mcp, agent.model, agent.verbose, agent.history = client, "fake", False, []
                agent.llm = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate)))
                await agent.setup()
                return await agent.ask("Where does U0042 live?"), agent.history

        answer, history = asyncio.run(run())
        self.assertEqual(answer, "U0042 lives in Strasbourg.")
        tool_reply = history[2].parts[0].function_response
        self.assertEqual(tool_reply.response["result"]["home_city"], "Strasbourg")

    def test_mcp_schemas_convert_to_gemini(self):
        async def run():
            async with Client(fs.mcp) as client:
                return ag.to_gemini_tools((await client.list_tools()).tools)
        decls = asyncio.run(run())[0].function_declarations
        self.assertEqual(len(decls), 5)
        self.assertTrue(all(d.parameters_json_schema is not None for d in decls))


if __name__ == "__main__":
    unittest.main()
