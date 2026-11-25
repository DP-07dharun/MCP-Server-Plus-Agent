"""MCP server exposing a bank-transactions dataset to LLM agents.

    python server/finance_server.py            # stdio transport (Claude Desktop, Claude Code, the agent)

The data is loaded into an in-memory DuckDB database. After loading, file and network
access is switched off and the configuration is locked, so even the free-form SQL tool
cannot read other files on the machine or change settings.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import duckdb
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FINANCE_DATA", ROOT / "data" / "transactions.csv"))
MAX_ROWS = 200

CITY_COORDS = {  # (lat, lon), for the impossible-travel check
    "Paris": (48.86, 2.35), "Lyon": (45.76, 4.84), "Marseille": (43.30, 5.37),
    "Toulouse": (43.60, 1.44), "Nice": (43.70, 7.27), "Nantes": (47.22, -1.55),
    "Lille": (50.63, 3.06), "Bordeaux": (44.84, -0.58), "Strasbourg": (48.57, 7.75),
    "Rouen": (49.44, 1.10),
}

RULES = {
    # rule -> (risk weight, explanation shown to the agent)
    "amount_spike": (40, "amount is more than 15x the median for its category"),
    "card_testing": (35, "one of 4+ charges under 2 EUR on the same account within 10 minutes"),
    "night_activity": (25, "made between 00:00 and 05:59, outside the user's usual hours"),
    "away_from_home": (10, "made outside the user's home city (their most frequent city)"),
    "impossible_travel": (40, "previous payment of the same user was in a city that could not be reached in time, even by plane (>500 km/h)"),
}
Rule = Literal["amount_spike", "card_testing", "night_activity", "away_from_home", "impossible_travel"]
GroupBy = Literal["category", "city", "third_party", "payment_method", "month", "user_id", "financial_institution_id"]


def connect(csv_path: Path = DATA) -> duckdb.DuckDBPyConnection:
    """Load the CSV into a locked-down in-memory database."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found: run `python data/generate_transactions.py` first")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE transactions AS
        SELECT user_id, transaction_id, bank_account_id,
               CAST(financial_institution_id AS INTEGER) AS financial_institution_id,
               CAST(date AS TIMESTAMP) AS date, third_party, category,
               CAST(transaction_amount AS DECIMAL(12, 2)) AS transaction_amount,
               city, payment_method, currency
        FROM read_csv(?, header = true, all_varchar = true)
        """,
        [str(csv_path)],
    )
    con.execute("CREATE TABLE city_coords(city VARCHAR, lat DOUBLE, lon DOUBLE)")
    con.executemany("INSERT INTO city_coords VALUES (?, ?, ?)", [(c, *xy) for c, xy in CITY_COORDS.items()])
    con.execute("SET enable_external_access = false")  # no reading/writing files or URLs from SQL
    con.execute("SET lock_configuration = true")  # and no way to switch it back on
    return con


DB = connect()


def rows(sql: str, params: list[Any] | None = None, limit: int = MAX_ROWS) -> dict[str, Any]:
    """Run a query and return JSON-friendly columns + rows."""
    cur = DB.cursor()
    cur.execute(sql, params or [])
    columns = [d[0] for d in cur.description]
    fetched = cur.fetchmany(limit + 1)
    data = [{c: _jsonable(v) for c, v in zip(columns, r)} for r in fetched[:limit]]
    return {"columns": columns, "rows": data, "row_count": len(data), "truncated": len(fetched) > limit}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat(sep=" ") if hasattr(v, "hour") else v.isoformat()
    return float(v) if v.__class__.__name__ == "Decimal" else str(v)


def _filters(start_date: date | None, end_date: date | None, user_id: str | None = None,
             category: str | None = None, city: str | None = None) -> tuple[str, list[Any]]:
    clauses, params = ["TRUE"], []
    if start_date:
        clauses.append("date >= ?"); params.append(start_date)
    if end_date:
        clauses.append("date < CAST(? AS DATE) + INTERVAL 1 DAY"); params.append(end_date)
    for col, val in (("user_id", user_id), ("category", category), ("city", city)):
        if val:
            clauses.append(f"{col} = ?"); params.append(val)
    return " AND ".join(clauses), params


mcp = MCPServer(
    "finance-transactions",
    title="MCP Server + Agent - finance data",
    instructions=(
        "Tools over a year (2025) of synthetic French bank transactions in EUR. "
        "Start with get_schema. Prefer the specialised tools; use query_transactions for anything else. "
        "Suspicious transactions are flags for review, not proof of fraud."
    ),
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


@mcp.tool(annotations=READ_ONLY)
def get_schema() -> dict[str, Any]:
    """Describe the transactions table: columns, row count, date range and the allowed values
    of the categorical columns. Call this first."""
    cur = DB.cursor()
    columns = [{"name": n, "type": t} for n, t, *_ in cur.execute("DESCRIBE transactions").fetchall()]
    n, first, last, total = cur.execute(
        "SELECT count(*), min(date), max(date), sum(transaction_amount) FROM transactions").fetchone()
    distinct = {
        col: [r[0] for r in cur.execute(f"SELECT DISTINCT {col} FROM transactions ORDER BY 1").fetchall()]
        for col in ("category", "city", "payment_method", "financial_institution_id")
    }
    users = cur.execute("SELECT count(DISTINCT user_id) FROM transactions").fetchone()[0]
    return {
        "table": "transactions",
        "columns": columns,
        "row_count": n,
        "distinct_users": users,
        "date_range": [first.isoformat(sep=" "), last.isoformat(sep=" ")],
        "total_amount_eur": float(total),
        "values": distinct,
    }


@mcp.tool(annotations=READ_ONLY)
def spending_summary(
    group_by: GroupBy,
    start_date: date | None = None,
    end_date: date | None = None,
    user_id: str | None = None,
    category: str | None = None,
    city: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Total amount, transaction count and average amount grouped by one column, highest total
    first. Optional filters: inclusive date range (YYYY-MM-DD), user_id, category, city.
    group_by='month' gives a monthly trend in calendar order."""
    where, params = _filters(start_date, end_date, user_id, category, city)
    key = "strftime(date, '%Y-%m')" if group_by == "month" else group_by
    order = "1" if group_by == "month" else "total_eur DESC"
    top_n = max(1, min(top_n, MAX_ROWS))
    return rows(
        f"""SELECT {key} AS {group_by}, round(sum(transaction_amount), 2) AS total_eur,
                   count(*) AS transactions, round(avg(transaction_amount), 2) AS avg_eur
            FROM transactions WHERE {where} GROUP BY 1 ORDER BY {order} LIMIT {top_n}""",
        params,
    )


@mcp.tool(annotations=READ_ONLY)
def user_profile(user_id: str) -> dict[str, Any]:
    """Profile of one customer (e.g. 'U0042'): totals, home city, bank accounts, favourite
    categories and their 5 most recent transactions."""
    cur = DB.cursor()
    summary = rows(
        """SELECT count(*) AS transactions, round(sum(transaction_amount), 2) AS total_eur,
                  round(avg(transaction_amount), 2) AS avg_eur, round(max(transaction_amount), 2) AS max_eur,
                  min(date) AS first_seen, max(date) AS last_seen, mode(city) AS home_city
           FROM transactions WHERE user_id = ?""", [user_id])["rows"][0]
    if not summary["transactions"]:
        return {"error": f"unknown user_id {user_id!r}"}
    accounts = cur.execute(
        "SELECT DISTINCT bank_account_id, financial_institution_id FROM transactions WHERE user_id = ? ORDER BY 1",
        [user_id]).fetchall()
    return {
        "user_id": user_id,
        **summary,
        "accounts": [{"bank_account_id": a, "financial_institution_id": i} for a, i in accounts],
        "top_categories": rows(
            """SELECT category, round(sum(transaction_amount), 2) AS total_eur, count(*) AS transactions
               FROM transactions WHERE user_id = ? GROUP BY 1 ORDER BY 2 DESC LIMIT 5""", [user_id])["rows"],
        "recent_transactions": rows(
            """SELECT transaction_id, date, third_party, category, transaction_amount, city, payment_method
               FROM transactions WHERE user_id = ? ORDER BY date DESC LIMIT 5""", [user_id])["rows"],
    }


SUSPICIOUS_SQL = """
WITH base AS (
    SELECT t.*,
           mode(city) OVER (PARTITION BY user_id) AS home_city,
           median(transaction_amount) OVER (PARTITION BY category) AS category_median,
           count(*) FILTER (WHERE transaction_amount < 2) OVER (
               PARTITION BY bank_account_id ORDER BY date
               RANGE BETWEEN INTERVAL 10 MINUTE PRECEDING AND INTERVAL 10 MINUTE FOLLOWING) AS tiny_nearby,
           lag(city) OVER (PARTITION BY user_id ORDER BY date) AS prev_city,
           lag(date) OVER (PARTITION BY user_id ORDER BY date) AS prev_date,
           lead(city) OVER (PARTITION BY user_id ORDER BY date) AS next_city,
           lead(date) OVER (PARTITION BY user_id ORDER BY date) AS next_date
    FROM transactions t
),
speeds AS (
    SELECT b.*,
           -- km/h needed to reach this city from the previous / next payment's city
           CASE WHEN prev_city <> b.city THEN
               2 * 6371 * asin(sqrt(pow(sin(radians(c.lat - p.lat) / 2), 2)
                   + cos(radians(p.lat)) * cos(radians(c.lat)) * pow(sin(radians(c.lon - p.lon) / 2), 2)))
               / greatest(epoch(b.date - prev_date) / 3600.0, 1 / 60.0) END AS speed_from_prev,
           CASE WHEN next_city <> b.city THEN
               2 * 6371 * asin(sqrt(pow(sin(radians(n.lat - c.lat) / 2), 2)
                   + cos(radians(c.lat)) * cos(radians(n.lat)) * pow(sin(radians(n.lon - c.lon) / 2), 2)))
               / greatest(epoch(next_date - b.date) / 3600.0, 1 / 60.0) END AS speed_to_next
    FROM base b
    JOIN city_coords c ON c.city = b.city
    LEFT JOIN city_coords p ON p.city = b.prev_city
    LEFT JOIN city_coords n ON n.city = b.next_city
),
flagged AS (
    SELECT *,
        transaction_amount > 15 * category_median AS amount_spike,
        transaction_amount < 2 AND tiny_nearby >= 4 AS card_testing,
        hour(date) < 6 AS night_activity,
        city <> home_city AS away_from_home,
        coalesce(greatest(speed_from_prev, speed_to_next), 0) > 500 AS impossible_travel
    FROM speeds
)
SELECT * FROM flagged
"""


@mcp.tool(annotations=READ_ONLY)
def find_suspicious_transactions(
    start_date: date | None = None,
    end_date: date | None = None,
    user_id: str | None = None,
    rules: list[Rule] | None = None,
    min_risk_score: int = 30,
    limit: int = 50,
) -> dict[str, Any]:
    """Flag transactions that match fraud heuristics and rank them by risk score (0-100).
    Rules: amount_spike, card_testing, night_activity, away_from_home, impossible_travel
    (all by default). Each result lists the rules it matched, the amount relative to the
    category median and, for impossible travel, the journey and speed it would need.
    Keep the default min_risk_score (30): every single fraud pattern scores >= 35, so raising
    it hides real cases. Results are leads for a human analyst, not proof of fraud."""
    active = rules or list(RULES)
    score = " + ".join(f"CASE WHEN {r} THEN {RULES[r][0]} ELSE 0 END" for r in active)
    where, params = _filters(start_date, end_date, user_id)
    limit = max(1, min(limit, MAX_ROWS))
    result = rows(
        f"""SELECT transaction_id, user_id, bank_account_id, date, third_party, category,
                   transaction_amount, city, home_city, payment_method,
                   round(transaction_amount / category_median, 1) AS times_category_median,
                   CASE WHEN impossible_travel THEN
                       CASE WHEN coalesce(speed_from_prev, 0) >= coalesce(speed_to_next, 0)
                            THEN prev_city || ' -> ' || city || ' in ' || CAST(round(epoch(date - prev_date) / 60) AS INTEGER) || ' min'
                            ELSE city || ' -> ' || next_city || ' in ' || CAST(round(epoch(next_date - date) / 60) AS INTEGER) || ' min' END
                       || ' (' || CAST(round(greatest(speed_from_prev, speed_to_next)) AS INTEGER) || ' km/h needed)' END AS travel_evidence,
                   least(100, {score}) AS risk_score,
                   list_filter([{", ".join(f"CASE WHEN {r} THEN '{r}' END" for r in active)}], lambda x: x IS NOT NULL) AS matched_rules
            FROM ({SUSPICIOUS_SQL}) WHERE {where} AND least(100, {score}) >= ?
            ORDER BY risk_score DESC, date LIMIT {limit}""",
        [*params, min_risk_score],
    )
    result["rule_definitions"] = {r: {"weight": RULES[r][0], "meaning": RULES[r][1]} for r in active}
    return result


@mcp.tool(annotations=READ_ONLY)
def query_transactions(sql: str) -> dict[str, Any]:
    """Run ONE read-only DuckDB SELECT (or WITH ... SELECT) over the `transactions` table
    and return at most 200 rows. Use for questions the other tools don't cover.
    Columns: user_id, transaction_id, bank_account_id, financial_institution_id, date (TIMESTAMP),
    third_party, category, transaction_amount (DECIMAL EUR), city, payment_method, currency."""
    try:
        statements = DB.extract_statements(sql)
    except duckdb.Error as exc:
        return {"error": f"SQL parse error: {exc}"}
    if len(statements) != 1:
        return {"error": "exactly one statement is allowed"}
    if statements[0].type != duckdb.StatementType.SELECT:
        return {"error": f"only SELECT queries are allowed, got {statements[0].type.name}"}
    try:
        return rows(statements[0].query)
    except duckdb.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@mcp.prompt()
def fraud_review(month: str = "2025-12") -> str:
    """Ask the model for a fraud review of one month (YYYY-MM)."""
    return (
        f"Review the transactions of {month} for fraud. Use find_suspicious_transactions for that "
        "month, group the results into patterns, look up the profile of the riskiest users, and "
        "write a short report for a bank analyst: what was found, which accounts to block or "
        "contact first, and which flags are probably false alarms."
    )


if __name__ == "__main__":
    mcp.run()
