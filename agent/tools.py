"""The three agent tools: get_schema, run_query, get_sample_rows.

Each tool is exposed in two forms:

* A plain Python function (``Toolbox.get_schema`` etc.) -- easy to unit test and
  to call directly from graders.
* A LangChain ``StructuredTool`` (``Toolbox.langchain_tools``) -- bound to the
  model so tool-calling works identically across providers.

``run_query`` returns errors *as data* (in the ``error`` field) rather than
raising, so the agent can observe and self-correct -- an eval signal in itself.

NOTE: if you remove tools (like `get_schema`) from the system prompt, remove or 
      comment out tool defs here so they don't show up in the agent's tool list
"""

from __future__ import annotations

import copy
import sqlite3
from dataclasses import dataclass

from agent import db

# Keywords that indicate a data-modifying or schema-changing statement. The
# read-only connection already blocks these at the SQLite level; this check
# rejects them earlier with a clear message and is the belt-and-suspenders layer
# the Tier 1 grader also looks for.
_DML_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "ATTACH", "PRAGMA",
}

_MAX_ROWS = 1000  # safety cap on returned rows
AVAILABLE_TOOLS = frozenset({"get_schema", "run_query", "get_sample_rows"})


def _looks_like_dml(sql: str) -> str | None:
    """Return an error string if the SQL is not a single read-only statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "Empty query."
    # Reject multiple statements outright.
    if ";" in stripped:
        return "Multiple statements are not allowed; submit a single SELECT."
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "Only SELECT (or WITH ... SELECT) queries are allowed."
    # Token-boundary scan so we don't trip on substrings like 'updated_at'.
    import re

    tokens = set(re.findall(r"[A-Za-z_]+", upper))
    hit = tokens & _DML_KEYWORDS
    if hit:
        return f"Disallowed keyword(s) in query: {', '.join(sorted(hit))}."
    return None


@dataclass
class Toolbox:
    """Bundles the agent tools around a single read-only DB connection."""

    conn: sqlite3.Connection
    enabled_tools: frozenset[str] = AVAILABLE_TOOLS

    @classmethod
    def open(cls, enabled_tools=None) -> "Toolbox":
        tools = AVAILABLE_TOOLS if enabled_tools is None else frozenset(enabled_tools)
        unknown = tools - AVAILABLE_TOOLS
        if unknown:
            raise ValueError(f"Unknown tool(s): {', '.join(sorted(unknown))}")
        return cls(conn=db.connect(), enabled_tools=tools)

    def close(self) -> None:
        self.conn.close()

    # -- Tool implementations ------------------------------------------------
    def get_schema(self) -> dict:
        """Return the full schema as a structured dict """
        # Deep copy so callers can't mutate the module-level SCHEMA.
        return copy.deepcopy(db.SCHEMA)

    def run_query(self, sql: str) -> dict:
        """Execute a read-only SELECT and return columns/rows or an error."""
        result = {"columns": [], "rows": [], "row_count": 0, "error": None}

        err = _looks_like_dml(sql)
        if err:
            result["error"] = err
            return result

        try:
            cursor = self.conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            fetched = cursor.fetchmany(_MAX_ROWS)
            rows = [dict(zip(columns, row)) for row in fetched]
            result["columns"] = columns
            result["rows"] = rows
            result["row_count"] = len(rows)
        except sqlite3.Error as e:
            # Return the error as data so the agent can self-correct.
            result["error"] = str(e)
        return result

    def get_sample_rows(self, table: str, n: int = 3) -> dict:
        """Return up to n sample rows from a named table."""
        if table not in db.TABLE_NAMES:
            return {"table": table, "rows": [], "error": f"Unknown table: {table}"}
        n = max(1, min(int(n), 20))
        # Table name validated against the allow-list above, so interpolation is safe.
        cursor = self.conn.execute(f"SELECT * FROM {table} LIMIT {n}")
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"table": table, "rows": rows, "error": None}

    # -- LangChain integration ----------------------------------------------
    @property
    def dispatch_map(self) -> dict:
        """Maps tool name -> callable taking a kwargs dict."""
        all_tools = {
            "get_schema": lambda args: self.get_schema(),
            "run_query": lambda args: self.run_query(args["sql"]),
            "get_sample_rows": lambda args: self.get_sample_rows(
                args["table"], args.get("n", 3)
            ),
        }
        return {name: fn for name, fn in all_tools.items() if name in self.enabled_tools}

    def dispatch(self, name: str, args: dict) -> dict:
        fn = self.dispatch_map.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        return fn(args or {})

    def langchain_tools(self) -> list:
        """Build StructuredTools for binding to a chat model.

        Imported lazily so that graders (which never call an LLM) don't need
        LangChain installed.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class GetSchemaArgs(BaseModel):
            pass

        class RunQueryArgs(BaseModel):
            sql: str = Field(description="A single read-only SELECT statement.")

        class GetSampleRowsArgs(BaseModel):
            table: str = Field(description="Name of the table to sample.")
            n: int = Field(default=3, description="Number of rows to return.")

        all_tools = {
            "get_schema":
            StructuredTool.from_function(
                func=lambda: self.get_schema(),
                name="get_schema",
                description=(
                    "Return the full database schema: tables, columns, types, "
                    "descriptions, and foreign keys. Enum values are not included."
                ),
                args_schema=GetSchemaArgs,
            ),
            "run_query": StructuredTool.from_function(
                func=lambda sql: self.run_query(sql),
                name="run_query",
                description=(
                    "Execute a read-only SELECT query. Returns columns and rows, "
                    "or an 'error' string if the query failed. DML is blocked."
                ),
                args_schema=RunQueryArgs,
            ),
            "get_sample_rows": StructuredTool.from_function(
                func=lambda table, n=3: self.get_sample_rows(table, n),
                name="get_sample_rows",
                description=(
                    "Return a few sample rows from a table -- useful for "
                    "discovering enum values and how values co-occur across columns."
                ),
                args_schema=GetSampleRowsArgs,
            ),
        }
        return [all_tools[name] for name in sorted(self.enabled_tools)]
