"""Tier 2b: gold-SQL comparison -- Execution Accuracy (EX) and Exact Set Match.

EX is the primary pass/fail signal: run gold and predicted SQL, compare result
sets order-invariantly. ESM and join evaluation are diagnostic: when EX fails,
they tell you *which clause* diverged.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from graders.sql_structural import SQLStructuralGrader

_DIALECT = "sqlite"


@dataclass(frozen=True)
class _ResultSet:
    """Raw SQLite output, retaining schema order and every returned value."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


def _run(conn: sqlite3.Connection, sql: str):
    try:
        cursor = conn.execute(sql)
        columns = tuple(d[0] for d in cursor.description) if cursor.description else ()
        rows = tuple(tuple(row) for row in cursor.fetchall())
        return _ResultSet(columns, rows), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _canonical_value(value: Any) -> tuple:
    """Return a typed, hashable scalar representation for strict EX."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, Decimal)):
        return ("number", Decimal(str(value)))
    if isinstance(value, str):
        return ("text", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, memoryview):
        return ("bytes", value.tobytes())
    value_type = type(value)
    return ("other", value_type.__module__, value_type.__qualname__, repr(value))


def _normalize(rows: tuple[tuple[Any, ...], ...]) -> Counter:
    """Order-invariant multiset of positional, typed rows.

    Column aliases and output schema are intentionally not compared. Values are
    never sorted within a row, and empty outputs are schema-insensitive.
    """
    return Counter(tuple(_canonical_value(value) for value in row) for row in rows)


def execution_accuracy(gold_sql: str, pred_sql: str, conn: sqlite3.Connection) -> dict:
    """Primary metric: True if result sets are equal, order-invariant."""
    gold_rows, gold_err = _run(conn, gold_sql)
    pred_rows, pred_err = _run(conn, pred_sql)

    if gold_err:
        return {"match": False, "reason": "gold_execution_error", "error": gold_err}
    if pred_err:
        return {"match": False, "reason": "pred_execution_error", "error": pred_err}

    assert gold_rows is not None and pred_rows is not None
    common = {
        "gold_row_count": len(gold_rows.rows),
        "pred_row_count": len(pred_rows.rows),
        "gold_column_count": len(gold_rows.columns),
        "pred_column_count": len(pred_rows.columns),
    }
    match = _normalize(gold_rows.rows) == _normalize(pred_rows.rows)
    return {
        "match": match,
        **common,
        "reason": None if match else "result_set_mismatch",
    }


def exact_set_match(gold_sql: str, pred_sql: str) -> dict:
    """Structural clause-by-clause comparison (Spider protocol). Diagnostic only."""
    gold_ast = sqlglot.parse_one(gold_sql, dialect=_DIALECT)
    pred_ast = sqlglot.parse_one(pred_sql, dialect=_DIALECT)

    def select_cols(ast):
        return {
            str(e)
            for e in ast.find_all(exp.Column)
            if isinstance(e.parent, (exp.Select, exp.Alias))
        }

    def tables(ast):
        return {t.name.lower() for t in ast.find_all(exp.Table)}

    def where(ast):
        w = ast.find(exp.Where)
        return str(w) if w else ""

    def group_by(ast):
        g = ast.find(exp.Group)
        return {str(e) for e in g.expressions} if g else set()

    def order_by(ast):
        o = ast.find(exp.Order)
        return str(o) if o else ""

    return {
        "select": select_cols(gold_ast) == select_cols(pred_ast),
        "from": tables(gold_ast) == tables(pred_ast),
        "where": where(gold_ast) == where(pred_ast),
        "group_by": group_by(gold_ast) == group_by(pred_ast),
        "order_by": order_by(gold_ast) == order_by(pred_ast),
    }


def evaluate_joins(gold_sql: str, pred_sql: str) -> dict:
    """Diagnostic join comparison. Notably flags INNER-where-LEFT-needed cases."""
    gold_joins = SQLStructuralGrader.get_join_info(gold_sql)
    pred_joins = SQLStructuralGrader.get_join_info(pred_sql)

    gold_tables = {j["table"].lower() for j in gold_joins}
    pred_tables = {j["table"].lower() for j in pred_joins}
    gold_types = sorted(j["kind"] for j in gold_joins)
    pred_types = sorted(j["kind"] for j in pred_joins)

    return {
        "join_count_match": len(gold_joins) == len(pred_joins),
        "join_tables_match": gold_tables == pred_tables,
        "join_type_match": gold_types == pred_types,
        "missing_tables": sorted(gold_tables - pred_tables),
        "extra_tables": sorted(pred_tables - gold_tables),
    }


def grade(gold_sql: str, pred_sql: str, conn: sqlite3.Connection) -> dict:
    """Run EX (always) plus ESM and join diagnostics (when both parse)."""
    out = {"execution_accuracy": execution_accuracy(gold_sql, pred_sql, conn)}
    if SQLStructuralGrader.is_parseable(gold_sql) and SQLStructuralGrader.is_parseable(pred_sql):
        out["exact_set_match"] = exact_set_match(gold_sql, pred_sql)
        out["joins"] = evaluate_joins(gold_sql, pred_sql)
    return out
