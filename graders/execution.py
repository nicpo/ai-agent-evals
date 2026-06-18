"""Tier 2b: gold-SQL comparison -- Execution Accuracy (EX) and Exact Set Match.

EX is the primary pass/fail signal: run gold and predicted SQL, compare result
sets order-invariantly. ESM and join evaluation are diagnostic: when EX fails,
they tell you *which clause* diverged.
"""

from __future__ import annotations

import sqlite3

import sqlglot
import sqlglot.expressions as exp

from graders.sql_structural import SQLStructuralGrader

_DIALECT = "sqlite"


def _run(conn: sqlite3.Connection, sql: str):
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(cols, row)) for row in rows], None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


def _normalize(rows: list[dict]):
    """Order-invariant canonical form for comparing result sets.

    Each row becomes a tuple of its values (column names discarded), stringified
    so NULLs (None) become "" and don't raise when compared/sorted against other
    values, then sorted within the row. The list of rows is sorted as well.
    Equality of the returned lists is multiset equality of rows, independent of
    row order and of column names/order.
    """
    return sorted(
        [tuple(sorted(str(v) if v is not None else "" for v in r.values()))
         for r in rows]
    )


def execution_accuracy(gold_sql: str, pred_sql: str, conn: sqlite3.Connection) -> dict:
    """Primary metric: True if result sets are equal, order-invariant."""
    gold_rows, gold_err = _run(conn, gold_sql)
    pred_rows, pred_err = _run(conn, pred_sql)

    if gold_err:
        return {"match": False, "reason": "gold_execution_error", "error": gold_err}
    if pred_err:
        return {"match": False, "reason": "pred_execution_error", "error": pred_err}

    match = _normalize(gold_rows) == _normalize(pred_rows)
    return {
        "match": match,
        "gold_row_count": len(gold_rows),
        "pred_row_count": len(pred_rows),
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
