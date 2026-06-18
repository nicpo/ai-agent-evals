"""Tier 2: structural SQL graders, built on sqlglot's AST.

Run when SQL was generated. No database access, no LLM calls.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

_DIALECT = "sqlite"


class SQLStructuralGrader:

    @staticmethod
    def _parse(sql: str):
        return sqlglot.parse_one(sql, dialect=_DIALECT)

    @staticmethod
    def is_parseable(sql: str) -> bool:
        try:
            SQLStructuralGrader._parse(sql)
            return True
        except sqlglot.errors.ParseError:
            return False

    @staticmethod
    def get_referenced_tables(sql: str) -> set[str]:
        ast = SQLStructuralGrader._parse(sql)
        return {t.name.lower() for t in ast.find_all(exp.Table)}

    @staticmethod
    def get_referenced_columns(sql: str) -> set[str]:
        ast = SQLStructuralGrader._parse(sql)
        return {c.name.lower() for c in ast.find_all(exp.Column)}

    @staticmethod
    def has_invalid_references(sql: str, known_schema: dict) -> list[str]:
        """Names referenced in SQL that don't exist in the schema.

        CTE names and table aliases are excluded so a WITH clause or a
        ``provinces p`` alias is not reported as a hallucinated table. Output
        aliases (``SUM(...) AS total_tribute_collected``) are excluded so a
        reference to one in ORDER BY/GROUP BY/HAVING is not reported as a
        hallucinated column.
        """
        valid_tables = {t["name"].lower() for t in known_schema["tables"]}
        valid_columns = {
            col["name"].lower()
            for t in known_schema["tables"]
            for col in t["columns"]
        }

        ast = SQLStructuralGrader._parse(sql)
        cte_names = {cte.alias.lower() for cte in ast.find_all(exp.CTE) if cte.alias}
        # Names introduced by `expr AS name` (SELECT output columns, subqueries).
        alias_names = {a.alias.lower() for a in ast.find_all(exp.Alias) if a.alias}

        hallucinated: list[str] = []
        for t in {tbl.name.lower() for tbl in ast.find_all(exp.Table)}:
            if t not in valid_tables and t not in cte_names:
                hallucinated.append(f"table:{t}")
        for c in {col.name.lower() for col in ast.find_all(exp.Column)}:
            if c not in valid_columns and c not in alias_names:
                hallucinated.append(f"column:{c}")
        return hallucinated

    @staticmethod
    def get_join_info(sql: str) -> list[dict]:
        ast = SQLStructuralGrader._parse(sql)
        joins = []
        for join in ast.find_all(exp.Join):
            joins.append(
                {
                    "table": join.this.name if hasattr(join.this, "name") else str(join.this),
                    # sqlglot puts LEFT/RIGHT/FULL in 'side' and INNER/CROSS in 'kind';
                    # default to INNER when neither is present.
                    "kind": (join.args.get("side") or join.args.get("kind") or "INNER").upper(),
                    "on": str(join.args.get("on", "")),
                }
            )
        return joins

    @staticmethod
    def has_aggregation(sql: str) -> bool:
        ast = SQLStructuralGrader._parse(sql)
        agg_funcs = (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min)
        return any(ast.find(f) for f in agg_funcs)

    @staticmethod
    def has_group_by(sql: str) -> bool:
        ast = SQLStructuralGrader._parse(sql)
        return ast.find(exp.Group) is not None


def grade(sql: str, known_schema: dict) -> dict:
    """Run all Tier 2 graders. Returns is_parseable=False (only) if unparseable."""
    g = SQLStructuralGrader
    if not g.is_parseable(sql):
        return {"is_parseable": False}
    return {
        "is_parseable": True,
        "referenced_tables": sorted(g.get_referenced_tables(sql)),
        "hallucinated_references": g.has_invalid_references(sql, known_schema),
        "joins": g.get_join_info(sql),
        "has_aggregation": g.has_aggregation(sql),
        "has_group_by": g.has_group_by(sql),
    }
