"""Generate valid, schema-preserving database perturbations ("scenarios").

Method (see the linked post): a query that passes EX and gets a `CORRECT`
verdict from the judge has only been checked against *one* database snapshot.
A scenario is a small, legal change to that snapshot -- one the schema already
permits, such as a NULL in a column that happens to have none, or two rows
sharing a value that happens to be unique so far. If gold SQL and agent SQL
still agree after the change, the original agreement holds up. If they don't,
the original pass may have been hiding a latent bug.

Scenarios are discovered generically from the live schema (via
``PRAGMA table_info`` / ``PRAGMA foreign_key_list``) rather than hardcoded, so
this works against whatever database the eval set under test points at:

* **null**: clone an existing, valid row and set one nullable, non-key column
  to NULL -- a legal state the schema already allows but the current data may
  not contain.
* **duplicate**: clone an existing, valid row and copy another row's value
  into one free-text column that isn't already declared unique -- e.g. two
  officials sharing a name.

Each scenario is materialized as its own temporary copy of the database, so
the real database file is never touched and scenarios never interact with
each other.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ScenarioError(RuntimeError):
    """A scenario could not be constructed or violated database integrity."""


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    table: str
    column: str
    kind: str  # "null" or "duplicate"


def _pk_column(conn: sqlite3.Connection, table: str) -> str:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        # cid, name, type, notnull, dflt_value, pk
        if row[5] == 1:
            return row[1]
    raise ScenarioError(f"Table {table!r} has no single-column primary key.")


def _columns(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(f"PRAGMA table_info({table})"))


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [r[0] for r in rows]


def discover_scenarios(conn: sqlite3.Connection) -> list[Scenario]:
    """Enumerate legal, generic perturbations for every table in the schema."""
    scenarios: list[Scenario] = []
    for table in _tables(conn):
        cols = _columns(conn, table)
        pk = {c["name"] for c in cols if c["pk"] == 1}
        for col in cols:
            name = col["name"]
            if name in pk:
                continue
            if col["notnull"] == 0:
                scenarios.append(Scenario(
                    name=f"{table}.{name}__null",
                    description=(
                        f"Clone an existing {table} row and set {name} to NULL "
                        f"-- a legal state the schema already allows."
                    ),
                    table=table, column=name, kind="null",
                ))
            elif (col["type"] or "").upper().startswith(("TEXT", "CHAR", "VARCHAR")):
                scenarios.append(Scenario(
                    name=f"{table}.{name}__duplicate",
                    description=(
                        f"Clone an existing {table} row and copy another row's "
                        f"{name} value into it -- two rows sharing a value that "
                        f"happens to be unique so far."
                    ),
                    table=table, column=name, kind="duplicate",
                ))
    return scenarios


def apply_scenario(conn: sqlite3.Connection, scenario: Scenario) -> None:
    """Mutate ``conn`` in place per the scenario. Raises ScenarioError on failure."""
    conn.row_factory = sqlite3.Row
    pk = _pk_column(conn, scenario.table)
    base = conn.execute(f"SELECT * FROM {scenario.table} LIMIT 1").fetchone()
    if base is None:
        raise ScenarioError(f"Table {scenario.table!r} is empty; cannot clone a row.")
    row = dict(base)

    (new_id,) = conn.execute(f"SELECT COALESCE(MAX({pk}), 0) + 1 FROM {scenario.table}").fetchone()
    row[pk] = new_id

    if scenario.kind == "null":
        row[scenario.column] = None
    elif scenario.kind == "duplicate":
        other = conn.execute(
            f"SELECT {scenario.column} FROM {scenario.table} WHERE {pk} != ? "
            f"AND {scenario.column} IS NOT NULL LIMIT 1",
            (base[pk],),
        ).fetchone()
        if other is None:
            raise ScenarioError(f"No other row to duplicate {scenario.column} from.")
        row[scenario.column] = other[0]
    else:
        raise ScenarioError(f"Unknown scenario kind: {scenario.kind!r}")

    columns = list(row)
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {scenario.table} ({', '.join(columns)}) VALUES ({placeholders})",
        [row[c] for c in columns],
    )
    conn.commit()

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ScenarioError(f"Scenario {scenario.name!r} violated foreign keys: {violations}")


def build_scenario_db(source_db: Path, scenario: Scenario) -> Path:
    """Copy ``source_db`` to a temp file, apply the scenario, and return its path.

    Caller is responsible for deleting the returned temp file.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix=f"scenario_{scenario.kind}_")
    import os

    os.close(fd)
    shutil.copyfile(source_db, tmp_path)
    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        apply_scenario(conn, scenario)
    finally:
        conn.close()
    return Path(tmp_path)
