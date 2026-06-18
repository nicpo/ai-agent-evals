"""Database access and authored schema metadata.

The SQLite database already exists at ``data/roman_empire.db``; this module only
*reads* it. Connections are opened read-only at the OS level (SQLite ``mode=ro``
URI) as the first line of defence against DML -- the keyword check in
``tools.run_query`` is the belt-and-suspenders second line.

``SCHEMA`` is authored here rather than reflected from the database because the
spec requires plain-language column/table descriptions and an explicit
``foreign_key`` for every column -- including the self-referential
``provinces.governor_id -> officials.official_id`` relationship, which is not
declared as a constraint in the DB (it is set after the initial insert).
Enum values are deliberately omitted so the agent must discover them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DB_PATH


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a read-only connection to the SQLite database.

    Uses the ``file:...?mode=ro`` URI so that any attempt to mutate data fails
    at the SQLite level with ``sqlite3.OperationalError`` rather than silently
    succeeding.
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# --- Authored schema ---------------------------------------------------------
SCHEMA: dict = {
    "tables": [
        {
            "name": "provinces",
            "description": "Provinces administered by the Roman Empire.",
            "columns": [
                {"name": "province_id", "type": "INTEGER", "description": "Unique identifier for the province.", "primary_key": True, "nullable": False, "foreign_key": None},
                {"name": "name", "type": "TEXT", "description": "Name of the province (e.g. 'Gaul', 'Hispania', 'Egypt').", "primary_key": False, "nullable": False, "foreign_key": None},
                {"name": "region", "type": "TEXT", "description": "Broad geographic grouping the province belongs to.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "annual_tribute", "type": "REAL", "description": "Target annual tribute owed by the province, in denarii.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "status", "type": "TEXT", "description": "Current administrative standing of the province.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "founded_year", "type": "INTEGER", "description": "Year the province was established (negative values denote BCE).", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "governor_id", "type": "INTEGER", "description": "The official currently registered as governor of this province.", "primary_key": False, "nullable": True, "foreign_key": "officials.official_id"},
            ],
        },
        {
            "name": "officials",
            "description": "Roman officials assigned to govern or administer provinces.",
            "columns": [
                {"name": "official_id", "type": "INTEGER", "description": "Unique identifier for the official.", "primary_key": True, "nullable": False, "foreign_key": None},
                {"name": "name", "type": "TEXT", "description": "Full name of the official.", "primary_key": False, "nullable": False, "foreign_key": None},
                {"name": "province_id", "type": "INTEGER", "description": "Province this official is assigned to; NULL for unassigned senators and tribunes.", "primary_key": False, "nullable": True, "foreign_key": "provinces.province_id"},
                {"name": "title", "type": "TEXT", "description": "The official's rank or office.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "annual_salary", "type": "REAL", "description": "Annual salary paid to the official, in denarii.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "appointed_date", "type": "TEXT", "description": "ISO-8601 date the official was appointed (e.g. '0063-04-01').", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "status", "type": "TEXT", "description": "Current standing of the official in the empire.", "primary_key": False, "nullable": True, "foreign_key": None},
            ],
        },
        {
            "name": "decrees",
            "description": "Edicts and decrees issued by officials.",
            "columns": [
                {"name": "decree_id", "type": "INTEGER", "description": "Unique identifier for the decree.", "primary_key": True, "nullable": False, "foreign_key": None},
                {"name": "title", "type": "TEXT", "description": "Title of the decree.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "issuing_official_id", "type": "INTEGER", "description": "Official who issued the decree.", "primary_key": False, "nullable": True, "foreign_key": "officials.official_id"},
                {"name": "scope", "type": "TEXT", "description": "Whether the decree applies empire-wide or to a single province.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "category", "type": "TEXT", "description": "Subject area of the decree.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "status", "type": "TEXT", "description": "Current legal standing of the decree.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "issued_year", "type": "INTEGER", "description": "Year the decree was issued.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "expiry_year", "type": "INTEGER", "description": "Year the decree expires; NULL if indefinite.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "fine_denarii", "type": "REAL", "description": "Penalty in denarii for non-compliance; 0 for non-financial orders.", "primary_key": False, "nullable": True, "foreign_key": None},
            ],
        },
        {
            "name": "tribute_records",
            "description": "Annual tribute collection records per province and collecting official.",
            "columns": [
                {"name": "record_id", "type": "INTEGER", "description": "Unique identifier for the tribute record.", "primary_key": True, "nullable": False, "foreign_key": None},
                {"name": "official_id", "type": "INTEGER", "description": "Official responsible for collecting this tribute.", "primary_key": False, "nullable": True, "foreign_key": "officials.official_id"},
                {"name": "province_id", "type": "INTEGER", "description": "Province the tribute was levied on.", "primary_key": False, "nullable": True, "foreign_key": "provinces.province_id"},
                {"name": "year", "type": "INTEGER", "description": "Year the tribute was due.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "amount_owed", "type": "REAL", "description": "Tribute amount owed for the year, in denarii.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "amount_collected", "type": "REAL", "description": "Tribute amount actually collected, in denarii.", "primary_key": False, "nullable": True, "foreign_key": None},
                {"name": "shortfall_reason", "type": "TEXT", "description": "Why collected tribute fell short of the amount owed, if at all.", "primary_key": False, "nullable": True, "foreign_key": None},
            ],
        },
    ]
}


# Set of valid table names, used by tools to validate ``get_sample_rows`` input.
TABLE_NAMES: set[str] = {t["name"] for t in SCHEMA["tables"]}
