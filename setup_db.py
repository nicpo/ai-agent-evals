"""Create and seed data/roman_empire.db from the CSV files in data/tables/.

Run from the repo root:
    python setup_db.py
"""

import csv
import sqlite3
from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parent / "data" / "tables"
DB_PATH = Path(__file__).resolve().parent / "data" / "roman_empire.db"


def _csv_rows(filename: str) -> list[dict]:
    with open(TABLES_DIR / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _null(value: str) -> str | None:
    return None if value == "" else value


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE provinces (
            province_id   INTEGER PRIMARY KEY,
            name          TEXT    NOT NULL,
            region        TEXT    NOT NULL,
            annual_tribute INTEGER NOT NULL,
            status        TEXT    NOT NULL,
            founded_year  INTEGER NOT NULL,
            governor_id   INTEGER REFERENCES officials(official_id)
        );

        CREATE TABLE officials (
            official_id    INTEGER PRIMARY KEY,
            name           TEXT    NOT NULL,
            province_id    INTEGER REFERENCES provinces(province_id),
            title          TEXT    NOT NULL,
            annual_salary  INTEGER NOT NULL,
            appointed_date TEXT    NOT NULL,
            status         TEXT    NOT NULL
        );

        CREATE TABLE decrees (
            decree_id          INTEGER PRIMARY KEY,
            title              TEXT    NOT NULL,
            issuing_official_id INTEGER NOT NULL REFERENCES officials(official_id),
            scope              TEXT    NOT NULL,
            category           TEXT    NOT NULL,
            status             TEXT    NOT NULL,
            issued_year        INTEGER NOT NULL,
            expiry_year        INTEGER,
            fine_denarii       INTEGER NOT NULL
        );

        CREATE TABLE tribute_records (
            record_id          INTEGER PRIMARY KEY,
            official_id        INTEGER NOT NULL REFERENCES officials(official_id),
            province_id        INTEGER NOT NULL REFERENCES provinces(province_id),
            year               INTEGER NOT NULL,
            amount_owed        INTEGER NOT NULL,
            amount_collected   INTEGER NOT NULL,
            shortfall_reason   TEXT    NOT NULL
        );
    """)


def seed(conn: sqlite3.Connection) -> None:
    # provinces and officials have a circular FK dependency:
    #   provinces.governor_id -> officials.official_id
    #   officials.province_id -> provinces.province_id
    # Disable FK enforcement during the load, then re-enable it.
    conn.execute("PRAGMA foreign_keys = OFF")

    provinces = _csv_rows("provinces.csv")
    conn.executemany(
        "INSERT INTO provinces VALUES (?,?,?,?,?,?,?)",
        [
            (
                int(r["province_id"]),
                r["name"],
                r["region"],
                int(r["annual_tribute"]),
                r["status"],
                int(r["founded_year"]),
                _null(r["governor_id"]) and int(r["governor_id"]),
            )
            for r in provinces
        ],
    )

    officials = _csv_rows("officials.csv")
    conn.executemany(
        "INSERT INTO officials VALUES (?,?,?,?,?,?,?)",
        [
            (
                int(r["official_id"]),
                r["name"],
                _null(r["province_id"]) and int(r["province_id"]),
                r["title"],
                int(r["annual_salary"]),
                r["appointed_date"],
                r["status"],
            )
            for r in officials
        ],
    )

    conn.execute("PRAGMA foreign_keys = ON")

    decrees = _csv_rows("decrees.csv")
    conn.executemany(
        "INSERT INTO decrees VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                int(r["decree_id"]),
                r["title"],
                int(r["issuing_official_id"]),
                r["scope"],
                r["category"],
                r["status"],
                int(r["issued_year"]),
                _null(r["expiry_year"]) and int(r["expiry_year"]),
                int(r["fine_denarii"]),
            )
            for r in decrees
        ],
    )

    tribute_records = _csv_rows("tribute_records.csv")
    conn.executemany(
        "INSERT INTO tribute_records VALUES (?,?,?,?,?,?,?)",
        [
            (
                int(r["record_id"]),
                int(r["official_id"]),
                int(r["province_id"]),
                int(r["year"]),
                int(r["amount_owed"]),
                int(r["amount_collected"]),
                r["shortfall_reason"],
            )
            for r in tribute_records
        ],
    )


def verify(conn: sqlite3.Connection) -> None:
    expected = {
        "provinces": 15,
        "officials": 20,
        "decrees": 18,
        "tribute_records": 47,
    }
    for table, count in expected.items():
        (actual,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        status = "OK" if actual == count else f"MISMATCH (expected {count})"
        print(f"  {table}: {actual} rows  {status}")


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH.name}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        create_schema(conn)
        seed(conn)
        conn.commit()
        print(f"Created {DB_PATH}")
        verify(conn)


if __name__ == "__main__":
    main()
