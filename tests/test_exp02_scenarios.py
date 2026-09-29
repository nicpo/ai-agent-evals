from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "roman_empire.db"

from experiments.exp_02_graders_agreement.scenarios import (
    ScenarioError,
    apply_scenario,
    build_scenario_db,
    discover_scenarios,
)


def _ensure_db() -> None:
    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "setup_db.py")], check=True, cwd=REPO_ROOT)


class ScenarioDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_db()
        cls.conn = sqlite3.connect(DB_PATH)
        cls.conn.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def test_discovers_at_least_one_null_and_one_duplicate_scenario(self) -> None:
        scenarios = discover_scenarios(self.conn)
        kinds = {s.kind for s in scenarios}
        self.assertIn("null", kinds)
        self.assertIn("duplicate", kinds)

    def test_every_discovered_scenario_applies_cleanly(self) -> None:
        for scenario in discover_scenarios(self.conn):
            with self.subTest(scenario=scenario.name):
                tmp_path = build_scenario_db(DB_PATH, scenario)
                try:
                    conn = sqlite3.connect(tmp_path)
                    try:
                        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                        self.assertEqual(violations, [])
                        (count,) = conn.execute(f"SELECT COUNT(*) FROM {scenario.table}").fetchone()
                        self.assertGreater(count, 0)
                    finally:
                        conn.close()
                finally:
                    os.unlink(tmp_path)

    def test_null_scenario_actually_inserts_a_null(self) -> None:
        scenarios = [s for s in discover_scenarios(self.conn) if s.kind == "null"]
        self.assertTrue(scenarios)
        scenario = scenarios[0]
        tmp_path = build_scenario_db(DB_PATH, scenario)
        try:
            conn = sqlite3.connect(tmp_path)
            try:
                (n,) = conn.execute(
                    f"SELECT COUNT(*) FROM {scenario.table} WHERE {scenario.column} IS NULL"
                ).fetchone()
                self.assertGreaterEqual(n, 1)
            finally:
                conn.close()
        finally:
            os.unlink(tmp_path)

    def test_duplicate_scenario_requires_another_row(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("INSERT INTO t VALUES (1, 'only')")
        from experiments.exp_02_graders_agreement.scenarios import Scenario

        scenario = Scenario(name="t.name__duplicate", description="", table="t", column="name", kind="duplicate")
        with self.assertRaises(ScenarioError):
            apply_scenario(conn, scenario)


if __name__ == "__main__":
    unittest.main()
