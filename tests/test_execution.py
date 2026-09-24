from __future__ import annotations

import sqlite3
import unittest

from graders.execution import execution_accuracy


class StrictExecutionAccuracyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")

    def tearDown(self) -> None:
        self.conn.close()

    def test_strict_typed_positional_comparison(self) -> None:
        ex = lambda gold, pred: execution_accuracy(gold, pred, self.conn)
        self.assertTrue(ex("SELECT 13400", "SELECT 13400.0")["match"])
        self.assertFalse(ex("SELECT 13400", "SELECT '13400'")["match"])
        self.assertFalse(ex("SELECT NULL", "SELECT ''")["match"])
        self.assertFalse(ex("SELECT 'Rome', 10", "SELECT 10, 'Rome'")["match"])
        self.assertTrue(ex("SELECT 1 UNION ALL SELECT 2", "SELECT 2 UNION ALL SELECT 1")["match"])
        self.assertFalse(ex("SELECT 1 UNION ALL SELECT 1", "SELECT 1")["match"])
        self.assertFalse(ex("SELECT 1", "SELECT 1, 2")["match"])
        self.assertFalse(ex("SELECT 1 AS x, 2 AS x", "SELECT 1 AS x, 3 AS x")["match"])
        self.assertTrue(ex("SELECT 1 WHERE 0", "SELECT 1, 2 WHERE 0")["match"])
        self.assertEqual(ex("SELECT missing", "SELECT 1")["reason"], "gold_execution_error")
        self.assertEqual(ex("SELECT 1", "SELECT missing")["reason"], "pred_execution_error")


if __name__ == "__main__":
    unittest.main()
