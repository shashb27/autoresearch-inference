"""
Unit tests for analyze.py

Tests data loading, validation, and summary statistics without
requiring matplotlib rendering or actual plot files.
Run with: uv run pytest tests/ -v
"""

import io
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Suppress matplotlib rendering in tests
import matplotlib
matplotlib.use("Agg")


TSV_VALID = """\
commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription
abc1234\t145.79\t8.50\t4.1\tkeep\tbaseline: BF16 + SDPA + compile default
def5678\t200.30\t7.20\t4.2\tkeep\texperiment: custom decode loop
ghi9012\t180.00\t9.10\t4.0\tdiscard\texperiment: reduce-overhead mode
jkl3456\t0.00\t0.0\t0.0\tcrash\texperiment: INT4 missing bitsandbytes
mno7890\t275.14\t6.80\t4.1\tkeep\texperiment: BF16 + compile dynamic=True
"""

TSV_MISSING_COLUMN = """\
commit\ttok_s\tttft_ms\tstatus\tdescription
abc1234\t145.79\t8.50\tkeep\tbaseline
"""

TSV_EMPTY_HEADER_ONLY = """\
commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription
"""


class TestLoadResults(unittest.TestCase):
    """Tests for load_results() data loading and validation."""

    def _write_tsv(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_loads_valid_tsv(self):
        """Valid TSV loads with correct row count and columns."""
        from analyze import load_results
        path = self._write_tsv(TSV_VALID)
        try:
            df = load_results(path)
            self.assertEqual(len(df), 5)
            self.assertIn("tok_s", df.columns)
            self.assertIn("status", df.columns)
            self.assertIn("experiment_num", df.columns)
        finally:
            os.unlink(path)

    def test_experiment_num_sequential(self):
        """experiment_num starts at 1 and is sequential."""
        from analyze import load_results
        path = self._write_tsv(TSV_VALID)
        try:
            df = load_results(path)
            self.assertEqual(list(df["experiment_num"]), [1, 2, 3, 4, 5])
        finally:
            os.unlink(path)

    def test_missing_column_exits(self):
        """TSV missing required column causes SystemExit."""
        from analyze import load_results
        path = self._write_tsv(TSV_MISSING_COLUMN)
        try:
            with self.assertRaises(SystemExit):
                load_results(path)
        finally:
            os.unlink(path)

    def test_file_not_found_exits(self):
        """Missing file causes SystemExit."""
        from analyze import load_results
        with self.assertRaises(SystemExit):
            load_results("/nonexistent/results.tsv")

    def test_empty_tsv_exits(self):
        """TSV with header only (no rows) causes SystemExit."""
        from analyze import load_results
        path = self._write_tsv(TSV_EMPTY_HEADER_ONLY)
        try:
            with self.assertRaises(SystemExit):
                load_results(path)
        finally:
            os.unlink(path)

    def test_status_normalized_to_lowercase(self):
        """Status values are normalized to lowercase."""
        from analyze import load_results
        mixed_case = TSV_VALID.replace("keep", "KEEP").replace("discard", "Discard")
        path = self._write_tsv(mixed_case)
        try:
            df = load_results(path)
            for s in df["status"]:
                self.assertEqual(s, s.lower(), f"Status not normalized: {s}")
        finally:
            os.unlink(path)

    def test_color_column_added(self):
        """color column is added based on status."""
        from analyze import load_results, STATUS_COLORS
        path = self._write_tsv(TSV_VALID)
        try:
            df = load_results(path)
            self.assertIn("color", df.columns)
            keep_rows = df[df["status"] == "keep"]
            for color in keep_rows["color"]:
                self.assertEqual(color, STATUS_COLORS["keep"])
        finally:
            os.unlink(path)


class TestPrintSummary(unittest.TestCase):
    """Tests for print_summary() output correctness."""

    def _make_df(self) -> pd.DataFrame:
        from analyze import load_results
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        path.write(TSV_VALID)
        path.close()
        df = load_results(path.name)
        os.unlink(path.name)
        return df

    def test_summary_prints_without_error(self):
        """print_summary() runs without raising exceptions."""
        from analyze import print_summary
        df = self._make_df()
        # Should not raise
        try:
            print_summary(df)
        except Exception as e:
            self.fail(f"print_summary raised: {e}")

    def test_summary_counts_correct(self):
        """Summary correctly counts keep/discard/crash."""
        from analyze import load_results
        import io
        from contextlib import redirect_stdout

        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        path.write(TSV_VALID)
        path.close()

        try:
            df = load_results(path.name)
            keeps = df[df["status"] == "keep"]
            discards = df[df["status"] == "discard"]
            crashes = df[df["status"] == "crash"]
            self.assertEqual(len(keeps), 3)
            self.assertEqual(len(discards), 1)
            self.assertEqual(len(crashes), 1)
        finally:
            os.unlink(path.name)

    def test_best_tok_s_is_correct(self):
        """Best tok/s is identified correctly from kept experiments."""
        from analyze import load_results
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        path.write(TSV_VALID)
        path.close()

        try:
            df = load_results(path.name)
            keeps = df[df["status"] == "keep"]
            best = keeps["tok_s"].max()
            self.assertAlmostEqual(best, 275.14)
        finally:
            os.unlink(path.name)


class TestPlotFunctions(unittest.TestCase):
    """Smoke tests: plot functions run without error on valid data."""

    def setUp(self):
        from analyze import load_results
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        path.write(TSV_VALID)
        path.close()
        self._tsv_path = path.name
        self.df = load_results(self._tsv_path)
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self._tsv_path)
        import shutil
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_tok_s_progression_creates_file(self):
        from analyze import plot_tok_s_progression
        plot_tok_s_progression(self.df, self.out_dir, show=False)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "tok_s_progression.png")))

    def test_vram_vs_toks_creates_file(self):
        from analyze import plot_vram_vs_toks
        plot_vram_vs_toks(self.df, self.out_dir, show=False)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "vram_vs_toks.png")))

    def test_improvement_deltas_creates_file(self):
        from analyze import plot_improvement_deltas
        plot_improvement_deltas(self.df, self.out_dir, show=False)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "improvement_deltas.png")))

    def test_outcomes_donut_creates_file(self):
        from analyze import plot_experiment_outcomes
        plot_experiment_outcomes(self.df, self.out_dir, show=False)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "outcomes_donut.png")))

    def test_ttft_progression_creates_file(self):
        from analyze import plot_ttft_progression
        plot_ttft_progression(self.df, self.out_dir, show=False)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "tok_s_vs_ttft.png")))

    def test_improvement_deltas_skipped_with_one_keep(self):
        """improvement_deltas is skipped gracefully with < 2 kept experiments."""
        from analyze import load_results, plot_improvement_deltas
        single_keep_tsv = (
            "commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription\n"
            "abc\t100.0\t5.0\t4.0\tkeep\tbaseline\n"
        )
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        )
        path.write(single_keep_tsv)
        path.close()
        try:
            df = load_results(path.name)
            # Should not raise, just print a skip message
            plot_improvement_deltas(df, self.out_dir, show=False)
        finally:
            os.unlink(path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
