"""
Unit tests for submit_run.py and leaderboard.py.

Run with: uv run pytest tests/test_leaderboard.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HARDWARE_JSON = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_family": "Ada Lovelace",
    "compute_capability": "8.9",
    "vram_total_gb": 24.0,
    "cuda_version": "12.4",
    "bf16_supported": True,
    "fp8_supported": True,
}

CONFIG_JSON = {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "device": "cuda",
    "vram_limit_gb": 20.0,
    "max_new_tokens": 512,
}

TSV_CONTENT = """\
commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription
abc1234\t145.79\t8.50\t4.1\tkeep\tbaseline: BF16 + SDPA + compile default
def5678\t200.30\t7.20\t4.2\tkeep\texperiment: BF16 + flash_attention_2
ghi9012\t180.00\t9.10\t4.0\tdiscard\texperiment: reduce-overhead mode
jkl3456\t0.00\t0.0\t0.0\tcrash\texperiment: INT4 missing bitsandbytes
mno7890\t275.14\t6.80\t4.1\tkeep\texperiment: BF16 + compile dynamic=True
"""


def _write_json(d: dict, suffix=".json") -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    json.dump(d, f)
    f.close()
    return f.name


def _write_text(content: str, suffix=".tsv") -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# submit_run.py tests
# ---------------------------------------------------------------------------

class TestLoadExperiments(unittest.TestCase):
    def test_parses_valid_tsv(self):
        from submit_run import _load_experiments
        path = _write_text(TSV_CONTENT)
        try:
            expts = _load_experiments(path)
            self.assertEqual(len(expts), 5)
            self.assertEqual(expts[0]["tok_s"], 145.79)
            self.assertEqual(expts[0]["status"], "keep")
        finally:
            os.unlink(path)

    def test_returns_empty_for_missing_file(self):
        from submit_run import _load_experiments
        result = _load_experiments("/nonexistent/results.tsv")
        self.assertEqual(result, [])

    def test_returns_empty_for_header_only(self):
        from submit_run import _load_experiments
        path = _write_text("commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription\n")
        try:
            result = _load_experiments(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def test_skips_malformed_rows(self):
        from submit_run import _load_experiments
        content = (
            "commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription\n"
            "abc\tnot_a_number\t8.0\t4.0\tkeep\tbaseline\n"
            "def\t100.0\t5.0\t4.0\tkeep\tgood row\n"
        )
        path = _write_text(content)
        try:
            result = _load_experiments(path)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0]["tok_s"], 100.0)
        finally:
            os.unlink(path)


class TestInferParamsB(unittest.TestCase):
    def test_known_model_qwen(self):
        from submit_run import _infer_params_b
        self.assertEqual(_infer_params_b("Qwen/Qwen2.5-0.5B"), 0.5)
        self.assertEqual(_infer_params_b("Qwen/Qwen2.5-7B"), 7.0)
        self.assertEqual(_infer_params_b("Qwen/Qwen2.5-72B"), 72.0)

    def test_known_model_llama(self):
        from submit_run import _infer_params_b
        self.assertEqual(_infer_params_b("meta-llama/Llama-3.1-8B"), 8.0)
        self.assertEqual(_infer_params_b("meta-llama/Llama-3.1-70B"), 70.0)

    def test_fallback_regex(self):
        from submit_run import _infer_params_b
        self.assertEqual(_infer_params_b("some-org/mystery-13b-instruct"), 13.0)

    def test_unknown_returns_zero(self):
        from submit_run import _infer_params_b
        self.assertEqual(_infer_params_b("unknown-model-no-params"), 0.0)


class TestInferModelFamily(unittest.TestCase):
    def test_qwen_family(self):
        from submit_run import _infer_model_family
        self.assertEqual(_infer_model_family("Qwen/Qwen2.5-7B"), "Qwen2.5")

    def test_llama_family(self):
        from submit_run import _infer_model_family
        self.assertIn("Llama", _infer_model_family("meta-llama/Llama-3.1-8B"))

    def test_mistral_family(self):
        from submit_run import _infer_model_family
        self.assertEqual(_infer_model_family("mistralai/Mistral-7B-v0.1"), "Mistral")

    def test_gemma_family(self):
        from submit_run import _infer_model_family
        self.assertIn("Gemma", _infer_model_family("google/gemma-2-9b"))


class TestExtractTechniques(unittest.TestCase):
    def test_bf16(self):
        from submit_run import _extract_techniques
        self.assertIn("bf16", _extract_techniques("BF16 + SDPA + compile default"))

    def test_flash_attention(self):
        from submit_run import _extract_techniques
        self.assertIn("flash_attention_2", _extract_techniques("BF16 + flash_attention_2 + compile"))

    def test_fp8(self):
        from submit_run import _extract_techniques
        self.assertIn("fp8_quantization", _extract_techniques("FP8 weight quantization"))

    def test_baseline_fallback(self):
        from submit_run import _extract_techniques
        result = _extract_techniques("nothing special here xyz")
        self.assertEqual(result, ["baseline"])

    def test_multiple_techniques(self):
        from submit_run import _extract_techniques
        desc = "BF16 + Flash Attention 2 + FP8 + torch.compile"
        techs = _extract_techniques(desc)
        self.assertIn("bf16", techs)
        self.assertIn("flash_attention_2", techs)
        self.assertIn("fp8_quantization", techs)
        self.assertIn("torch_compile", techs)


class TestBuildSubmission(unittest.TestCase):
    def setUp(self):
        self.hw_path = _write_json(HARDWARE_JSON)
        self.cfg_path = _write_json(CONFIG_JSON)
        self.tsv_path = _write_text(TSV_CONTENT)

    def tearDown(self):
        for p in [self.hw_path, self.cfg_path, self.tsv_path]:
            if os.path.exists(p):
                os.unlink(p)

    def _build(self, contributor="tester", branch="test/branch"):
        from submit_run import build_submission
        return build_submission(
            contributor=contributor,
            branch=branch,
            tsv_path=self.tsv_path,
            hardware_path=self.hw_path,
            config_path=self.cfg_path,
        )

    def test_submission_has_required_keys(self):
        sub = self._build()
        for key in ("run_id", "submitted_at", "contributor", "hardware", "model", "results", "best_config"):
            self.assertIn(key, sub, f"Missing key: {key}")

    def test_contributor_stored(self):
        sub = self._build(contributor="my-handle")
        self.assertEqual(sub["contributor"], "my-handle")

    def test_best_tok_s_is_max_keep(self):
        sub = self._build()
        self.assertAlmostEqual(sub["results"]["best_tok_s"], 275.14)

    def test_baseline_tok_s_is_first_keep(self):
        sub = self._build()
        self.assertAlmostEqual(sub["results"]["baseline_tok_s"], 145.79)

    def test_gain_pct_is_correct(self):
        sub = self._build()
        expected = round((275.14 / 145.79 - 1) * 100, 1)
        self.assertAlmostEqual(sub["results"]["gain_pct"], expected, places=0)

    def test_experiment_counts(self):
        sub = self._build()
        r = sub["results"]
        self.assertEqual(r["total_experiments"], 5)
        self.assertEqual(r["keep_count"], 3)
        self.assertEqual(r["discard_count"], 1)
        self.assertEqual(r["crash_count"], 1)

    def test_hardware_fields(self):
        sub = self._build()
        hw = sub["hardware"]
        self.assertEqual(hw["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertTrue(hw["bf16_supported"])
        self.assertTrue(hw["fp8_supported"])
        self.assertEqual(hw["vram_total_gb"], 24.0)

    def test_model_fields(self):
        sub = self._build()
        m = sub["model"]
        self.assertEqual(m["id"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(m["params_b"], 0.5)
        self.assertIn("Qwen", m["family"])

    def test_run_id_contains_date(self):
        sub = self._build()
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.assertIn(today, sub["run_id"])

    def test_no_keeps_exits(self):
        """If there are no 'keep' experiments, build_submission should exit."""
        content = (
            "commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription\n"
            "abc\t0.0\t0.0\t0.0\tcrash\teverything crashed\n"
        )
        path = _write_text(content)
        try:
            from submit_run import build_submission
            with self.assertRaises(SystemExit):
                build_submission("anon", "branch", path, self.hw_path, self.cfg_path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# leaderboard.py tests
# ---------------------------------------------------------------------------

SAMPLE_RUN = {
    "run_id": "test-run-001",
    "submitted_at": "2026-03-22T18:41:00Z",
    "contributor": "tester",
    "branch": "test/branch",
    "hardware": {
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_family": "Ada Lovelace",
        "compute_capability": "8.9",
        "vram_total_gb": 24.0,
        "cuda_version": "12.4",
        "bf16_supported": True,
        "fp8_supported": True,
    },
    "model": {
        "id": "Qwen/Qwen2.5-0.5B",
        "params_b": 0.5,
        "family": "Qwen2.5",
    },
    "results": {
        "baseline_tok_s": 145.79,
        "best_tok_s": 275.14,
        "gain_pct": 88.7,
        "best_ttft_ms": 3.63,
        "best_vram_gb": 1.0,
        "total_experiments": 5,
        "keep_count": 3,
        "discard_count": 1,
        "crash_count": 1,
    },
    "best_config": {
        "description": "BF16 + flash_attention_2 + compile dynamic",
        "techniques": ["bf16", "flash_attention_2", "torch_compile"],
    },
    "experiments": [],
}

SAMPLE_RUN_2 = {
    **SAMPLE_RUN,
    "run_id": "test-run-002",
    "contributor": "user2",
    "hardware": {**SAMPLE_RUN["hardware"], "gpu_name": "NVIDIA H100 SXM5 80GB", "gpu_family": "Hopper"},
    "model": {**SAMPLE_RUN["model"], "id": "meta-llama/Llama-3.1-8B", "params_b": 8.0, "family": "Llama 3.1"},
    "results": {**SAMPLE_RUN["results"], "best_tok_s": 120.0, "baseline_tok_s": 60.0, "gain_pct": 100.0},
}


def _make_runs_dir(runs: list[dict]) -> str:
    d = tempfile.mkdtemp()
    for i, run in enumerate(runs):
        with open(os.path.join(d, f"run_{i}.json"), "w") as f:
            json.dump(run, f)
    return d


class TestLoadRuns(unittest.TestCase):
    def test_loads_json_files(self):
        from leaderboard import load_runs
        d = _make_runs_dir([SAMPLE_RUN, SAMPLE_RUN_2])
        try:
            runs = load_runs(d)
            self.assertEqual(len(runs), 2)
        finally:
            import shutil
            shutil.rmtree(d)

    def test_skips_invalid_json(self):
        from leaderboard import load_runs
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "valid.json"), "w") as f:
            json.dump(SAMPLE_RUN, f)
        with open(os.path.join(d, "bad.json"), "w") as f:
            f.write("not valid json {{{")
        try:
            runs = load_runs(d)
            self.assertEqual(len(runs), 1)
        finally:
            import shutil
            shutil.rmtree(d)

    def test_returns_empty_for_empty_dir(self):
        from leaderboard import load_runs
        d = tempfile.mkdtemp()
        try:
            runs = load_runs(d)
            self.assertEqual(runs, [])
        finally:
            import shutil
            shutil.rmtree(d)


class TestStatsSummary(unittest.TestCase):
    def test_correct_counts(self):
        from leaderboard import stats_summary
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        s = stats_summary(runs)
        self.assertEqual(s["total_runs"], 2)
        self.assertEqual(s["total_gpus"], 2)
        self.assertEqual(s["total_models"], 2)
        self.assertEqual(s["total_contributors"], 2)

    def test_top_tok_s(self):
        from leaderboard import stats_summary
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        s = stats_summary(runs)
        self.assertAlmostEqual(s["top_tok_s"], 275.14)

    def test_returns_empty_for_no_runs(self):
        from leaderboard import stats_summary
        self.assertEqual(stats_summary([]), {})


class TestBuildChartData(unittest.TestCase):
    def test_bar_lengths_match_run_count(self):
        from leaderboard import build_chart_data
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        chart = build_chart_data(runs)
        n = len(runs)
        self.assertEqual(len(chart["bar"]["labels"]), n)
        self.assertEqual(len(chart["bar"]["baseline"]), n)
        self.assertEqual(len(chart["bar"]["best"]), n)

    def test_bar_sorted_descending(self):
        from leaderboard import build_chart_data
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        chart = build_chart_data(runs)
        best = chart["bar"]["best"]
        self.assertGreaterEqual(best[0], best[-1])

    def test_scatter_has_all_runs(self):
        from leaderboard import build_chart_data
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        chart = build_chart_data(runs)
        self.assertEqual(len(chart["scatter"]), 2)

    def test_techniques_counted(self):
        from leaderboard import build_chart_data
        runs = [SAMPLE_RUN, SAMPLE_RUN_2]
        chart = build_chart_data(runs)
        labels = chart["techniques"]["labels"]
        self.assertGreater(len(labels), 0)
        # bf16 appears in both runs — should be in top techniques
        self.assertIn("bf16", labels)


class TestBuildTableRows(unittest.TestCase):
    def test_sorted_by_best_tok_s_desc(self):
        from leaderboard import build_table_rows
        rows = build_table_rows([SAMPLE_RUN, SAMPLE_RUN_2])
        self.assertGreaterEqual(rows[0]["best_tok_s"], rows[1]["best_tok_s"])

    def test_row_has_required_fields(self):
        from leaderboard import build_table_rows
        rows = build_table_rows([SAMPLE_RUN])
        r = rows[0]
        for field in ("run_id", "contributor", "gpu", "model", "best_tok_s", "gain_pct"):
            self.assertIn(field, r, f"Missing field: {field}")

    def test_gpu_name_stripped(self):
        from leaderboard import build_table_rows
        rows = build_table_rows([SAMPLE_RUN])
        # "NVIDIA GeForce RTX 4090" -> "RTX 4090"
        self.assertNotIn("NVIDIA", rows[0]["gpu"])
        self.assertNotIn("GeForce", rows[0]["gpu"])


class TestRenderHTML(unittest.TestCase):
    def test_renders_without_error(self):
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN, SAMPLE_RUN_2])
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)

    def test_html_contains_contributor(self):
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN])
        self.assertIn("tester", html)

    def test_html_contains_model_name(self):
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN])
        self.assertIn("Qwen2.5-0.5B", html)

    def test_html_contains_chart_js_cdn(self):
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN])
        self.assertIn("chart.js", html.lower())

    def test_no_template_placeholders_remain(self):
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN])
        self.assertNotIn("__GENERATED_AT__", html)
        self.assertNotIn("__STATS_CARDS__", html)
        self.assertNotIn("__TABLE_ROWS_JSON__", html)
        self.assertNotIn("__CHART_DATA_JSON__", html)

    def test_html_is_valid_json_data(self):
        """The embedded JSON data in the HTML must be parseable."""
        import re
        from leaderboard import render_html
        html = render_html([SAMPLE_RUN])
        # Extract ROWS JSON
        m = re.search(r"const ROWS = (.*?);", html)
        self.assertIsNotNone(m)
        rows = json.loads(m.group(1))
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
