import csv
import subprocess
import sys
import unittest
from pathlib import Path

from aliyun_video.batch_variants import (
    BatchPrompt,
    MasterResult,
    build_master_task,
    load_prompt_csv,
    run_batch_variant_workflow,
    write_sample_18_rows,
)
from aliyun_video.variants import build_variant_matrix


class BatchVariantTests(unittest.TestCase):
    def _test_dir(self) -> Path:
        path = Path("outputs/test_batch_variants")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_csv(self, name: str, content: str) -> Path:
        path = self._test_dir() / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_prompt_csv_reads_required_and_optional_fields(self):
        path = self._write_csv(
            "prompts.csv",
            "id,prompt,duration,watermark,seed\n001,fast break,6,true,123\n",
        )

        prompts = load_prompt_csv(path)

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].id, "001")
        self.assertEqual(prompts[0].prompt, "fast break")
        self.assertEqual(prompts[0].duration, 6)
        self.assertTrue(prompts[0].watermark)
        self.assertEqual(prompts[0].seed, 123)

    def test_build_master_task_forces_1080p_square(self):
        prompt = BatchPrompt(id="001", prompt="fast break", duration=5, watermark=False, seed=123)

        task = build_master_task(prompt)

        self.assertEqual(task.id, "001_1080P_1x1")
        self.assertEqual(task.variant.resolution, "1080P")
        self.assertEqual(task.variant.ratio, "1:1")
        self.assertEqual(task.prompt, "fast break")

    def test_build_variant_matrix_returns_18_outputs(self):
        matrix = build_variant_matrix()

        self.assertEqual(len(matrix), 18)
        self.assertEqual({item.quality for item in matrix}, {"1080P", "720P"})
        self.assertIn(("16x9", "1080P", 1920, 1080), [(v.ratio_slug, v.quality, v.target_w, v.target_h) for v in matrix])
        self.assertIn(("16x9", "720P", 1280, 720), [(v.ratio_slug, v.quality, v.target_w, v.target_h) for v in matrix])
        self.assertIn(("9x16", "720P", 720, 1280), [(v.ratio_slug, v.quality, v.target_w, v.target_h) for v in matrix])

    def test_run_batch_variant_workflow_writes_18_success_rows(self):
        output_dir = self._test_dir() / "workflow_success"
        prompt = BatchPrompt(id="001", prompt="fast break", duration=5, watermark=False, seed=123)

        def generate_master(batch_prompt, task):
            master_path = output_dir / "masters" / f"{task.id}.mp4"
            master_path.parent.mkdir(parents=True, exist_ok=True)
            master_path.write_bytes(b"master")
            return MasterResult(task_id="task-001", status="SUCCEEDED", master_path=master_path)

        derived = []

        def derive_variant(master_path, output_path, variant):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"variant")
            derived.append((master_path, output_path, variant))

        results_path = run_batch_variant_workflow([prompt], output_dir, generate_master, derive_variant)

        with results_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(derived), 18)
        self.assertEqual(rows[0]["prompt_id"], "001")
        self.assertEqual(rows[0]["status"], "SUCCEEDED")
        self.assertEqual(rows[0]["aliyun_task_id"], "task-001")
        self.assertTrue(rows[0]["variant_video"].endswith(".mp4"))

    def test_run_batch_variant_workflow_records_failed_master(self):
        output_dir = self._test_dir() / "workflow_failed"
        prompt = BatchPrompt(id="bad", prompt="bad prompt")

        def generate_master(batch_prompt, task):
            return MasterResult(task_id="task-bad", status="FAILED", error="api failed")

        def derive_variant(master_path, output_path, variant):
            raise AssertionError("derive should not run for failed masters")

        results_path = run_batch_variant_workflow([prompt], output_dir, generate_master, derive_variant)

        with results_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "bad")
        self.assertEqual(rows[0]["status"], "FAILED")
        self.assertEqual(rows[0]["error"], "api failed")

    def test_generate_batch_variants_cli_dry_run_prints_18_planned_outputs(self):
        csv_path = self._write_csv("cli_prompts.csv", "id,prompt\n001,fast break\n")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_batch_variants.py",
                "--csv",
                str(csv_path),
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.strip()]), 18)
        self.assertIn("001_16x9_1080P.mp4", result.stdout)
        self.assertIn("001_9x16_720P.mp4", result.stdout)

    def test_write_sample_18_rows_creates_results_csv(self):
        output_dir = self._test_dir() / "sample_18"

        results_path = write_sample_18_rows(output_dir)

        with results_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0]["prompt_id"], "sample_001")
        self.assertEqual({row["status"] for row in rows}, {"SUCCEEDED"})


if __name__ == "__main__":
    unittest.main()
