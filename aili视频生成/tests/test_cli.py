import os
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_dry_run_script_can_run_from_scripts_directory(self):
        env = os.environ.copy()
        env["DASHSCOPE_API_KEY"] = "sk-test"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_videos.py",
                "--config",
                "configs/jobs.basketball.example.yaml",
                "--dry-run",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fast_break_720P_16x9", result.stdout)
        self.assertIn("three_point_1080P_9x16", result.stdout)
        self.assertIn("street_dunk_1080P_1x1", result.stdout)


if __name__ == "__main__":
    unittest.main()
