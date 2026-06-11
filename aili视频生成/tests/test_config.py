import unittest
from pathlib import Path

from aliyun_video.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_basketball_config_from_yaml(self):
        path = Path("tests/fixtures/basketball_jobs.yaml")

        config = load_config(path, env={"DASHSCOPE_API_KEY": "sk-test"})

        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(
            config.create_url,
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
        )
        self.assertEqual(config.query_base_url, "https://dashscope.aliyuncs.com/api/v1/tasks")
        self.assertEqual(config.rps, 20)
        self.assertEqual(config.jobs[0].id, "fast_break")
        self.assertFalse(config.jobs[0].watermark)

    def test_requires_api_key(self):
        path = Path("tests/fixtures/no_api_key.yaml")

        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            load_config(path, env={})

    def test_germany_requires_workspace_id(self):
        path = Path("tests/fixtures/germany_without_workspace.yaml")

        with self.assertRaisesRegex(ValueError, "workspace_id"):
            load_config(path, env={})


if __name__ == "__main__":
    unittest.main()
