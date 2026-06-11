import unittest

from aliyun_video.models import PromptJob, VideoVariant, expand_jobs


class ModelTests(unittest.TestCase):
    def test_expand_jobs_creates_stable_basketball_variant_ids(self):
        jobs = [PromptJob(id="dunk", prompt="篮球运动员扣篮", duration=5)]
        variants = [
            VideoVariant(resolution="720P", ratio="16:9"),
            VideoVariant(resolution="1080P", ratio="9:16"),
        ]

        expanded = expand_jobs(jobs, variants)

        self.assertEqual([task.id for task in expanded], ["dunk_720P_16x9", "dunk_1080P_9x16"])
        self.assertEqual(expanded[0].prompt, "篮球运动员扣篮")
        self.assertEqual(expanded[1].duration, 5)

    def test_rejects_invalid_duration(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            PromptJob(id="bad", prompt="篮球", duration=2)

    def test_rejects_invalid_ratio(self):
        with self.assertRaisesRegex(ValueError, "ratio"):
            VideoVariant(resolution="720P", ratio="2:1")


if __name__ == "__main__":
    unittest.main()
