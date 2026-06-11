import unittest
from pathlib import Path

from aliyun_video.downloader import build_output_path, download_video
from aliyun_video.models import TaskRecord, VideoTask, VideoVariant


class DownloaderTests(unittest.TestCase):
    def test_build_output_path_contains_job_resolution_and_ratio(self):
        task = VideoTask(
            id="fast_break_1080P_9x16",
            job_id="fast_break",
            prompt="篮球快攻",
            variant=VideoVariant(resolution="1080P", ratio="9:16"),
            duration=5,
            watermark=False,
        )

        path = build_output_path(Path("outputs"), task)

        self.assertEqual(path, Path("outputs/fast_break_1080P_9x16.mp4"))

    def test_download_video_writes_transport_bytes(self):
        task = VideoTask(
            id="shot_720P_16x9",
            job_id="shot",
            prompt="篮球投篮",
            variant=VideoVariant(resolution="720P", ratio="16:9"),
            duration=5,
            watermark=True,
        )
        record = TaskRecord(
            local_id=task.id,
            task_id="task-1",
            status="SUCCEEDED",
            video_url="https://example.test/video.mp4",
        )

        path = download_video(Path("test_outputs"), task, record, fetch=lambda url: b"mp4-bytes")

        self.assertTrue(path.name.endswith(".mp4"))
        self.assertEqual(path.read_bytes(), b"mp4-bytes")
        path.unlink()
        path.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
