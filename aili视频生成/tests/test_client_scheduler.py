import unittest

from aliyun_video.client import HappyHorseClient
from aliyun_video.models import VideoTask, VideoVariant
from aliyun_video.scheduler import RateLimiter


class ClientSchedulerTests(unittest.TestCase):
    def test_create_task_sends_async_header_and_video_payload(self):
        calls = []

        def transport(method, url, headers=None, body=None):
            calls.append({"method": method, "url": url, "headers": headers, "body": body})
            return {"output": {"task_id": "task-123"}, "request_id": "req-1"}

        client = HappyHorseClient(
            api_key="sk-test",
            model="happyhorse-1.0-t2v",
            create_url="https://example.test/create",
            query_base_url="https://example.test/tasks",
            transport=transport,
        )
        task = VideoTask(
            id="dunk_1080P_9x16",
            job_id="dunk",
            prompt="篮球运动员飞身扣篮",
            variant=VideoVariant(resolution="1080P", ratio="9:16"),
            duration=5,
            watermark=False,
            seed=123,
        )

        record = client.create_task(task)

        self.assertEqual(record.task_id, "task-123")
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["headers"]["X-DashScope-Async"], "enable")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(calls[0]["body"]["model"], "happyhorse-1.0-t2v")
        self.assertEqual(calls[0]["body"]["input"]["prompt"], "篮球运动员飞身扣篮")
        self.assertEqual(calls[0]["body"]["parameters"]["resolution"], "1080P")
        self.assertEqual(calls[0]["body"]["parameters"]["ratio"], "9:16")
        self.assertEqual(calls[0]["body"]["parameters"]["duration"], 5)
        self.assertFalse(calls[0]["body"]["parameters"]["watermark"])
        self.assertEqual(calls[0]["body"]["parameters"]["seed"], 123)

    def test_query_task_extracts_status_and_video_url(self):
        def transport(method, url, headers=None, body=None):
            return {
                "request_id": "req-2",
                "output": {
                    "task_status": "SUCCEEDED",
                    "video_url": "https://cdn.example/video.mp4",
                },
            }

        client = HappyHorseClient(
            api_key="sk-test",
            model="happyhorse-1.0-t2v",
            create_url="https://example.test/create",
            query_base_url="https://example.test/tasks",
            transport=transport,
        )

        record = client.query_task("local-1", "task-123")

        self.assertEqual(record.local_id, "local-1")
        self.assertEqual(record.status, "SUCCEEDED")
        self.assertEqual(record.video_url, "https://cdn.example/video.mp4")

    def test_rate_limiter_sleeps_after_twenty_calls_in_one_second(self):
        now = [100.0]
        sleeps = []

        def monotonic():
            return now[0]

        def sleep(seconds):
            sleeps.append(round(seconds, 3))
            now[0] += seconds

        limiter = RateLimiter(rps=20, monotonic=monotonic, sleep=sleep)

        for _ in range(21):
            limiter.acquire()

        self.assertEqual(sleeps, [1.0])


if __name__ == "__main__":
    unittest.main()
