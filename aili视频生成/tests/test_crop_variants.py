import unittest

from scripts.derive_crop_variants import VARIANTS, build_filter, parse_size_from_ffmpeg_output


class CropVariantTests(unittest.TestCase):
    def test_builds_center_crop_for_wide_16x9_from_square(self):
        variant = VARIANTS["16x9"]

        filter_text = build_filter(1080, 1080, variant)

        self.assertEqual(filter_text, "crop=1080:608:0:236,scale=1920:1080")

    def test_builds_center_crop_for_tall_9x16_from_square(self):
        variant = VARIANTS["9x16"]

        filter_text = build_filter(1080, 1080, variant)

        self.assertEqual(filter_text, "crop=608:1080:236:0,scale=1080:1920")

    def test_square_variant_keeps_full_frame(self):
        variant = VARIANTS["1x1"]

        filter_text = build_filter(1080, 1080, variant)

        self.assertEqual(filter_text, "crop=1080:1080:0:0,scale=1080:1080")

    def test_parses_video_size_from_ffmpeg_output(self):
        output = "Stream #0:0: Video: h264, yuv420p(progressive), 1080x1080, 24 fps"

        self.assertEqual(parse_size_from_ffmpeg_output(output), (1080, 1080))


if __name__ == "__main__":
    unittest.main()
