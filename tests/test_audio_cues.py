from eog_app.app import ASSET_ROOT, TEST_TARGETS


def test_live_prompts_have_the_legacy_audio_assets() -> None:
    assert [target[3] for target in TEST_TARGETS] == [
        "right.mp3",
        "backward.mp3",
        "left.mp3",
        "forward.mp3",
    ]
    assert all((ASSET_ROOT / target[3]).is_file() for target in TEST_TARGETS)
