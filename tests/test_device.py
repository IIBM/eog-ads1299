import numpy as np

from eog_app.device import SerialReader


def test_serial_stream_sync_finds_float_frame_boundary() -> None:
    frames = np.tile(np.array([[123.5, -276.25]], dtype="<f4"), (100, 1))
    raw_stream = b"\xaa\xbb\xcc" + frames.tobytes()

    assert SerialReader._alignment_offset(raw_stream, channels=2) == 3
