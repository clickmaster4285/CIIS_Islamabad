"""Live inference workers: only streaming cameras are shared out, evenly."""

import threading
import unittest
from types import SimpleNamespace

from live_stream import LiveStreamManager


def manager(cams: dict[str, bool], workers: int = 4) -> SimpleNamespace:
    sessions = {k: SimpleNamespace(ip=k, stream=SimpleNamespace(connected=on)) for k, on in cams.items()}
    return SimpleNamespace(_infer_workers=workers, _lock=threading.Lock(), _sessions=sessions)


def shards(m) -> list[list[str]]:
    return [[s.ip for s in LiveStreamManager._sessions_for_worker(m, i)] for i in range(m._infer_workers)]


class WorkerShardTests(unittest.TestCase):
    def test_offline_cameras_get_no_worker(self):
        m = manager({"cam-20": False, "cam-21": False, "cam-22": False, "cam-27": True,
                     "cam-28": True, "cam-29": True, "cam-35": False})
        self.assertEqual(shards(m), [["cam-27"], ["cam-28"], ["cam-29"], []])

    def test_every_online_camera_is_covered_once(self):
        cams = {f"cam-{i:02d}": i % 3 != 0 for i in range(1, 30)}
        got = [k for shard in shards(manager(cams)) for k in shard]
        self.assertEqual(sorted(got), sorted(k for k, on in cams.items() if on))

    def test_reconnected_camera_is_picked_up(self):
        m = manager({"cam-27": True, "cam-28": False})
        self.assertEqual(shards(m)[:2], [["cam-27"], []])
        m._sessions["cam-28"].stream.connected = True
        self.assertEqual(shards(m)[:2], [["cam-27"], ["cam-28"]])

    def test_no_online_cameras(self):
        self.assertEqual(shards(manager({"cam-1": False})), [[], [], [], []])


if __name__ == "__main__":
    unittest.main()
