from django.test import SimpleTestCase

from person_journey.identities import (
    format_person_code,
    format_tracklet_id,
    is_global_person_code,
    parse_person_code_seq,
)


class IdentityHelpersTests(SimpleTestCase):
    def test_global_person_code_format(self):
        self.assertEqual(format_person_code(42), "PJ-00042")
        self.assertEqual(format_person_code(1), "PJ-00001")
        self.assertTrue(is_global_person_code("PJ-00042"))
        self.assertFalse(is_global_person_code("P42"))
        self.assertFalse(is_global_person_code("C01-T18492"))

    def test_parse_legacy_and_pj_codes(self):
        self.assertEqual(parse_person_code_seq("PJ-00042"), 42)
        self.assertEqual(parse_person_code_seq("P42"), 42)
        self.assertEqual(parse_person_code_seq("V55"), 55)
        self.assertEqual(parse_person_code_seq("U300"), 300)

    def test_tracklet_id_is_not_person_id(self):
        self.assertEqual(format_tracklet_id(1, 18492), "C01-T18492")
        self.assertEqual(format_tracklet_id(2, 93281), "C02-T93281")
        self.assertEqual(format_tracklet_id(7, 11283), "C07-T11283")
        self.assertNotEqual(format_tracklet_id(1, 42), format_person_code(42))
        self.assertEqual(format_tracklet_id(None, 1), "")
        self.assertEqual(format_tracklet_id(1, None), "")


class TopologyScoreTests(SimpleTestCase):
    """Only a too-fast hop is impossible; a slow one just scores low."""

    edges = {
        (1, 2): {
            "from_camera_id": 1,
            "to_camera_id": 2,
            "typical_min_sec": 20,
            "typical_max_sec": 60,
            "hard_max_sec": 180,
            "bidirectional": True,
        }
    }

    def score(self, a, b, sec):
        from person_journey.association import topology_travel_score

        return topology_travel_score(a, b, sec, edges=self.edges)

    def test_typical_hop_scores_full(self):
        self.assertEqual(self.score(1, 2, 30), (1.0, False))

    def test_too_fast_is_impossible(self):
        self.assertTrue(self.score(1, 2, 2)[1])

    def test_slow_hop_is_possible_but_low(self):
        score, impossible = self.score(1, 2, 3600)
        self.assertFalse(impossible)
        self.assertLess(score, 0.2)

    def test_reverse_edge_used_when_bidirectional(self):
        self.assertEqual(self.score(2, 1, 30), (1.0, False))

    def test_no_topology_slow_hop_not_excluded(self):
        score, impossible = self.score(3, 4, 10_000)
        self.assertFalse(impossible)
        self.assertGreater(score, 0)


class SpansConflictTests(SimpleTestCase):
    def test_same_camera_overlap_conflicts(self):
        from datetime import datetime, timedelta, timezone as tz

        from person_journey.identity import spans_conflict

        t0 = datetime(2026, 1, 1, tzinfo=tz.utc)
        a = [(1, t0, t0 + timedelta(seconds=30))]
        b = [(1, t0 + timedelta(seconds=10), t0 + timedelta(seconds=40))]
        c = [(1, t0 + timedelta(seconds=60), t0 + timedelta(seconds=90))]
        self.assertTrue(spans_conflict(a, b))
        self.assertFalse(spans_conflict(a, c))


class SnapshotRecaptureRuleTests(SimpleTestCase):
    def test_unlabeled_full_frame_retried_once(self):
        from person_journey.snapshot_capture import _meta_needs_recapture

        self.assertTrue(_meta_needs_recapture("", {}))
        self.assertTrue(_meta_needs_recapture("/m/a.jpg", {"snapshot_kind": "full_frame"}))
        self.assertFalse(
            _meta_needs_recapture("/m/a.jpg", {"snapshot_kind": "full_frame", "recapture_attempted": True})
        )
        self.assertFalse(_meta_needs_recapture("/m/a.jpg", {"snapshot_kind": "full_frame", "person_labeled": True}))
        self.assertFalse(_meta_needs_recapture("", {"snapshot_unavailable": True}))
        self.assertFalse(_meta_needs_recapture("", {"snapshot_purged": True}))
