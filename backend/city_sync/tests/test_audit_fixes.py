"""Regression tests for the 2026-11 sync audit fixes (ordering, retries, versions, media)."""

from __future__ import annotations

import uuid
from datetime import timedelta
from unittest import mock

import requests
from django.db import transaction
from django.test import TestCase, override_settings
from django.utils import timezone

from city_sync import worker
from city_sync.hq_receiver import RESULT_APPLIED, apply_event
from city_sync.models import (
    CityReferenceCache,
    HqConflict,
    HqMediaObject,
    HqReplicaRow,
    MediaQueue,
    ReferenceCursor,
    SyncOutbox,
)
from city_sync.outbox import enqueue_outbox, payload_hash, sync_update
from city_sync.tests.test_sites import CITY, HQ
from users.models import Attendance, Staff


def _row(entity_id, version, **extra) -> SyncOutbox:
    row = enqueue_outbox(
        entity_type="attendance",
        entity_id=entity_id,
        entity_version=version,
        operation="upsert",
        payload={"v": version},
        location="ABBOTTABAD",
    )
    SyncOutbox.objects.filter(pk=row.pk).update(created_at=timezone.now() - timedelta(minutes=5), **extra)
    return SyncOutbox.objects.get(pk=row.pk)


@override_settings(**{**CITY, "OUTBOX_MIN_AGE_SECONDS": 0})
class OutboxOrderingAndRetryTests(TestCase):
    def test_later_version_waits_while_earlier_backs_off(self):
        eid = uuid.uuid4()
        _row(eid, 1, next_attempt_at=timezone.now() + timedelta(minutes=10))
        later = _row(eid, 2)
        other = _row(uuid.uuid4(), 1)
        claimed = worker.claim_batch(limit=10, min_age_seconds=0)
        self.assertEqual([r.pk for r in claimed], [other.pk])
        self.assertEqual(SyncOutbox.objects.get(pk=later.pk).status, SyncOutbox.STATUS_PENDING)

    def test_due_versions_of_one_entity_go_together_in_order(self):
        eid = uuid.uuid4()
        a, b = _row(eid, 1), _row(eid, 2)
        claimed = worker.claim_batch(limit=10, min_age_seconds=0)
        self.assertEqual([r.pk for r in claimed], [a.pk, b.pk])
        self.assertTrue(all(r.claimed_at for r in SyncOutbox.objects.filter(pk__in=[a.pk, b.pk])))

    def test_dead_earlier_row_does_not_block(self):
        eid = uuid.uuid4()
        _row(eid, 1, status=SyncOutbox.STATUS_DEAD)
        later = _row(eid, 2)
        self.assertEqual([r.pk for r in worker.claim_batch(limit=10, min_age_seconds=0)], [later.pk])

    def test_network_errors_never_dead_letter(self):
        row = _row(uuid.uuid4(), 1, attempts=50)
        with mock.patch("city_sync.worker.requests.post", side_effect=requests.ConnectionError("down")):
            worker.push_once()
        row.refresh_from_db()
        self.assertEqual(row.status, SyncOutbox.STATUS_PENDING)
        self.assertIsNone(row.claimed_at)

    def test_rejected_dead_letters_and_unanswered_requeues(self):
        a, b = _row(uuid.uuid4(), 1), _row(uuid.uuid4(), 1)

        class R:
            status_code = 200
            content = b"x"
            text = ""

            def json(self):
                return {"results": [{"event_id": str(a.event_id), "result": "rejected", "detail": "bad"}]}

        with mock.patch("city_sync.worker.requests.post", return_value=R()):
            worker.push_once()
        self.assertEqual(SyncOutbox.objects.get(pk=a.pk).status, SyncOutbox.STATUS_DEAD)
        self.assertEqual(SyncOutbox.objects.get(pk=b.pk).status, SyncOutbox.STATUS_PENDING)

    def test_stuck_recovery_uses_claim_time(self):
        row = _row(uuid.uuid4(), 1, status=SyncOutbox.STATUS_SENDING, claimed_at=timezone.now())
        SyncOutbox.objects.filter(pk=row.pk).update(next_attempt_at=timezone.now() - timedelta(hours=1))
        self.assertEqual(worker.recover_stuck_sending(), 0)
        SyncOutbox.objects.filter(pk=row.pk).update(claimed_at=timezone.now() - timedelta(hours=1))
        self.assertEqual(worker.recover_stuck_sending(), 1)

    def test_requeue_dead_command(self):
        from django.core.management import call_command

        row = _row(uuid.uuid4(), 1, status=SyncOutbox.STATUS_DEAD, attempts=20)
        call_command("sync_requeue_dead", stdout=mock.MagicMock())
        row.refresh_from_db()
        self.assertEqual((row.status, row.attempts), (SyncOutbox.STATUS_PENDING, 0))


@override_settings(**CITY)
class VersionAndMediaSignalTests(TestCase):
    def setUp(self):
        self.staff = Staff.objects.create(full_name="V Test", employee_id="V-1")

    def test_update_fields_without_version_still_persists_the_bump(self):
        att = Attendance.objects.create(staff=self.staff, date=timezone.localdate(), status=Attendance.STATUS_PRESENT, source=Attendance.SOURCE_MANUAL)
        att.status = Attendance.STATUS_LATE
        att.save(update_fields=["status"])
        self.assertEqual(Attendance.objects.get(pk=att.pk).version, 2)
        att2 = Attendance.objects.get(pk=att.pk)
        att2.save()
        versions = list(
            SyncOutbox.objects.filter(entity_id=att.entity_id).order_by("seq").values_list("entity_version", flat=True)
        )
        self.assertEqual(versions, [1, 2, 3])

    def test_stale_instance_gets_a_new_version_not_a_duplicate(self):
        att = Attendance.objects.create(staff=self.staff, date=timezone.localdate(), status=Attendance.STATUS_PRESENT, source=Attendance.SOURCE_MANUAL)
        first = Attendance.objects.get(pk=att.pk)
        second = Attendance.objects.get(pk=att.pk)
        first.save()
        second.save()
        versions = list(
            SyncOutbox.objects.filter(entity_id=att.entity_id).order_by("seq").values_list("entity_version", flat=True)
        )
        self.assertEqual(versions, [1, 2, 3])

    def test_unchanged_file_is_not_queued_again(self):
        att = Attendance.objects.create(staff=self.staff, date=timezone.localdate(), status=Attendance.STATUS_PRESENT, source=Attendance.SOURCE_MANUAL)
        Attendance.objects.filter(pk=att.pk).update(image="attendance/a.jpg")
        att.refresh_from_db()
        with mock.patch("city_sync.media_sync.enqueue_media_file") as enqueue:
            att.status = Attendance.STATUS_LATE
            att.save()
            enqueue.assert_not_called()
            att.image = "attendance/b.jpg"
            att.save()
            enqueue.assert_called_once()

    def test_sync_update_bumps_version_and_enqueues(self):
        att = Attendance.objects.create(staff=self.staff, date=timezone.localdate(), status=Attendance.STATUS_PRESENT, source=Attendance.SOURCE_MANUAL)
        n = sync_update(Attendance.objects.filter(pk=att.pk), status=Attendance.STATUS_LATE)
        self.assertEqual(n, 1)
        self.assertEqual(Attendance.objects.get(pk=att.pk).version, 2)
        last = SyncOutbox.objects.filter(entity_id=att.entity_id).order_by("-seq").first()
        self.assertEqual((last.entity_version, last.payload["status"]), (2, Attendance.STATUS_LATE))


@override_settings(**HQ)
class HqReceiverAuditTests(TestCase):
    def _event(self, eid, version, entity_type="detention_memo", seq=1):
        payload = {"entity_id": str(eid), "v": version}
        return {
            "seq": seq,
            "event_id": str(uuid.uuid4()),
            "location_code": "ABBOTTABAD",
            "entity_type": entity_type,
            "entity_id": str(eid),
            "operation": "upsert",
            "entity_version": version,
            "schema_version": 1,
            "payload": payload,
            "payload_hash": payload_hash(payload),
        }

    def test_version_gap_on_full_state_payload_is_applied(self):
        eid = uuid.uuid4()
        with mock.patch("city_sync.hq_receiver._materialize"):
            apply_event(self._event(eid, 1), credential_location="ABBOTTABAD")
            result = apply_event(self._event(eid, 4, seq=2), credential_location="ABBOTTABAD")
        self.assertEqual(result["result"], RESULT_APPLIED)
        self.assertEqual(HqReplicaRow.objects.get(entity_id=eid).entity_version, 4)
        self.assertFalse(HqConflict.objects.filter(entity_id=eid, reason="version_gap").exists())

    def test_failed_materialization_is_retried(self):
        from city_sync.hq_receiver import retry_failed_materializations

        eid = uuid.uuid4()
        with mock.patch("city_sync.materialize.materialize_event", side_effect=RuntimeError("boom")):
            apply_event(self._event(eid, 1, entity_type="attendance"), credential_location="ABBOTTABAD")
        self.assertTrue(HqConflict.objects.filter(entity_id=eid, reason__startswith="materialize_failed").exists())
        with mock.patch("city_sync.materialize.materialize_event", return_value="created"):
            stats = retry_failed_materializations()
        self.assertEqual(stats["fixed"], 1)
        self.assertFalse(
            HqConflict.objects.filter(entity_id=eid, reason__startswith="materialize_failed", resolved_at__isnull=True).exists()
        )

    def test_media_id_of_another_city_is_refused_and_verified_never_changes(self):
        from city_sync.media_sync import hq_media_init

        mid = uuid.uuid4()
        HqMediaObject.objects.create(
            media_id=mid, location_code="ABBOTTABAD", sha256="a" * 64, size_bytes=3,
            status=HqMediaObject.STATUS_VERIFIED, source_path="x.jpg",
        )
        with self.assertRaises(PermissionError):
            hq_media_init(loc="DIKHAN", media_id=str(mid), sha256="b" * 64, size_bytes=3, entity_type="", entity_id=None)
        out = hq_media_init(
            loc="ABBOTTABAD", media_id=str(mid), sha256="b" * 64, size_bytes=3, entity_type="", entity_id=None,
            source_path="evil.html",
        )
        self.assertTrue(out["verified"])
        obj = HqMediaObject.objects.get(media_id=mid)
        self.assertEqual((obj.sha256, obj.source_path), ("a" * 64, "x.jpg"))

    def test_env_key_refused_once_site_has_a_managed_key(self):
        from rest_framework.test import APIClient

        from city_sync.models import SyncCredential

        with self.settings(SYNC_API_KEYS="ABBOTTABAD=env-key"):
            client = APIClient()
            ok = client.post("/api/sync/v1/heartbeat", {}, format="json", HTTP_AUTHORIZATION="Bearer env-key")
            self.assertEqual(ok.status_code, 200)
            SyncCredential.objects.create(location_code="ABBOTTABAD", key_hash=SyncCredential.hash_key("managed"))
            denied = client.post("/api/sync/v1/heartbeat", {}, format="json", HTTP_AUTHORIZATION="Bearer env-key")
            self.assertIn(denied.status_code, (401, 403))

    def test_events_endpoint_rejects_non_object_events(self):
        from rest_framework.test import APIClient

        with self.settings(SYNC_API_KEYS="ABBOTTABAD=env-key"):
            resp = APIClient().post(
                "/api/sync/v1/events", {"events": ["x"]}, format="json", HTTP_AUTHORIZATION="Bearer env-key"
            )
        self.assertEqual(resp.status_code, 400)


@override_settings(**HQ)
class DownlinkScheduleTests(TestCase):
    def test_rolled_back_transaction_does_not_block_a_later_one(self):
        from city_sync.downlink import _schedule

        calls = []
        try:
            with transaction.atomic():
                _schedule(("staff", 1), lambda: calls.append("first"))
                raise RuntimeError("rollback")
        except RuntimeError:
            pass
        with self.captureOnCommitCallbacks(execute=True):
            with transaction.atomic():
                _schedule(("staff", 1), lambda: calls.append("second"))
                _schedule(("staff", 1), lambda: calls.append("dupe"))
        self.assertEqual(calls, ["second"])


@override_settings(**CITY)
class ReferenceOverlapTests(TestCase):
    def test_overlap_window_and_version_dedupe(self):
        from city_sync import reference_sync

        since = timezone.now() - timedelta(hours=1)
        ReferenceCursor.objects.create(pk=1, since=since)
        CityReferenceCache.objects.create(
            entity_type="config", entity_key="a", version=3, payload={"x": 1}, updated_at=since
        )
        items = [
            {"entity_type": "config", "entity_key": "a", "version": 3, "payload": {"x": 2},
             "updated_at": since.isoformat()},
            {"entity_type": "config", "entity_key": "b", "version": 1, "payload": {"y": 1},
             "updated_at": since.isoformat()},
        ]

        class R:
            status_code = 200
            content = b"x"

            def json(self):
                return {"items": items}

        with mock.patch("city_sync.reference_sync.requests.get", return_value=R()) as get:
            out = reference_sync.pull_reference_once()
        sent_since = get.call_args.kwargs["params"]["since"]
        self.assertLess(sent_since, since.isoformat())
        self.assertEqual(out["applied"], 1)
        self.assertEqual(CityReferenceCache.objects.get(entity_key="a").payload, {"x": 1})


@override_settings(**CITY)
class MediaRequeueTests(TestCase):
    def test_requeue_failed_media(self):
        from city_sync.media_sync import requeue_failed_media

        MediaQueue.objects.create(location_code="ABBOTTABAD", path="a.jpg", sha256="x", status="failed", attempts=30)
        MediaQueue.objects.create(
            location_code="ABBOTTABAD", path="b.jpg", sha256="y", status="failed", last_error="local_file_missing"
        )
        self.assertEqual(requeue_failed_media(), 1)
        self.assertEqual(MediaQueue.objects.filter(status="pending", attempts=0).count(), 1)
