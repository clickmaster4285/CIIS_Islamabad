"""Resolve journey event snapshot URLs from stored path or linked DetectionEvent clip."""

from __future__ import annotations

from cameras.models import DetectionEvent

from .models import JourneyEvent


def detection_clip_url(detection_event_id: int | None) -> str:
    if not detection_event_id:
        return ""
    det = DetectionEvent.objects.filter(pk=detection_event_id).only("clip").first()
    if det is None or not det.clip:
        return ""
    return det.clip.url or ""


def journey_event_snapshot_url(event: JourneyEvent) -> str:
    """Only return images tied to this specific journey/detection event."""
    path = (event.snapshot_path or "").strip()
    if path:
        return path
    return detection_clip_url(event.detection_event_id)


def journey_person_latest_snapshot_url(person) -> str:
    events = list(
        JourneyEvent.objects.filter(journey_person=person)
        .exclude(snapshot_path="")
        .order_by("-created_at")
        .only("snapshot_path", "detection_event_id", "metadata")[:20]
    )
    for event in events:
        meta = event.metadata if isinstance(event.metadata, dict) else {}
        if meta.get("snapshot_kind") == "person_crop":
            url = journey_event_snapshot_url(event)
            if url:
                return url
    for event in events:
        url = journey_event_snapshot_url(event)
        if url:
            return url
    meta = person.metadata if isinstance(getattr(person, "metadata", None), dict) else {}
    thumb = str(meta.get("thumbnail_url") or "").strip()
    if thumb:
        return thumb
    event = (
        JourneyEvent.objects.filter(journey_person=person, detection_event_id__isnull=False)
        .order_by("-created_at")
        .only("snapshot_path", "detection_event_id")
        .first()
    )
    if event is None:
        return ""
    return journey_event_snapshot_url(event)


def build_detection_clip_map(events: list[JourneyEvent]) -> dict[int, str]:
    ids = [e.detection_event_id for e in events if e.detection_event_id]
    if not ids:
        return {}
    out: dict[int, str] = {}
    for row in DetectionEvent.objects.filter(pk__in=ids).only("clip"):
        if row.clip:
            out[row.pk] = row.clip.url or ""
    return out


def serializer_context_for_events(events: list[JourneyEvent], person=None) -> dict:
    del person
    return {"detection_clip_map": build_detection_clip_map(events)}


def backfill_snapshot_paths_for_person(person) -> int:
    """Copy clip URL only when journey event has no snapshot and clip matches that detection."""
    rows = JourneyEvent.objects.filter(
        journey_person=person,
        detection_event_id__isnull=False,
        snapshot_path="",
    ).only("pk", "detection_event_id")
    if not rows:
        return 0

    clip_map = build_detection_clip_map(
        [JourneyEvent(detection_event_id=r.detection_event_id) for r in rows if r.detection_event_id]
    )
    updated = 0
    for row in rows:
        url = clip_map.get(row.detection_event_id or 0, "")
        if url:
            JourneyEvent.objects.filter(pk=row.pk).update(snapshot_path=url)
            updated += 1
    return updated


def person_camera_captures(person, *, since=None) -> list[dict]:
    """Latest sighting image per camera — only event-specific snapshots."""
    event_qs = JourneyEvent.objects.filter(journey_person=person, camera__isnull=False)
    if since is not None:
        event_qs = event_qs.filter(created_at__gte=since)

    camera_ids = sorted({cid for cid in event_qs.values_list("camera_id", flat=True) if cid})
    if not camera_ids:
        return []

    scoped_events = JourneyEvent.objects.filter(journey_person=person, camera__isnull=False)
    if since is not None:
        scoped_events = scoped_events.filter(created_at__gte=since)

    clip_map = build_detection_clip_map(
        list(scoped_events.filter(detection_event_id__isnull=False).only("detection_event_id"))
    )

    results: list[dict] = []
    for camera_id in camera_ids:
        events = scoped_events.filter(camera_id=camera_id).select_related("camera").order_by(
            "-created_at"
        )
        snapshot_url = ""
        chosen = None
        crop_url = ""
        crop_event = None
        for ev in events:
            url = journey_event_snapshot_url(ev)
            if not url and ev.detection_event_id:
                url = clip_map.get(ev.detection_event_id, "")
            if not url:
                continue
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            if meta.get("snapshot_kind") == "person_crop" and crop_event is None:
                crop_url = url
                crop_event = ev
            if chosen is None:
                snapshot_url = url
                chosen = ev
            if crop_event is not None:
                break
        if crop_event is not None:
            chosen = crop_event
            snapshot_url = crop_url
        if chosen is None:
            chosen = events.first()
        if chosen is None:
            continue

        results.append(
            {
                "camera_id": camera_id,
                "camera_name": chosen.camera.name if chosen.camera else "",
                "camera_code": chosen.camera.code if chosen.camera else "",
                "zone": chosen.zone or (chosen.camera.zone if chosen.camera else ""),
                "snapshot_url": snapshot_url,
                "event_id": chosen.pk,
                "detection_event_id": chosen.detection_event_id,
                "event_type": chosen.event_type,
                "title": chosen.title,
                "confidence": chosen.confidence,
                "captured_at": chosen.created_at,
            }
        )

    results.sort(key=lambda row: row.get("camera_name") or "")
    return results


def annotate_person_list_extras(persons, *, tracklet_limit: int = 40, events_per_person: int = 20) -> list:
    """
    Precompute latest_snapshot_url / active_tracklet_count / tracklet_ids for a page of persons in a few
    queries (the serializer otherwise runs 1-3 queries per row). Mirrors journey_person_latest_snapshot_url.
    """
    from django.db.models import F, Window
    from django.db.models.functions import RowNumber

    from .models import CameraTrack

    persons = list(persons)
    ids = [p.pk for p in persons]
    if not ids:
        return persons

    events: dict[int, list] = {}
    rows = (
        JourneyEvent.objects.filter(journey_person_id__in=ids)
        .exclude(snapshot_path="")
        .annotate(rn=Window(RowNumber(), partition_by=[F("journey_person_id")], order_by=F("created_at").desc()))
        .filter(rn__lte=events_per_person)
        .values_list("journey_person_id", "snapshot_path", "metadata")
    )
    for pid, path, meta in rows:
        events.setdefault(pid, []).append((path, meta if isinstance(meta, dict) else {}))

    tracks: dict[int, list] = {}
    active: dict[int, int] = {}
    track_rows = (
        CameraTrack.objects.filter(journey_person_id__in=ids)
        .annotate(rn=Window(RowNumber(), partition_by=[F("journey_person_id")], order_by=F("started_at").desc()))
        .filter(rn__lte=tracklet_limit)
        .values_list("journey_person_id", "tracklet_id")
    )
    for pid, label in track_rows:
        label = (label or "").strip()
        if label and label not in tracks.setdefault(pid, []):
            tracks[pid].append(label)
    for pid in CameraTrack.objects.filter(journey_person_id__in=ids, status="active").values_list(
        "journey_person_id", flat=True
    ):
        active[pid] = active.get(pid, 0) + 1

    need_clip: list[int] = []
    for person in persons:
        url = ""
        rows_for = events.get(person.pk, [])
        for path, meta in rows_for:
            if meta.get("snapshot_kind") == "person_crop" and (path or "").strip():
                url = path.strip()
                break
        if not url:
            for path, _meta in rows_for:
                if (path or "").strip():
                    url = path.strip()
                    break
        if not url:
            meta = person.metadata if isinstance(getattr(person, "metadata", None), dict) else {}
            url = str(meta.get("thumbnail_url") or "").strip()
        if not url:
            need_clip.append(person.pk)
        person._latest_snapshot_url = url
        person._active_tracklet_count = active.get(person.pk, 0)
        person._tracklet_ids = tracks.get(person.pk, [])

    if need_clip:
        latest_det = {}
        for pid, det_id in (
            JourneyEvent.objects.filter(journey_person_id__in=need_clip, detection_event_id__isnull=False)
            .annotate(rn=Window(RowNumber(), partition_by=[F("journey_person_id")], order_by=F("created_at").desc()))
            .filter(rn=1)
            .values_list("journey_person_id", "detection_event_id")
        ):
            latest_det[pid] = det_id
        clip_map = build_detection_clip_map([JourneyEvent(detection_event_id=d) for d in latest_det.values()])
        by_pk = {p.pk: p for p in persons}
        for pid, det_id in latest_det.items():
            by_pk[pid]._latest_snapshot_url = clip_map.get(det_id, "")
    return persons
