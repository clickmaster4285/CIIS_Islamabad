"""Enrich journey timelines from existing modules without modifying their logic."""

from __future__ import annotations

import logging

from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


def _get_or_create_staff_person(staff_id: int):
    from users.models import Staff

    from .models import JourneyPerson, PersonStatus, PersonType
    from .services import register_staff_journey_person

    staff = Staff.objects.filter(pk=staff_id).first()
    if staff is None:
        return None
    return register_staff_journey_person(staff)


@receiver(post_save, sender="visitors.ZoneAccessLog")
def on_zone_access_log(sender, instance, created, **kwargs):
    if not created:
        return
    try:
        with transaction.atomic():  # savepoint: a failure here must not break the caller's transaction
            _record_zone_access(instance)
    except Exception:
        logger.exception("Journey zone access hook failed")


def _record_zone_access(instance) -> None:
    from .models import JourneyEvent, JourneyEventType
    from .services import register_visitor_journey_person

    visitor = instance.visitor
    person = register_visitor_journey_person(visitor)

    event_type = (
        JourneyEventType.ZONE_EXIT if instance.scan_type == "exit" else JourneyEventType.ZONE_ENTRY
    )
    JourneyEvent.objects.create(
        journey_person=person,
        event_type=event_type,
        title=f"{'Exit' if instance.scan_type == 'exit' else 'Entry'} — {instance.zone}",
        description=instance.message or "",
        zone=instance.zone,
        gate=instance.gate or "",
        zone_access_log_id=instance.pk,
        metadata={
            "allowed": instance.allowed,
            "scan_type": instance.scan_type,
            "visitor_id": visitor.pk,
            "person_id": person.code,
        },
    )
    person.latest_zone = instance.zone
    person.latest_seen_at = instance.scanned_at
    person.save(update_fields=["latest_zone", "latest_seen_at", "updated_at"])


@receiver(post_save, sender="users.Attendance")
def on_attendance_record(sender, instance, created, **kwargs):
    try:
        with transaction.atomic():  # savepoint: a failure here must not break the attendance save
            _record_attendance(instance, created, kwargs.get("update_fields"))
    except Exception:
        logger.exception("Journey attendance hook failed")


def _record_attendance(instance, created, update_fields) -> None:
    from .models import JourneyEvent, JourneyEventType

    staff_id = instance.staff_id
    if not staff_id:
        return
    person = _get_or_create_staff_person(staff_id)
    if person is None:
        return

    # Prefer update_fields-aware journey events for check-in/out
    check_in_changed = created or (update_fields is None) or (
        update_fields is not None and "check_in" in update_fields
    )
    check_out_changed = created or (update_fields is None) or (
        update_fields is not None and "check_out" in update_fields
    )

    if instance.check_in and check_in_changed:
        if not JourneyEvent.objects.filter(
            journey_person=person,
            event_type=JourneyEventType.ATTENDANCE_CHECK_IN,
            attendance_id=instance.pk,
        ).exists():
            JourneyEvent.objects.create(
                journey_person=person,
                event_type=JourneyEventType.ATTENDANCE_CHECK_IN,
                title="Attendance Check-In",
                attendance_id=instance.pk,
                metadata={
                    "source": getattr(instance, "source", None) or "attendance_hook",
                    "status": getattr(instance, "status", None),
                },
            )

    if instance.check_out and check_out_changed:
        if not JourneyEvent.objects.filter(
            journey_person=person,
            event_type=JourneyEventType.ATTENDANCE_CHECK_OUT,
            attendance_id=instance.pk,
        ).exists():
            JourneyEvent.objects.create(
                journey_person=person,
                event_type=JourneyEventType.ATTENDANCE_CHECK_OUT,
                title="Attendance Check-Out",
                attendance_id=instance.pk,
                metadata={
                    "source": getattr(instance, "source", None) or "attendance_hook",
                    "status": getattr(instance, "status", None),
                },
            )


@receiver(post_save, sender="cameras.DetectionEvent")
def on_detection_event(sender, instance, created, **kwargs):
    """Bridge person detections into Person Journey (movement only — no security alerts)."""
    if not created:
        return
    try:
        from .bridge_detection import bridge_detection_event

        with transaction.atomic():  # savepoint: never poison the DetectionEvent save
            bridge_detection_event(instance)
    except Exception:
        logger.exception("Journey detection bridge failed")
    # Weapon / fire / smoke / panic alerts belong on AI Detection Alerts,
    # not on the person movement timeline.
