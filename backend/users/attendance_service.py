"""Mark HR attendance when registered staff are recognized (webcam, CCTV, or manual)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dt_time, timedelta
from typing import Literal

from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from users.models import Attendance, Staff, User

logger = logging.getLogger(__name__)

AttendanceAction = Literal[
    "check_in",
    "check_out",
    "already_complete",
    "ignored",
    "skipped_cooldown",
    "skipped_no_staff",
    "skipped_not_enrolled",
]

_GENERIC_LABELS = frozenset({"unknown", "person", "face", ""})
_recent_camera_marks: dict[str, float] = {}


def _work_start() -> dt_time:
    raw = getattr(settings, "ATTENDANCE_WORK_START", "09:00")
    h, m = [int(x) for x in str(raw).split(":")[:2]]
    return dt_time(h, m)


def _work_end():
    raw = getattr(settings, "ATTENDANCE_WORK_END", "17:00")
    h, m = [int(x) for x in str(raw).split(":")[:2]]
    return dt_time(h, m)


def camera_attendance_zone(camera) -> str:
    """entry / exit / interior / mixed. Place names stay mixed so existing cameras keep last-seen checkout."""
    if camera is None:
        return "mixed"
    for raw in (getattr(camera, "passage_role", ""), getattr(camera, "zone", "")):
        token = (raw or "").strip().lower()
        if token in {"mixed", "entry", "exit", "interior"}:
            return token
    return "mixed"


def _late_after() -> dt_time:
    raw = getattr(settings, "ATTENDANCE_LATE_AFTER", "09:30")
    h, m = [int(x) for x in str(raw).split(":")[:2]]
    return dt_time(h, m)


def _min_checkout_after_in() -> timedelta:
    minutes = float(getattr(settings, "ATTENDANCE_MIN_CHECKOUT_AFTER_IN_MINUTES", 1))
    return timedelta(minutes=max(0.0, minutes))


def _camera_mark_cooldown() -> int:
    return max(30, int(getattr(settings, "ATTENDANCE_CAMERA_MARK_COOLDOWN_SECONDS", 120)))


def _min_checkout_seconds() -> int:
    hours = float(getattr(settings, "ATTENDANCE_MIN_CHECKOUT_HOURS", 4))
    return max(0, int(hours * 3600))


def _cooldown_key(*, user: User | None = None, staff: Staff | None = None) -> str | None:
    if user is not None:
        return f"user:{user.pk}"
    if staff is not None:
        return f"staff:{staff.pk}"
    return None


def _is_recognized_identity(identity: str) -> bool:
    lbl = (identity or "").strip()
    if not lbl:
        return False
    low = lbl.lower()
    if low in _GENERIC_LABELS or low.startswith("unknown"):
        return False
    if low.startswith(("gp", "go", "gv")) and low[2:].isdigit():
        return False
    if low.startswith("t") and low[1:].isdigit():
        return False
    return True


def resolve_staff_for_face_identity(identity: str) -> Staff | None:
    """Map identity label to a Staff row (full name, face label, or linked username)."""
    lbl = (identity or "").strip()
    if not _is_recognized_identity(lbl):
        return None

    return (
        Staff.objects.filter(
            Q(face_identity_label__iexact=lbl)
            | Q(full_name__iexact=lbl)
            | Q(user__username__iexact=lbl)
            | Q(employee_id__iexact=lbl)
        )
        .select_related("user")
        .first()
    )


def resolve_user_for_face_identity(identity: str) -> User | None:
    staff = resolve_staff_for_face_identity(identity)
    if staff and staff.user_id:
        linked = staff.user
        if linked and not linked.is_deleted:
            return linked

    lbl = (identity or "").strip()
    if not _is_recognized_identity(lbl):
        return None
    return User.objects.filter(username__iexact=lbl, is_deleted=False).first()


def staff_is_enrolled_for_attendance(staff: Staff | None) -> bool:
    """True when InsightFace enrollment is trained (preferred) or legacy photos exist."""
    if staff is None:
        return False
    try:
        enrollment = getattr(staff, "face_enrollment", None)
        if enrollment is not None and enrollment.is_trained and enrollment.embedding:
            return True
    except Exception:
        pass
    # Legacy fallback during migration — prefer InsightFace going forward
    from ml.face_sync import staff_has_face_embedding
    from users.staff_photos import staff_photo_paths

    if staff_has_face_embedding(staff):
        return True
    return bool(staff_photo_paths(staff))


def _touch_camera_cooldown(key: str) -> None:
    _recent_camera_marks[key] = time.monotonic()


def _camera_cooldown_active(key: str) -> bool:
    last = _recent_camera_marks.get(key, 0.0)
    return time.monotonic() - last < _camera_mark_cooldown()


_AUTO_NOTE_TAGS = ("Half day", "Early leave", "Overtime")


def _merge_auto_notes(existing: str | None, tags: list[str]) -> str:
    """Keep whatever HR wrote; replace only the engine's own tags (check-out repeats all day)."""
    parts = [p.strip() for p in (existing or "").split(";")]
    kept = [p for p in parts if p and p not in _AUTO_NOTE_TAGS]
    return "; ".join([*kept, *tags])


class AttendanceDecisionEngine:
    """
    First recognition of the day → check-in (present/late).
    Later recognition → update check-out (last seen).
    Ignore detections within MIN_CHECKOUT_AFTER_IN of check-in.
    """

    @classmethod
    def shift_start_for(cls, moment: datetime, staff: Staff | None = None) -> datetime:
        """Aware start of the shift this moment belongs to.

        Day shifts start on the moment's local date. A night shift (end before start,
        e.g. 22:00-06:00) belongs to whichever start is nearest, so 00:30 counts
        against yesterday's 22:00 and 21:50 against today's.
        """
        local = timezone.localtime(moment)
        tz = local.tzinfo
        shift_start = getattr(staff, "work_shift_start", None) if staff is not None else None
        shift_end = getattr(staff, "work_shift_end", None) if staff is not None else None
        start = shift_start or _work_start()
        today = timezone.make_aware(datetime.combine(local.date(), start), tz)
        if shift_start is None or shift_end is None or shift_end > shift_start:
            return today
        candidates = [today - timedelta(days=1), today, today + timedelta(days=1)]
        return min(candidates, key=lambda c: abs((local - c).total_seconds()))

    @classmethod
    def shift_length(cls, staff: Staff | None = None) -> timedelta:
        start = getattr(staff, "work_shift_start", None) if staff is not None else None
        end = getattr(staff, "work_shift_end", None) if staff is not None else None
        if start is None or end is None:
            start, end = _work_start(), _work_end()
        day = datetime(2000, 1, 1)
        length = datetime.combine(day, end) - datetime.combine(day, start)
        if length <= timedelta(0):
            length += timedelta(days=1)  # crosses midnight
        return length

    @classmethod
    def is_on_time(cls, check_in_time: datetime, staff: Staff | None = None) -> bool:
        # Grace = ATTENDANCE_LATE_AFTER - ATTENDANCE_WORK_START (30 min by default),
        # applied to the employee's own shift start when one is set.
        day = datetime(2000, 1, 1)
        grace = datetime.combine(day, _late_after()) - datetime.combine(day, _work_start())
        deadline = cls.shift_start_for(check_in_time, staff) + grace
        return check_in_time <= deadline

    @classmethod
    def determine_status(cls, check_in_time: datetime, staff: Staff | None = None) -> str:
        if cls.is_on_time(check_in_time, staff):
            return Attendance.STATUS_PRESENT
        return Attendance.STATUS_LATE

    @classmethod
    def half_day_hours(cls, staff: Staff | None = None) -> float:
        """ATTENDANCE_HALF_DAY_HOURS if set, else half the employee's shift, else 4 h."""
        configured = getattr(settings, "ATTENDANCE_HALF_DAY_HOURS", None)
        if configured not in (None, ""):
            try:
                return max(0.0, float(configured))
            except (TypeError, ValueError):
                pass
        has_shift = (
            staff is not None
            and getattr(staff, "work_shift_start", None) is not None
            and getattr(staff, "work_shift_end", None) is not None
        )
        if has_shift:
            return cls.shift_length(staff).total_seconds() / 3600.0 / 2
        return 4.0

    @classmethod
    def apply_checkout_rules(cls, record: Attendance, staff: Staff | None, now: datetime, *, zone: str = "mixed") -> None:
        if not record.check_in or not record.check_out:
            return
        hours = (record.check_out - record.check_in).total_seconds() / 3600.0
        # Full datetimes, so a night shift's end on the next calendar day compares correctly.
        shift_end = cls.shift_start_for(record.check_in, staff) + cls.shift_length(staff)
        tags: list[str] = []
        if zone == "exit" and hours < cls.half_day_hours(staff):
            record.status = Attendance.STATUS_HALF_DAY
            tags.append("Half day")
        else:
            if record.status == Attendance.STATUS_HALF_DAY:
                # A later check-out made it a full day again.
                record.status = cls.determine_status(record.check_in, staff)
            if zone == "exit" and record.check_out < shift_end:
                tags.append("Early leave")
        if record.check_out > shift_end:
            tags.append("Overtime")
        record.notes = _merge_auto_notes(record.notes, tags)

    @classmethod
    def _merge_duplicate_rows(cls, staff: Staff, user: User, today) -> Attendance | None:
        """Today has a staff-only row and a user-only row: fold the user row into the staff row."""
        with transaction.atomic():
            rows = list(
                Attendance.objects.select_for_update()
                .filter(Q(staff=staff) | Q(user=user), date=today)
                .order_by("pk")
            )
            staff_row = next((r for r in rows if r.staff_id == staff.pk), None)
            user_row = next((r for r in rows if r.user_id == user.pk and r is not staff_row), None)
            if staff_row is None or user_row is None:
                return None
            if user_row.staff_id not in (None, staff.pk):
                # The login's row belongs to another staff record: leave both alone.
                return staff_row
            if user_row.check_in and (not staff_row.check_in or user_row.check_in < staff_row.check_in):
                staff_row.check_in = user_row.check_in
                staff_row.check_in_confidence = user_row.check_in_confidence
                staff_row.status = user_row.status
                staff_row.source = user_row.source
            if user_row.check_out and (not staff_row.check_out or user_row.check_out > staff_row.check_out):
                staff_row.check_out = user_row.check_out
                staff_row.check_out_confidence = user_row.check_out_confidence
            if user_row.notes and user_row.notes not in (staff_row.notes or ""):
                staff_row.notes = "; ".join(p for p in (staff_row.notes, user_row.notes) if p)
            if not staff_row.image and user_row.image:
                staff_row.image = user_row.image.name
            if not staff_row.video and user_row.video:
                staff_row.video = user_row.video.name
            user_row.delete()  # frees (user, date) before the staff row takes the user
            staff_row.user = user
            staff_row.save()
            logger.info("Merged duplicate attendance rows for %s on %s", staff.full_name, today)
            return staff_row

    @classmethod
    def _get_or_create_today(cls, staff: Staff, source: str, now: datetime) -> tuple[Attendance, bool]:
        """Today's row for this staff member, and whether this call created it."""
        today = timezone.localdate(now)
        user = staff.user if staff.user_id and staff.user and not staff.user.is_deleted else None
        # Today's row may be keyed by staff only (marked before the account was linked) or by user only.
        match = Q(staff=staff) | Q(user=user) if user is not None else Q(staff=staff)
        for attempt in range(2):
            rows = list(Attendance.objects.filter(match, date=today).order_by("pk")[:2])
            if not rows:
                try:
                    with transaction.atomic():
                        return Attendance.objects.create(user=user, staff=staff, date=today, source=source), True
                except IntegrityError:
                    if attempt:
                        raise
                    continue  # created at the same moment by another camera; read it back
            if user is not None and len(rows) > 1:
                merged = cls._merge_duplicate_rows(staff, user, today)
                if merged is not None:
                    return merged, False
            # Prefer the row that already holds the user, so filling gaps never clashes.
            record = next((r for r in rows if user is not None and r.user_id == user.pk), rows[0])
            updates = []
            if record.staff_id is None:
                record.staff = staff
                updates.append("staff")
            if user is not None and record.user_id is None:
                record.user = user
                updates.append("user")
            if updates:
                record.save(update_fields=[*updates, "updated_at"])
            return record, False
        raise RuntimeError("unreachable")

    @classmethod
    def process_recognition(
        cls,
        staff: Staff,
        confidence: float = 0.0,
        source: str = Attendance.SOURCE_CCTV,
        now: datetime | None = None,
        *,
        allow_checkout: bool = True,
        enforce_camera_cooldown: bool = False,
        enforce_min_checkout_hours: bool = False,
        camera_zone: str = "mixed",
    ) -> dict:
        now = now or timezone.now()
        zone = (camera_zone or "mixed").strip().lower()
        if zone not in {"mixed", "entry", "exit", "interior"}:
            zone = "mixed"
        key = _cooldown_key(staff=staff) if not (staff.user_id) else _cooldown_key(user=staff.user)

        camera_like = source in (
            Attendance.SOURCE_CCTV,
            Attendance.SOURCE_CAMERA,
            "camera",
        )
        if enforce_camera_cooldown and camera_like and key and _camera_cooldown_active(key):
            return {
                "action": "skipped_cooldown",
                "message": "Camera mark cooldown active",
                "record": None,
            }

        record, created = cls._get_or_create_today(staff, source, now)

        if record.check_in is None:
            if zone == "exit":
                # Drop only the empty row this call just made; an existing row (manual
                # entry, notes, leave marker) is never deleted by a camera.
                if created and record.pk:
                    record.delete()
                return {
                    "action": "ignored",
                    "message": "Exit camera with no check-in",
                    "record": None,
                }
            record.check_in = now
            record.check_in_confidence = confidence
            record.status = cls.determine_status(now, staff)
            record.source = source
            record.save(
                update_fields=[
                    "check_in",
                    "check_in_confidence",
                    "status",
                    "source",
                    "updated_at",
                ]
            )
            if enforce_camera_cooldown and camera_like and key:
                _touch_camera_cooldown(key)
            logger.info("Attendance check-in: %s (%s)", staff.full_name, source)
            return {
                "action": "check_in",
                "message": f"Checked in at {timezone.localtime(now).strftime('%H:%M')}",
                "record": record,
            }

        if not allow_checkout:
            return {
                "action": "already_complete",
                "message": "Already checked in",
                "record": record,
            }

        if now - record.check_in < _min_checkout_after_in():
            return {
                "action": "ignored",
                "message": "Checked in just now — waiting before first check-out update",
                "record": record,
            }

        if zone in {"entry", "interior"}:
            return {
                "action": "ignored",
                "message": "Already checked in — entry/interior camera does not check out",
                "record": record,
            }

        if enforce_min_checkout_hours and camera_like:
            min_checkout = _min_checkout_seconds()
            if min_checkout > 0 and (now - record.check_in).total_seconds() < min_checkout:
                return {
                    "action": "skipped_cooldown",
                    "message": "Minimum hours before camera check-out not reached",
                    "record": record,
                }

        # Continuously update check-out (last seen) for CCTV/webcam
        record.check_out = now
        record.check_out_confidence = confidence
        cls.apply_checkout_rules(record, staff, now, zone=zone)
        record.save(update_fields=["check_out", "check_out_confidence", "status", "notes", "updated_at"])
        if enforce_camera_cooldown and camera_like and key:
            _touch_camera_cooldown(key)
        logger.info("Attendance check-out: %s (%s)", staff.full_name, source)
        return {
            "action": "check_out",
            "message": f"Check-out updated to {timezone.localtime(now).strftime('%H:%M')}",
            "record": record,
        }


def mark_attendance_for_staff(
    staff: Staff,
    *,
    source: str = "manual",
    allow_checkout: bool = True,
    confidence: float = 0.0,
) -> tuple[AttendanceAction, Attendance | None]:
    """Public helper used by kiosk/camera detection paths."""
    enforce_camera = source in ("camera", Attendance.SOURCE_CAMERA, Attendance.SOURCE_CCTV)
    decision = AttendanceDecisionEngine.process_recognition(
        staff=staff,
        confidence=confidence,
        source=source,
        allow_checkout=allow_checkout,
        enforce_camera_cooldown=enforce_camera,
        enforce_min_checkout_hours=source in ("camera", Attendance.SOURCE_CAMERA),
    )
    return decision["action"], decision["record"]  # type: ignore[return-value]


def mark_attendance_for_user(
    user: User,
    *,
    source: str = "manual",
    allow_checkout: bool = True,
    confidence: float = 0.0,
) -> tuple[AttendanceAction, Attendance | None]:
    staff = getattr(user, "staff_profile", None)
    if staff is None:
        # Create/update user-keyed attendance without staff
        now = timezone.now()
        today = timezone.localdate()
        key = _cooldown_key(user=user)
        if source == "camera" and key and _camera_cooldown_active(key):
            return "skipped_cooldown", None
        attendance, _ = Attendance.objects.get_or_create(
            user=user, date=today, defaults={"source": source}
        )
        if not attendance.check_in:
            attendance.check_in = now
            attendance.status = AttendanceDecisionEngine.determine_status(now)
            attendance.source = source
            attendance.check_in_confidence = confidence
            attendance.save(
                update_fields=["check_in", "status", "source", "check_in_confidence", "updated_at"]
            )
            if source == "camera" and key:
                _touch_camera_cooldown(key)
            return "check_in", attendance
        if not allow_checkout:
            return "already_complete", attendance
        if now - attendance.check_in < _min_checkout_after_in():
            return "ignored", attendance
        attendance.check_out = now
        attendance.check_out_confidence = confidence
        attendance.save(update_fields=["check_out", "check_out_confidence", "updated_at"])
        if source == "camera" and key:
            _touch_camera_cooldown(key)
        return "check_out", attendance
    return mark_attendance_for_staff(
        staff, source=source, allow_checkout=allow_checkout, confidence=confidence
    )


def _attendance_camera_purposes():
    from cameras.models import CameraPurpose

    purposes = {CameraPurpose.ATTENDANCE}
    if getattr(settings, "ATTENDANCE_MARK_ON_FACE_RECOGNITION_CAMERAS", True):
        purposes.add(CameraPurpose.FACE_RECOGNITION)
    return purposes


def _camera_allows_attendance(camera) -> bool:
    purposes = set(camera.purpose_list()) if hasattr(camera, "purpose_list") else {camera.purpose}
    if purposes & _attendance_camera_purposes():
        return True
    return getattr(settings, "ATTENDANCE_MARK_ON_ALL_CAMERAS", False)


