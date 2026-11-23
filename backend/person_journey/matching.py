"""Label / visitor resolution helpers, plus the legacy find_best_match() wrapper around identity.identify()."""

from __future__ import annotations

from dataclasses import dataclass

from city_sync.identity import local_rows_q

from .models import JourneyPerson, PersonStatus, PersonType


@dataclass
class MatchCandidate:
    person: JourneyPerson
    face_score: float
    reid_score: float
    travel_score: float
    combined_score: float


def find_best_match(
    *,
    face_embedding: list[float] | None,
    reid_embedding: list[float] | None,
    camera_id: int | None,
    person_type_hint: str | None = None,
    staff_id: int | None = None,
    visitor_id: int | None = None,
) -> MatchCandidate | None:
    """Legacy entry point — delegates to the single calibrated engine in identity.identify()."""
    from .association import find_best_global_match

    person, detail = find_best_global_match(
        face_embedding=face_embedding,
        reid_embedding=reid_embedding,
        camera_id=camera_id,
        person_type_hint=person_type_hint,
        staff_id=staff_id,
        visitor_id=visitor_id,
    )
    if person is None or detail.get("decision") != "match":
        return None
    return MatchCandidate(
        person=person,
        face_score=float(detail.get("face_score") or 0.0),
        reid_score=float(detail.get("reid_score") or 0.0),
        travel_score=float(detail.get("topology_score") or 0.0),
        combined_score=float(detail.get("combined") or 0.0),
    )


def resolve_staff_from_face_label(label: str) -> tuple[int | None, str]:
    from users.models import Staff
    from django.db.models import Q

    lbl = (label or "").strip()
    low = lbl.lower()
    if not lbl or low in {"unknown", "person", "face", ""}:
        return None, ""
    if low.startswith(("gp", "go", "gv")) and low[2:].isdigit():
        return None, ""
    if low.startswith("t") and low[1:].isdigit():
        return None, ""
    if low.startswith("unknown"):
        return None, ""

    staff = (
        Staff.objects.filter(
            local_rows_q(),
            Q(face_identity_label__iexact=lbl)
            | Q(full_name__iexact=lbl)
            | Q(user__username__iexact=lbl)
        )
        .select_related("user")
        .first()
    )
    if staff is None:
        return None, lbl
    return staff.pk, (staff.full_name or lbl).strip()


def resolve_visitor_from_face_label(label: str) -> tuple[int | None, str]:
    """Map a recognized face label to an on-site visitor, if one exists."""
    from visitors.models import Visitor
    from django.db.models import Q

    lbl = (label or "").strip()
    low = lbl.lower()
    if not lbl or low in {"unknown", "person", "face", ""}:
        return None, ""
    if low.startswith(("gp", "go", "gv", "unknown")):
        return None, ""
    if low.startswith("t") and low[1:].isdigit():
        return None, ""

    visitor = Visitor.objects.filter(local_rows_q(), Q(full_name__iexact=lbl)).order_by("-id").first()
    if visitor is None:
        person = (
            JourneyPerson.objects.filter(
                local_rows_q(),
                person_type=PersonType.VISITOR,
                status=PersonStatus.ACTIVE,
                display_name__iexact=lbl,
            )
            .order_by("-latest_seen_at")
            .first()
        )
        if person and person.visitor_id:
            return person.visitor_id, person.display_name or lbl
        return None, lbl
    return visitor.pk, (visitor.full_name or lbl).strip()


def resolve_visitor_from_embedding(embedding) -> tuple[int | None, str, float]:
    """Match a probe embedding against the visitor face gallery (not staff attendance)."""
    from visitors.face_gallery import search_visitor_gallery

    hit = search_visitor_gallery(embedding)
    if not hit:
        return None, "", 0.0
    return (
        int(hit["visitor_id"]),
        str(hit.get("visitor_name") or ""),
        float(hit.get("confidence") or 0.0),
    )
