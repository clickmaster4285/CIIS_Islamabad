"""Automatic watchlist / blacklist screening against VMS list records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from django.utils import timezone

from city_sync.identity import local_rows_q

from .models import SecurityAlert, Visitor, VmsListRecord

WATCHLIST_MODULE = "vms_watchlist_screening_rows"
BLACKLIST_MODULE = "vms_blacklist_management_rows"


def _normalize_id(value: str) -> str:
    return re.sub(r"\D", "", (value or "").strip())


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _name_ratio(a: str, b: str) -> float:
    na, nb = _normalize_name(a), _normalize_name(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _load_list_rows(module: str, location: str = "") -> list[dict[str, Any]]:
    # This server's own lists only (HQ copies of city lists do not screen HQ visitors).
    qs = VmsListRecord.objects.filter(local_rows_q(), module=module, is_deleted=False)
    if location:
        qs = qs.filter(location=location)
    return [dict(data or {}, id=record_id) for record_id, data in qs.values_list("record_id", "data")]


def _row_document(row: dict[str, Any]) -> str:
    for key in (
        "document_number",
        "document",
        "cnicOrBForm",
        "cnic_or_b_form",
        "cnic",
        "passport",
        "passport_number",
    ):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return ""


def _row_name(row: dict[str, Any]) -> str:
    for key in ("visitor", "name", "visitor_name", "full_name"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return ""


@dataclass
class ScreeningResult:
    status: str
    match_level: str
    score: int
    source: str
    message: str
    remarks: str
    block_entry: bool
    matched_record_id: str | None = None

    def as_extra(self) -> dict[str, Any]:
        return {
            "match": self.match_level,
            "score": self.score,
            "source": self.source,
            "status": self.status,
            "message": self.message,
            "remarks": self.remarks,
            "matched_record_id": self.matched_record_id,
            "screened_at": timezone.now().isoformat(),
        }


ScreeningLists = tuple  # (blacklist_rows, watchlist_rows)


def load_screening_lists(location: str = "") -> ScreeningLists:
    """Both lists for one location, loaded once (reuse across many visitors)."""
    return _load_list_rows(BLACKLIST_MODULE, location), _load_list_rows(WATCHLIST_MODULE, location)


def screen_visitor_data(
    *,
    full_name: str = "",
    cnic_number: str = "",
    passport_number: str = "",
    location: str = "",
    lists: ScreeningLists | None = None,
) -> ScreeningResult:
    cnic_norm = _normalize_id(cnic_number)
    passport_norm = _normalize_id(passport_number)
    name = (full_name or "").strip()
    blacklist_rows, watchlist_rows = lists if lists is not None else load_screening_lists(location)

    for row in blacklist_rows:
        doc = _row_document(row)
        doc_norm = _normalize_id(doc)
        row_name = _row_name(row)
        if doc_norm and (doc_norm == cnic_norm or doc_norm == passport_norm):
            reason = str(row.get("reason") or row.get("violation") or "Blacklisted").strip()
            return ScreeningResult(
                status="blacklisted",
                match_level="Yes",
                score=100,
                source="blacklist",
                message="Visitor matches blacklist.",
                remarks=reason or "Confirmed blacklist match.",
                block_entry=True,
                matched_record_id=str(row.get("id") or ""),
            )
        if name and row_name and _name_ratio(name, row_name) >= 0.92:
            reason = str(row.get("reason") or "Name match on blacklist").strip()
            return ScreeningResult(
                status="blacklisted",
                match_level="Yes",
                score=96,
                source="blacklist",
                message="Visitor name matches blacklist.",
                remarks=reason,
                block_entry=True,
                matched_record_id=str(row.get("id") or ""),
            )

    best_watch: ScreeningResult | None = None
    for row in watchlist_rows:
        doc = _row_document(row)
        doc_norm = _normalize_id(doc)
        row_name = _row_name(row)
        row_match = str(row.get("match") or "").strip()
        row_score_raw = str(row.get("score") or "0").replace("%", "").strip()
        try:
            row_score = int(float(row_score_raw))
        except ValueError:
            row_score = 0
        remarks = str(row.get("remarks") or "").strip()

        hit = False
        score = row_score
        match_level = row_match or "No"

        if doc_norm and (doc_norm == cnic_norm or doc_norm == passport_norm):
            hit = True
            score = max(score, 98)
            match_level = "Yes"
        elif name and row_name:
            ratio = _name_ratio(name, row_name)
            if ratio >= 0.92:
                hit = True
                score = max(score, int(ratio * 100))
                match_level = "Yes"
            elif ratio >= 0.78:
                hit = True
                score = max(score, int(ratio * 100))
                if match_level not in ("Yes", "Potential"):
                    match_level = "Potential"

        if not hit:
            continue

        candidate = ScreeningResult(
            status="flagged" if match_level == "Yes" else "potential",
            match_level=match_level,
            score=score,
            source="watchlist",
            message=(
                "Watchlist match detected."
                if match_level == "Yes"
                else "Potential watchlist match."
            ),
            remarks=remarks or f"Matched watchlist entry for {row_name or doc}.",
            block_entry=match_level == "Yes",
            matched_record_id=str(row.get("id") or ""),
        )
        if best_watch is None or candidate.score > best_watch.score:
            best_watch = candidate

    if best_watch:
        return best_watch

    return ScreeningResult(
        status="cleared",
        match_level="No",
        score=0,
        source="none",
        message="No watchlist or blacklist match.",
        remarks="Cleared by automated screening.",
        block_entry=False,
    )


def apply_visitor_screening(
    visitor: Visitor,
    *,
    create_alert: bool = True,
    lists: ScreeningLists | None = None,
    only_if_changed: bool = False,
) -> ScreeningResult:
    result = screen_visitor_data(
        full_name=visitor.full_name or "",
        cnic_number=visitor.cnic_number or visitor.cnic_passport or "",
        passport_number=visitor.passport_number or "",
        location=visitor.location or "",
        lists=lists,
    )

    previous = (visitor.extra_data or {}).get("screening") or {}
    unchanged = (
        visitor.watchlist_check_status == result.status
        and previous.get("matched_record_id") == result.matched_record_id
        and previous.get("score") == result.score
    )
    if not (only_if_changed and unchanged):
        extra = dict(visitor.extra_data or {})
        extra["screening"] = result.as_extra()
        visitor.watchlist_check_status = result.status
        visitor.extra_data = extra
        visitor.save(update_fields=["watchlist_check_status", "extra_data", "updated_at", "version"])

    if create_alert and result.status in ("blacklisted", "flagged"):
        alert_type = "watchlist_hit" if result.source == "watchlist" else "other"
        severity = "critical" if result.status == "blacklisted" else "high"
        SecurityAlert.objects.create(
            visitor=visitor,
            alert_type=alert_type,
            severity=severity,
            message=result.message + (f" {result.remarks}" if result.remarks else ""),
        )

    return result


def visitor_blocks_entry(visitor: Visitor) -> tuple[bool, str]:
    status = (visitor.watchlist_check_status or "").lower()
    if "blacklist" in status:
        return True, "Visitor is blacklisted."
    if status == "flagged":
        return True, "Visitor failed watchlist screening."
    return False, ""


def rescreen_all_visitors(*, location: str = "") -> int:
    """Re-screen this server's (non-deleted) visitors, optionally for one location.

    Each location's blacklist/watchlist is loaded once, not once per visitor, and a visitor row
    is only written when its screening outcome actually changed.
    """
    qs = Visitor.objects.filter(local_rows_q(), is_deleted=False).order_by("id")
    if location:
        qs = qs.filter(location=location)
    lists_by_location: dict[str, ScreeningLists] = {}
    count = 0
    for visitor in qs.iterator():
        loc = visitor.location or ""
        if loc not in lists_by_location:
            lists_by_location[loc] = load_screening_lists(loc)
        apply_visitor_screening(
            visitor, create_alert=False, lists=lists_by_location[loc], only_if_changed=True
        )
        count += 1
    return count


def _visitor_document_fields(visitor: Visitor) -> tuple[str, str, str]:
    cnic = (visitor.cnic_number or visitor.cnic_passport or "").strip()
    passport = (visitor.passport_number or "").strip()
    if cnic:
        return "CNIC", cnic, cnic
    if passport:
        return "Passport", passport, passport
    return "", "", ""


def _upsert_list_record(
    module: str,
    *,
    location: str,
    record_id: str,
    data: dict[str, Any],
) -> None:
    qs = VmsListRecord.objects.filter(local_rows_q(), module=module, record_id=record_id)
    if location:
        qs = qs.filter(location=location)
    existing = qs.first()
    if existing:
        merged = dict(existing.data or {})
        merged.update(data)
        existing.data = merged
        existing.save(update_fields=["data", "updated_at"])
    else:
        VmsListRecord.objects.create(
            module=module,
            record_id=record_id,
            data=data,
            location=location or "",
        )


def mark_visitor_flagged(
    visitor: Visitor,
    *,
    remarks: str = "",
    marked_by: str = "",
) -> ScreeningResult:
    doc_type, doc_number, _ = _visitor_document_fields(visitor)
    record_id = f"visitor-{visitor.pk}"
    if doc_number:
        record_id = f"doc-{_normalize_id(doc_number)}"

    _upsert_list_record(
        WATCHLIST_MODULE,
        location=visitor.location or "",
        record_id=record_id,
        data={
            "visitor": visitor.full_name or "",
            "phone": visitor.mobile_number or "",
            "location": str((visitor.extra_data or {}).get("residential_address") or visitor.location or ""),
            "document_type": doc_type,
            "document_number": doc_number,
            "match": "Yes",
            "score": "96%",
            "remarks": remarks or f"Manually flagged by {marked_by or 'security'}.",
            "flagged_by": marked_by or "security",
            "visitor_id": str(visitor.pk),
        },
    )
    result = apply_visitor_screening(visitor)
    rescreen_all_visitors(location=visitor.location or "")
    return result


def mark_visitor_blacklisted(
    visitor: Visitor,
    *,
    reason: str = "",
    marked_by: str = "",
) -> ScreeningResult:
    doc_type, doc_number, _ = _visitor_document_fields(visitor)
    record_id = f"visitor-{visitor.pk}"
    if doc_number:
        record_id = f"doc-{_normalize_id(doc_number)}"

    today = timezone.localdate().isoformat()
    _upsert_list_record(
        BLACKLIST_MODULE,
        location=visitor.location or "",
        record_id=record_id,
        data={
            "name": visitor.full_name or "",
            "phone": visitor.mobile_number or "",
            "document_type": doc_type or "CNIC",
            "document": doc_number,
            "document_number": doc_number,
            "restriction": "Permanent",
            "duration": "N/A",
            "reason": reason or f"Manually blacklisted by {marked_by or 'security'}.",
            "blacklisted_date": today,
            "expiry_date": "—",
            "blacklisted_by": marked_by or "Security",
            "visitor_id": str(visitor.pk),
        },
    )
    result = apply_visitor_screening(visitor)
    rescreen_all_visitors(location=visitor.location or "")
    return result
