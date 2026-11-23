from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.db import transaction
from django.db.models import Count, Prefetch, Q
from django.db.models.functions import TruncDate, TruncMonth, TruncWeek
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import APIView

from detentions.access import (
    DocumentWritePermission,
    can_full_edit,
    can_write_documents,
    is_local_row,
    readonly_copy_response,
    scoped,
)
from detentions.models import DetentionMemo

from .models import (
    AssessmentNotification,
    DetentionAssessment,
    DetentionAssessmentAttachment,
    NoteSheet,
    NoteSheetAttachment,
    NoteSheetItem,
    NoteSheetItemImage,
    NoteSheetNotification,
    RecoveryMemo,
    RecoveryNotification,
    SeizureReport,
)
from .notifications import (
    RECOVERY_FORWARD_TO_LABEL,
    mark_assessment_notifications_resolved,
    mark_note_sheet_notifications_resolved,
    mark_recovery_notifications_resolved,
    notify_assessment_submitted,
    notify_recovery_submitted,
    prime_submitter_locations,
    user_can_approve_assessment,
    user_can_approve_note_sheet,
    user_can_approve_recovery,
    user_can_delete_note_sheet,
)
from .serializers import (
    AssessmentApprovalSerializer,
    AssessmentWriteSerializer,
    LinkDetentionSerializer,
    NoteSheetApprovalSerializer,
    NoteSheetWriteSerializer,
    RecoveryApprovalSerializer,
    RecoveryMemoWriteSerializer,
    SeizureReportWriteSerializer,
    apply_assessment,
    apply_note_sheet,
    apply_recovery,
    apply_seizure_report,
    assessment_to_dict,
    body_from_request,
    build_recovery_assessment_sheet,
    maybe_create_deposit_for_recovery,
    note_sheet_to_dict,
    recovery_memo_to_dict,
    save_assessment_uploads,
    save_note_sheet_goods_images,
    save_note_sheet_uploads,
    seizure_report_to_dict,
    validate_seizure_uploads,
)

logger = logging.getLogger(__name__)

GENERIC_SAVE_ERROR = "The request could not be completed because of a server error. Please try again."


class _SeizureAPIView(APIView):
    """Unexpected errors are logged and answered with a generic 500 (no internals leaked)."""

    permission_classes = [permissions.IsAuthenticated]

    def handle_exception(self, exc):
        if isinstance(exc, (APIException, Http404, DjangoPermissionDenied)):
            return super().handle_exception(exc)
        logger.exception("[seizure_management] %s %s failed", self.request.method, self.request.path)
        return Response({"detail": GENERIC_SAVE_ERROR}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _username(request) -> str:
    user = getattr(request, "user", None)
    if user and getattr(user, "is_authenticated", False):
        return user.get_username() or ""
    return ""


def _actor_display(request) -> str:
    """Approver name always comes from the logged-in user, never from the body."""
    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return ""
    return (getattr(user, "full_name", None) or "").strip() or user.get_username() or ""


def _forbidden(detail: str) -> Response:
    return Response({"detail": detail}, status=status.HTTP_403_FORBIDDEN)


def _soft_delete(obj) -> None:
    obj.is_deleted = True
    obj.save()


# --------------------------------------------------------------------------- querysets


def _note_sheet_base_qs():
    return NoteSheet.objects.filter(is_deleted=False).prefetch_related(
        Prefetch(
            "items",
            queryset=NoteSheetItem.objects.filter(is_deleted=False).select_related(
                "located_camera__nvr__site",
                "located_camera__ml_server",
                "detection_event",
            ),
        ),
        Prefetch("items__images", queryset=NoteSheetItemImage.objects.filter(is_deleted=False)),
        Prefetch("attachments", queryset=NoteSheetAttachment.objects.filter(is_deleted=False)),
    )


def _note_sheets(user):
    return scoped(_note_sheet_base_qs(), user)


def _assessments(user):
    return scoped(
        DetentionAssessment.objects.filter(is_deleted=False)
        .select_related("detention_memo")
        .prefetch_related(
            Prefetch(
                "attachments",
                queryset=DetentionAssessmentAttachment.objects.filter(is_deleted=False),
            )
        ),
        user,
    )


def _recoveries(user):
    return scoped(
        RecoveryMemo.objects.filter(is_deleted=False).select_related(
            "detention_memo", "assessment", "deposit_account"
        ),
        user,
    )


def _reports(user):
    return scoped(
        SeizureReport.objects.filter(is_deleted=False).select_related(
            "detention_memo", "assessment", "recovery_memo"
        ),
        user,
    )


def _memos(user):
    return scoped(DetentionMemo.objects.filter(is_deleted=False), user)


def _writable_memo(request, memo_id):
    """(memo, error response). The memo must be visible to the user and owned by this server."""
    memo = get_object_or_404(_memos(request.user), pk=memo_id)
    return memo, readonly_copy_response(memo)


# --------------------------------------------------------------------------- notifications dicts


def _notification_to_dict(n: NoteSheetNotification) -> dict:
    sheet = n.note_sheet
    return {
        "id": str(n.id),
        "title": n.title,
        "message": n.message,
        "isRead": n.is_read,
        "createdAt": n.created_at.isoformat() if n.created_at else "",
        "noteSheetId": str(n.note_sheet_id),
        "noteSheetNo": (sheet.note_sheet_no or sheet.reference_number or "") if sheet else "",
        "assessmentId": "",
        "recoveryMemoId": "",
        "type": "note_sheet_approval",
        "hrefKind": "note_sheet",
    }


def _assessment_notification_to_dict(n: AssessmentNotification) -> dict:
    assessment = n.assessment
    memo = assessment.detention_memo if assessment else None
    return {
        "id": str(n.id),
        "title": n.title,
        "message": n.message,
        "isRead": n.is_read,
        "createdAt": n.created_at.isoformat() if n.created_at else "",
        "noteSheetId": "",
        "noteSheetNo": "",
        "assessmentId": str(n.assessment_id),
        "recoveryMemoId": "",
        "caseNo": (memo.case_no if memo else "") or "",
        "type": "assessment_approval",
        "hrefKind": "assessment",
    }


def _recovery_notification_to_dict(n: RecoveryNotification) -> dict:
    recovery = n.recovery_memo
    memo = recovery.detention_memo if recovery else None
    return {
        "id": str(n.id),
        "title": n.title,
        "message": n.message,
        "isRead": n.is_read,
        "createdAt": n.created_at.isoformat() if n.created_at else "",
        "noteSheetId": "",
        "noteSheetNo": "",
        "assessmentId": "",
        "recoveryMemoId": str(n.recovery_memo_id),
        "caseNo": (memo.case_no if memo else "") or "",
        "type": "recovery_approval",
        "hrefKind": "recovery",
    }


# --------------------------------------------------------------------------- note sheets


class NoteSheetListAPIView(_SeizureAPIView):
    def get(self, request):
        status_filter = (request.query_params.get("status") or "").strip()
        qs = _note_sheets(request.user)
        if status_filter:
            qs = qs.filter(status=status_filter)
        if request.query_params.get("available") == "1":
            qs = qs.filter(status=NoteSheet.STATUS_APPROVED, detention_memo__isnull=True)
        rows = list(qs)
        # One user query for every submitter location instead of one per row.
        loc_map = prime_submitter_locations(o.created_by for o in rows)
        return Response([note_sheet_to_dict(o, request, loc_map=loc_map) for o in rows])


def _parse_iso_date(raw: str | None) -> date | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _period_date(value, group: str = "day") -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        d = timezone.localtime(value).date() if timezone.is_aware(value) else value.date()
    elif isinstance(value, date):
        d = value
    else:
        return None
    if group == "week":
        d = d - timedelta(days=d.weekday())
    elif group == "month":
        d = date(d.year, d.month, 1)
    return d


def _period_label(group: str, period: date) -> str:
    if group == "month":
        return period.strftime("%B %Y")
    if group == "week":
        end = period + timedelta(days=6)
        return f"{period.strftime('%d %b')} – {end.strftime('%d %b %Y')}"
    return period.strftime("%d %b %Y")


def _iter_periods(group: str, start: date, end: date) -> list[date]:
    out: list[date] = []
    if group == "day":
        cursor = start
        while cursor <= end:
            out.append(cursor)
            cursor += timedelta(days=1)
            if len(out) >= 400:
                break
    elif group == "week":
        cursor = start - timedelta(days=start.weekday())
        while cursor <= end:
            out.append(cursor)
            cursor += timedelta(days=7)
            if len(out) >= 120:
                break
    else:
        cursor = date(start.year, start.month, 1)
        while cursor <= end:
            out.append(cursor)
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)
            if len(out) >= 120:
                break
    return out


class NoteSheetCreatedReportAPIView(_SeizureAPIView):
    """Day / week / month counts of note sheets created, with optional custom dates."""

    def get(self, request):
        group = (request.query_params.get("group") or "day").strip().lower()
        if group not in ("day", "week", "month"):
            group = "day"

        today = timezone.localdate()
        date_from = _parse_iso_date(request.query_params.get("date_from"))
        date_to = _parse_iso_date(request.query_params.get("date_to"))
        month = (request.query_params.get("month") or "").strip()

        if month and len(month) >= 7 and date_from is None and date_to is None:
            try:
                year, month_no = int(month[:4]), int(month[5:7])
                date_from = date(year, month_no, 1)
                date_to = _month_end(year, month_no)
            except ValueError:
                pass

        if date_from is None and date_to is None:
            if group == "month":
                date_from = date(today.year, 1, 1)
                date_to = today
            elif group == "week":
                date_from = today - timedelta(days=83)
                date_to = today
            else:
                date_from = today - timedelta(days=29)
                date_to = today
        if date_from is None:
            date_from = date_to or today
        if date_to is None:
            date_to = today
        if date_from > date_to:
            date_from, date_to = date_to, date_from

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(date_from, datetime.min.time()), tz)
        end_dt = timezone.make_aware(datetime.combine(date_to + timedelta(days=1), datetime.min.time()), tz)

        all_sheets = scoped(NoteSheet.objects.filter(is_deleted=False), request.user)
        qs = all_sheets.filter(created_at__gte=start_dt, created_at__lt=end_dt)
        trunc = {
            "day": TruncDate("created_at", tzinfo=tz),
            "week": TruncWeek("created_at", tzinfo=tz),
            "month": TruncMonth("created_at", tzinfo=tz),
        }[group]

        buckets = {
            _period_date(row["period"], group): row
            for row in qs.annotate(period=trunc)
            .values("period")
            .annotate(
                count=Count("id"),
                draft=Count("id", filter=Q(status=NoteSheet.STATUS_DRAFT)),
                submitted=Count("id", filter=Q(status=NoteSheet.STATUS_SUBMITTED)),
                approved=Count("id", filter=Q(status=NoteSheet.STATUS_APPROVED)),
                rejected=Count("id", filter=Q(status=NoteSheet.STATUS_REJECTED)),
            )
            if _period_date(row["period"], group) is not None
        }

        series = []
        for period in _iter_periods(group, date_from, date_to):
            row = buckets.get(period) or {}
            series.append(
                {
                    "period": period.isoformat(),
                    "label": _period_label(group, period),
                    "count": int(row.get("count") or 0),
                    "draft": int(row.get("draft") or 0),
                    "submitted": int(row.get("submitted") or 0),
                    "approved": int(row.get("approved") or 0),
                    "rejected": int(row.get("rejected") or 0),
                }
            )

        week_start = today - timedelta(days=today.weekday())
        month_start = date(today.year, today.month, 1)
        summary = {
            "allTime": all_sheets.count(),
            "today": all_sheets.filter(created_at__date=today).count(),
            "thisWeek": all_sheets.filter(created_at__date__gte=week_start, created_at__date__lte=today).count(),
            "thisMonth": all_sheets.filter(created_at__date__gte=month_start, created_at__date__lte=today).count(),
        }

        rows = [
            {
                "id": str(obj.id),
                "noteSheetNo": obj.note_sheet_no or obj.reference_number or "",
                "caseNo": obj.case_no or "",
                "status": obj.status or "",
                "priority": obj.priority or "",
                "preparedBy": obj.prepared_by or "",
                "accusedName": obj.accused_name or "",
                "subject": obj.subject or "",
                "createdAt": obj.created_at.isoformat() if obj.created_at else "",
            }
            for obj in qs.order_by("-created_at")[:1000]
        ]

        by_status = {
            "Draft": qs.filter(status=NoteSheet.STATUS_DRAFT).count(),
            "Submitted": qs.filter(status=NoteSheet.STATUS_SUBMITTED).count(),
            "Approved": qs.filter(status=NoteSheet.STATUS_APPROVED).count(),
            "Rejected": qs.filter(status=NoteSheet.STATUS_REJECTED).count(),
        }

        return Response(
            {
                "group": group,
                "dateFrom": date_from.isoformat(),
                "dateTo": date_to.isoformat(),
                "total": qs.count(),
                "byStatus": by_status,
                "summary": summary,
                "series": series,
                "rows": rows,
            }
        )


def _mark_note_sheet_submitted(obj: NoteSheet, *, submitted_by_user_id: int | None) -> NoteSheet:
    """Move draft/rejected sheet into the approval queue and notify AC/DC/Location Admin."""
    from .notifications import NOTE_SHEET_FORWARD_TO_LABEL, notify_note_sheet_submitted

    if obj.status not in (NoteSheet.STATUS_DRAFT, NoteSheet.STATUS_REJECTED):
        return obj
    obj.forward_to = NOTE_SHEET_FORWARD_TO_LABEL
    obj.forward_to_user_id = None
    obj.status = NoteSheet.STATUS_SUBMITTED
    obj.submitted_at = timezone.now()
    obj.rejection_reason = ""
    obj.approval_remarks = ""
    obj.viewed_at = None
    obj.save(
        update_fields=[
            "forward_to",
            "forward_to_user_id",
            "status",
            "submitted_at",
            "rejection_reason",
            "approval_remarks",
            "viewed_at",
            "updated_at",
        ]
    )
    notify_note_sheet_submitted(obj, submitted_by_user_id=submitted_by_user_id)
    return obj


def _wants_send_for_approval(body: dict, validated: dict | None = None) -> bool:
    """sendForApproval / submitForApproval flag, or status="Submitted" (Draft -> Submitted only)."""
    if validated and validated.get("status") == NoteSheet.STATUS_SUBMITTED:
        return True
    raw = body.get("sendForApproval", body.get("submitForApproval"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes")
    return False


_EDITABLE_NOTE_SHEET = (NoteSheet.STATUS_DRAFT, NoteSheet.STATUS_REJECTED)


class NoteSheetCreateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def post(self, request):
        body = body_from_request(request)
        ser = NoteSheetWriteSerializer(data=body)
        ser.is_valid(raise_exception=True)
        send_for_approval = _wants_send_for_approval(body, ser.validated_data)
        validate_seizure_uploads(request)
        with transaction.atomic():
            obj = NoteSheet()
            apply_note_sheet(obj, ser.validated_data, username=_username(request))
            save_note_sheet_uploads(request, obj)
            save_note_sheet_goods_images(request, obj, ser.validated_data.get("items"))
            if send_for_approval:
                _mark_note_sheet_submitted(obj, submitted_by_user_id=getattr(request.user, "id", None))
        obj = _note_sheet_base_qs().get(pk=obj.pk)
        return Response(note_sheet_to_dict(obj, request), status=status.HTTP_201_CREATED)


class NoteSheetReadAPIView(_SeizureAPIView):
    def get(self, request, pk):
        obj = get_object_or_404(_note_sheets(request.user), pk=pk)
        if (
            obj.status == NoteSheet.STATUS_SUBMITTED
            and not obj.viewed_at
            and is_local_row(obj)
            and user_can_approve_note_sheet(request.user, obj)
        ):
            obj.viewed_at = timezone.now()
            obj.save(update_fields=["viewed_at", "updated_at"])
        return Response(note_sheet_to_dict(obj, request))


class NoteSheetUpdateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def put(self, request, pk):
        obj = get_object_or_404(_note_sheets(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        body = body_from_request(request)
        ser = NoteSheetWriteSerializer(data=body, partial=True)
        ser.is_valid(raise_exception=True)
        send_for_approval = _wants_send_for_approval(body, ser.validated_data)
        validate_seizure_uploads(request)
        with transaction.atomic():
            obj = NoteSheet.objects.select_for_update().get(pk=obj.pk)
            if obj.status not in _EDITABLE_NOTE_SHEET and not can_full_edit(request.user):
                return Response(
                    {"detail": "Only draft or rejected note sheets can be edited."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            apply_note_sheet(obj, ser.validated_data, username=_username(request))
            save_note_sheet_uploads(request, obj)
            save_note_sheet_goods_images(request, obj, ser.validated_data.get("items"))
            if send_for_approval:
                obj.refresh_from_db()
                _mark_note_sheet_submitted(obj, submitted_by_user_id=getattr(request.user, "id", None))
        obj = _note_sheet_base_qs().get(pk=obj.pk)
        return Response(note_sheet_to_dict(obj, request))


class NoteSheetDeleteAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def delete(self, request, pk):
        obj = get_object_or_404(_note_sheets(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        if obj.detention_memo_id:
            return Response(
                {"detail": "Cannot delete note sheet linked to a detention memo."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not user_can_delete_note_sheet(request.user, obj):
            return _forbidden("Only higher officials can delete an approved note sheet.")
        _soft_delete(obj)
        return Response(status=status.HTTP_204_NO_CONTENT)


class NoteSheetApprovalAPIView(_SeizureAPIView):
    def post(self, request, pk):
        obj = get_object_or_404(_note_sheets(request.user), pk=pk)
        ser = NoteSheetApprovalSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        action = ser.validated_data["action"]
        remarks = (
            ser.validated_data.get("approvalRemarks")
            or ser.validated_data.get("rejectionReason")
            or ""
        )
        if not is_local_row(obj):
            if action == "view":
                return Response(note_sheet_to_dict(obj, request))
            return readonly_copy_response(obj)

        with transaction.atomic():
            # Row lock: two approvers (or approve + edit) cannot both act on the same status.
            locked = NoteSheet.objects.select_for_update().get(pk=obj.pk)
            if action == "submit":
                if not can_write_documents(request.user):
                    return _forbidden("Your role cannot submit note sheets.")
                if locked.status not in _EDITABLE_NOTE_SHEET:
                    return Response({"detail": "Only draft/rejected sheets can be submitted."}, status=400)
                # Auto-route to Assistant Collector, Deputy Collector, Location Admin, Super Admin
                _mark_note_sheet_submitted(locked, submitted_by_user_id=getattr(request.user, "id", None))
            elif action == "view":
                if (
                    locked.status == NoteSheet.STATUS_SUBMITTED
                    and not locked.viewed_at
                    and user_can_approve_note_sheet(request.user, locked)
                ):
                    locked.viewed_at = timezone.now()
                    locked.save(update_fields=["viewed_at", "updated_at"])
            elif action in ("approve", "reject"):
                verb = "approved" if action == "approve" else "rejected"
                if locked.status != NoteSheet.STATUS_SUBMITTED:
                    return Response({"detail": f"Only submitted sheets can be {verb}."}, status=400)
                if not user_can_approve_note_sheet(request.user, locked):
                    return _forbidden(
                        "Only an Assistant Collector, Deputy Collector or Location Admin of the "
                        "submitter's location, or the Super Admin, can decide this note sheet "
                        "(never the officer who created it)."
                    )
                locked.status = (
                    NoteSheet.STATUS_APPROVED if action == "approve" else NoteSheet.STATUS_REJECTED
                )
                locked.approved_by = _actor_display(request)
                locked.approved_at = timezone.now()
                locked.approval_remarks = remarks
                locked.rejection_reason = "" if action == "approve" else remarks
                locked.save(
                    update_fields=[
                        "status",
                        "approved_by",
                        "approved_at",
                        "approval_remarks",
                        "rejection_reason",
                        "updated_at",
                    ]
                )
                mark_note_sheet_notifications_resolved(locked)
        obj = _note_sheet_base_qs().get(pk=obj.pk)
        return Response(note_sheet_to_dict(obj, request))


class NoteSheetLinkDetentionAPIView(_SeizureAPIView):
    """Link an approved note sheet to a newly created detention memo (one-time)."""

    permission_classes = [DocumentWritePermission]

    def post(self, request, pk):
        obj = get_object_or_404(_note_sheets(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        ser = LinkDetentionSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        memo, denied = _writable_memo(request, ser.validated_data["detentionMemoId"])
        if denied is not None:
            return denied
        with transaction.atomic():
            locked = NoteSheet.objects.select_for_update().get(pk=obj.pk)
            DetentionMemo.objects.select_for_update().filter(pk=memo.pk).first()
            if locked.status != NoteSheet.STATUS_APPROVED:
                return Response(
                    {"detail": "Note sheet must be approved before creating a detention memo."},
                    status=400,
                )
            if locked.detention_memo_id:
                return Response({"detail": "Note sheet already linked to a detention memo."}, status=400)
            if NoteSheet.objects.filter(detention_memo=memo).exists():
                return Response(
                    {"detail": "Detention memo already linked to another note sheet."},
                    status=400,
                )
            locked.detention_memo = memo
            locked.save(update_fields=["detention_memo", "updated_at"])
        obj = _note_sheet_base_qs().get(pk=obj.pk)
        return Response(note_sheet_to_dict(obj, request))


# --------------------------------------------------------------------------- assessments


_EDITABLE_ASSESSMENT = (DetentionAssessment.STATUS_DRAFT, DetentionAssessment.STATUS_REJECTED)


def _assessment_response(request, pk, status_code=status.HTTP_200_OK):
    obj = _assessments(request.user).get(pk=pk)
    return Response(assessment_to_dict(obj, request), status=status_code)


class AssessmentListAPIView(_SeizureAPIView):
    def get(self, request):
        memo_id = (request.query_params.get("detentionMemoId") or "").strip()
        status_filter = (request.query_params.get("status") or "").strip()
        qs = _assessments(request.user)
        if memo_id:
            qs = qs.filter(detention_memo_id=memo_id)
        if status_filter:
            qs = qs.filter(status=status_filter)
        return Response([assessment_to_dict(o, request) for o in qs])


class AssessmentCreateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def post(self, request):
        body = body_from_request(request)
        ser = AssessmentWriteSerializer(data=body)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        memo_id = data.get("detentionMemoId")
        if not memo_id:
            return Response({"detail": "detentionMemoId is required."}, status=400)
        memo, denied = _writable_memo(request, memo_id)
        if denied is not None:
            return denied
        validate_seizure_uploads(request)
        with transaction.atomic():
            # Lock the memo so two requests cannot both create its one assessment.
            DetentionMemo.objects.select_for_update().get(pk=memo.pk)
            if DetentionAssessment.objects.filter(detention_memo=memo, is_deleted=False).exists():
                return Response(
                    {"detail": "Assessment already exists for this detention memo."},
                    status=400,
                )
            obj = DetentionAssessment(detention_memo=memo, status=DetentionAssessment.STATUS_DRAFT)
            apply_assessment(obj, data, username=_username(request))
            save_assessment_uploads(request, obj)
        return _assessment_response(request, obj.pk, status.HTTP_201_CREATED)


class AssessmentReadAPIView(_SeizureAPIView):
    def get(self, request, pk):
        obj = get_object_or_404(_assessments(request.user), pk=pk)
        if (
            obj.status == DetentionAssessment.STATUS_SUBMITTED
            and not obj.viewed_at
            and is_local_row(obj)
            and user_can_approve_assessment(request.user, obj)
        ):
            obj.viewed_at = timezone.now()
            obj.save(update_fields=["viewed_at", "updated_at"])
        return Response(assessment_to_dict(obj, request))


class AssessmentUpdateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def put(self, request, pk):
        obj = get_object_or_404(_assessments(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        body = body_from_request(request)
        ser = AssessmentWriteSerializer(data=body, partial=True)
        ser.is_valid(raise_exception=True)
        validate_seizure_uploads(request)
        with transaction.atomic():
            locked = DetentionAssessment.objects.select_for_update().get(pk=obj.pk)
            if locked.status not in _EDITABLE_ASSESSMENT and not can_full_edit(request.user):
                return Response(
                    {"detail": "Only draft or rejected assessments can be edited."},
                    status=400,
                )
            apply_assessment(locked, ser.validated_data, username=_username(request))
            save_assessment_uploads(request, locked)
        return _assessment_response(request, obj.pk)


class AssessmentDeleteAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def delete(self, request, pk):
        obj = get_object_or_404(_assessments(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        if obj.status == DetentionAssessment.STATUS_APPROVED:
            return Response({"detail": "Cannot delete an approved assessment."}, status=400)
        _soft_delete(obj)
        return Response(status=status.HTTP_204_NO_CONTENT)


class AssessmentApprovalAPIView(_SeizureAPIView):
    def post(self, request, pk):
        obj = get_object_or_404(_assessments(request.user), pk=pk)
        ser = AssessmentApprovalSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        action = ser.validated_data["action"]
        remarks = (
            ser.validated_data.get("approvalRemarks")
            or ser.validated_data.get("rejectionReason")
            or ""
        )
        if not is_local_row(obj):
            if action == "view":
                return Response(assessment_to_dict(obj, request))
            return readonly_copy_response(obj)

        with transaction.atomic():
            locked = DetentionAssessment.objects.select_for_update().get(pk=obj.pk)
            if action == "submit":
                if not can_write_documents(request.user):
                    return _forbidden("Your role cannot submit assessments.")
                if locked.status not in _EDITABLE_ASSESSMENT:
                    return Response({"detail": "Only draft/rejected assessments can be submitted."}, status=400)
                if not (locked.examining_officer or "").strip():
                    return Response({"detail": "Examining officer is required before submit."}, status=400)
                if locked.document_relevance == DetentionAssessment.RELEVANCE_PENDING:
                    return Response(
                        {"detail": "Set document relevance (Relevant / Not Relevant) before submit."},
                        status=400,
                    )
                locked.status = DetentionAssessment.STATUS_SUBMITTED
                locked.submitted_at = timezone.now()
                locked.rejection_reason = ""
                locked.approval_remarks = ""
                locked.viewed_at = None
                locked.updated_by = _username(request) or locked.updated_by
                locked.save(
                    update_fields=[
                        "status",
                        "submitted_at",
                        "rejection_reason",
                        "approval_remarks",
                        "viewed_at",
                        "updated_by",
                        "updated_at",
                    ]
                )
                notify_assessment_submitted(locked, submitted_by_user_id=getattr(request.user, "id", None))
            elif action == "view":
                if (
                    locked.status == DetentionAssessment.STATUS_SUBMITTED
                    and not locked.viewed_at
                    and user_can_approve_assessment(request.user, locked)
                ):
                    locked.viewed_at = timezone.now()
                    locked.save(update_fields=["viewed_at", "updated_at"])
            elif action in ("approve", "reject"):
                verb = "approved" if action == "approve" else "rejected"
                if locked.status != DetentionAssessment.STATUS_SUBMITTED:
                    return Response({"detail": f"Only submitted assessments can be {verb}."}, status=400)
                if not user_can_approve_assessment(request.user, locked):
                    return _forbidden(
                        "Only an Assistant Collector, Deputy Collector or Location Admin of the "
                        "submitter's location, or the Super Admin, can decide this assessment "
                        "(never the officer who created it)."
                    )
                locked.status = (
                    DetentionAssessment.STATUS_APPROVED
                    if action == "approve"
                    else DetentionAssessment.STATUS_REJECTED
                )
                locked.approved_by = _actor_display(request)
                locked.approved_at = timezone.now()
                locked.approval_remarks = remarks
                locked.rejection_reason = "" if action == "approve" else remarks
                locked.save(
                    update_fields=[
                        "status",
                        "approved_by",
                        "approved_at",
                        "approval_remarks",
                        "rejection_reason",
                        "updated_at",
                    ]
                )
                mark_assessment_notifications_resolved(locked)
        return _assessment_response(request, obj.pk)


# --------------------------------------------------------------------------- recovery memos


_EDITABLE_RECOVERY = (RecoveryMemo.STATUS_DRAFT, RecoveryMemo.STATUS_REJECTED)


def _assessment_for_memo(request, memo, assessment_id):
    """Assessment must exist, be visible and belong to ``memo`` (else 400)."""
    if not assessment_id:
        return None, None
    assessment = _assessments(request.user).filter(pk=assessment_id).first()
    if assessment is None:
        return None, Response({"detail": "Assessment not found."}, status=400)
    if assessment.detention_memo_id != memo.pk:
        return None, Response({"detail": "Assessment does not belong to this detention memo."}, status=400)
    return assessment, None


def _recovery_payload(request) -> dict:
    data = request.data
    if isinstance(data, dict):
        return dict(data)
    try:
        return dict(data)
    except Exception:  # noqa: BLE001
        return {}


class RecoveryMemoListAPIView(_SeizureAPIView):
    def get(self, request):
        memo_id = (request.query_params.get("detentionMemoId") or "").strip()
        qs = _recoveries(request.user)
        if memo_id:
            qs = qs.filter(detention_memo_id=memo_id)
        return Response([recovery_memo_to_dict(o) for o in qs])


class RecoveryMemoCreateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def post(self, request):
        ser = RecoveryMemoWriteSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        memo, denied = _writable_memo(request, data["detentionMemoId"])
        if denied is not None:
            return denied
        assessment, error = _assessment_for_memo(request, memo, data.get("assessmentId"))
        if error is not None:
            return error
        with transaction.atomic():
            obj = RecoveryMemo(detention_memo=memo, assessment=assessment)
            apply_recovery(obj, data, username=_username(request))
            if data.get("createDeposit"):
                maybe_create_deposit_for_recovery(obj)
        obj = _recoveries(request.user).get(pk=obj.pk)
        return Response(recovery_memo_to_dict(obj), status=status.HTTP_201_CREATED)


class RecoveryMemoReadAPIView(_SeizureAPIView):
    def get(self, request, pk):
        obj = get_object_or_404(_recoveries(request.user), pk=pk)
        return Response(recovery_memo_to_dict(obj))


class RecoveryMemoUpdateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def put(self, request, pk):
        obj = get_object_or_404(_recoveries(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        payload = _recovery_payload(request)
        # The memo of a recovery never changes.
        payload["detentionMemoId"] = str(obj.detention_memo_id)
        ser = RecoveryMemoWriteSerializer(data=payload)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        if "assessmentId" in data:
            _assessment, error = _assessment_for_memo(request, obj.detention_memo, data.get("assessmentId"))
            if error is not None:
                return error
        with transaction.atomic():
            locked = RecoveryMemo.objects.select_for_update().get(pk=obj.pk)
            if locked.approval_status not in _EDITABLE_RECOVERY and not can_full_edit(request.user):
                return Response(
                    {"detail": "Only draft or rejected recovery memos can be edited."},
                    status=400,
                )
            apply_recovery(locked, data, username=_username(request))
            if data.get("createDeposit"):
                maybe_create_deposit_for_recovery(locked)
        obj = _recoveries(request.user).get(pk=obj.pk)
        return Response(recovery_memo_to_dict(obj))


class RecoveryMemoDeleteAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def delete(self, request, pk):
        obj = get_object_or_404(_recoveries(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        if obj.approval_status == RecoveryMemo.STATUS_APPROVED:
            return Response({"detail": "Cannot delete an approved recovery memo."}, status=400)
        _soft_delete(obj)
        return Response(status=status.HTTP_204_NO_CONTENT)


class RecoveryMemoApprovalAPIView(_SeizureAPIView):
    def post(self, request, pk):
        obj = get_object_or_404(_recoveries(request.user), pk=pk)
        ser = RecoveryApprovalSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        action = ser.validated_data["action"]
        remarks = (
            ser.validated_data.get("approvalRemarks")
            or ser.validated_data.get("rejectionReason")
            or ""
        ).strip()
        if not is_local_row(obj):
            if action == "view":
                return Response(recovery_memo_to_dict(obj))
            return readonly_copy_response(obj)

        with transaction.atomic():
            locked = RecoveryMemo.objects.select_for_update().get(pk=obj.pk)
            if action == "view":
                if (
                    locked.approval_status == RecoveryMemo.STATUS_PENDING
                    and not locked.viewed_at
                    and user_can_approve_recovery(request.user, locked)
                ):
                    locked.viewed_at = timezone.now()
                    locked.save(update_fields=["viewed_at", "updated_at"])
            elif action == "submit":
                if not can_write_documents(request.user):
                    return _forbidden("Your role cannot submit recovery memos.")
                if locked.approval_status not in _EDITABLE_RECOVERY:
                    return Response(
                        {"detail": "Only draft/rejected recovery memos can be submitted."},
                        status=400,
                    )
                if not (locked.recovery_officer or "").strip():
                    return Response({"detail": "Recovery officer is required before submit."}, status=400)
                locked.approval_status = RecoveryMemo.STATUS_PENDING
                locked.submitted_at = timezone.now()
                locked.rejection_reason = ""
                locked.approval_remarks = ""
                locked.save(
                    update_fields=[
                        "approval_status",
                        "submitted_at",
                        "rejection_reason",
                        "approval_remarks",
                        "updated_at",
                    ]
                )
                notify_recovery_submitted(locked, submitted_by_user_id=getattr(request.user, "id", None))
            elif action in ("approve", "reject"):
                verb = "approved" if action == "approve" else "rejected"
                if locked.approval_status != RecoveryMemo.STATUS_PENDING:
                    return Response({"detail": f"Only pending recovery memos can be {verb}."}, status=400)
                if not user_can_approve_recovery(request.user, locked):
                    return _forbidden(
                        f"Only {RECOVERY_FORWARD_TO_LABEL} of the submitter's location can decide this "
                        "recovery memo (never the officer who created it)."
                    )
                if action == "reject" and not remarks:
                    return Response({"detail": "Rejection reason is required."}, status=400)
                locked.approval_status = (
                    RecoveryMemo.STATUS_APPROVED if action == "approve" else RecoveryMemo.STATUS_REJECTED
                )
                locked.approved_by = _actor_display(request)
                locked.approved_at = timezone.now()
                locked.approval_remarks = remarks
                locked.rejection_reason = "" if action == "approve" else remarks
                locked.save(
                    update_fields=[
                        "approval_status",
                        "approved_by",
                        "approved_at",
                        "approval_remarks",
                        "rejection_reason",
                        "updated_at",
                    ]
                )
                mark_recovery_notifications_resolved(locked)

        obj = _recoveries(request.user).get(pk=obj.pk)
        return Response(recovery_memo_to_dict(obj))


# --------------------------------------------------------------------------- seizure reports


def _recovery_for_memo(request, memo, recovery_id):
    if not recovery_id:
        return None, None
    recovery = _recoveries(request.user).filter(pk=recovery_id).first()
    if recovery is None:
        return None, Response({"detail": "Recovery memo not found."}, status=400)
    if recovery.detention_memo_id != memo.pk:
        return None, Response({"detail": "Recovery memo does not belong to this detention memo."}, status=400)
    return recovery, None


def _submit_prerequisites(assessment, recovery) -> Response | None:
    if not assessment or assessment.status != DetentionAssessment.STATUS_APPROVED:
        return Response({"detail": "Approved assessment is required to submit."}, status=400)
    if not recovery or recovery.approval_status != RecoveryMemo.STATUS_APPROVED:
        return Response({"detail": "Approved recovery memo is required to submit."}, status=400)
    return None


class SeizureReportListAPIView(_SeizureAPIView):
    def get(self, request):
        return Response([seizure_report_to_dict(o) for o in _reports(request.user)])


class SeizureReportCreateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def post(self, request):
        ser = SeizureReportWriteSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        memo, denied = _writable_memo(request, data["detentionMemoId"])
        if denied is not None:
            return denied
        if data.get("assessmentId"):
            assessment, error = _assessment_for_memo(request, memo, data["assessmentId"])
            if error is not None:
                return error
        else:
            assessment = (
                DetentionAssessment.objects.filter(detention_memo=memo, is_deleted=False)
                .order_by("-created_at")
                .first()
            )
        if data.get("recoveryMemoId"):
            recovery, error = _recovery_for_memo(request, memo, data["recoveryMemoId"])
            if error is not None:
                return error
        else:
            recovery = (
                RecoveryMemo.objects.filter(
                    detention_memo=memo,
                    approval_status=RecoveryMemo.STATUS_APPROVED,
                    is_deleted=False,
                )
                .order_by("-created_at")
                .first()
            )
        if data.get("status") == SeizureReport.STATUS_SUBMITTED:
            error = _submit_prerequisites(assessment, recovery)
            if error is not None:
                return error

        with transaction.atomic():
            obj = SeizureReport(detention_memo=memo, assessment=assessment, recovery_memo=recovery)
            notes = data.get("recoveryAssessmentNotes") or build_recovery_assessment_sheet(memo, assessment, recovery)
            data = {**data, "recoveryAssessmentNotes": notes}
            # assessment / recovery already resolved (and checked) above.
            data.pop("assessmentId", None)
            data.pop("recoveryMemoId", None)
            apply_seizure_report(obj, data)
        return Response(seizure_report_to_dict(obj), status=status.HTTP_201_CREATED)


class SeizureReportReadAPIView(_SeizureAPIView):
    def get(self, request, pk):
        obj = get_object_or_404(_reports(request.user), pk=pk)
        return Response(seizure_report_to_dict(obj))


class SeizureReportUpdateAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def put(self, request, pk):
        obj = get_object_or_404(_reports(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        payload = _recovery_payload(request)
        payload["detentionMemoId"] = str(obj.detention_memo_id)
        ser = SeizureReportWriteSerializer(data=payload)
        ser.is_valid(raise_exception=True)
        data = dict(ser.validated_data)
        memo = obj.detention_memo

        assessment = obj.assessment
        recovery = obj.recovery_memo
        if data.get("assessmentId"):
            assessment, error = _assessment_for_memo(request, memo, data["assessmentId"])
            if error is not None:
                return error
        if data.get("recoveryMemoId"):
            recovery, error = _recovery_for_memo(request, memo, data["recoveryMemoId"])
            if error is not None:
                return error

        with transaction.atomic():
            locked = SeizureReport.objects.select_for_update().get(pk=obj.pk)
            if locked.status == SeizureReport.STATUS_SUBMITTED and not can_full_edit(request.user):
                return Response(
                    {"detail": "A submitted seizure report can no longer be edited."},
                    status=400,
                )
            if data.get("status") == SeizureReport.STATUS_SUBMITTED:
                error = _submit_prerequisites(assessment, recovery)
                if error is not None:
                    return error
            apply_seizure_report(locked, data)
        obj = _reports(request.user).get(pk=obj.pk)
        return Response(seizure_report_to_dict(obj))


class SeizureReportDeleteAPIView(_SeizureAPIView):
    permission_classes = [DocumentWritePermission]

    def delete(self, request, pk):
        obj = get_object_or_404(_reports(request.user), pk=pk)
        denied = readonly_copy_response(obj)
        if denied is not None:
            return denied
        if obj.status == SeizureReport.STATUS_SUBMITTED:
            return Response({"detail": "Cannot delete a submitted seizure report."}, status=400)
        _soft_delete(obj)
        return Response(status=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- overview


DETENTION_WINDOW_DAYS = 60


def _parse_detention_datetime(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    text = text.replace(" ", "T", 1)
    candidates = [text, text[:19], text[:10]]
    for value in candidates:
        try:
            dt = datetime.fromisoformat(value)
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt, timezone.get_current_timezone())
            return dt
        except ValueError:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(value, fmt)
                return timezone.make_aware(dt, timezone.get_current_timezone())
            except ValueError:
                continue
    return None


def _iso(dt) -> str:
    return dt.isoformat() if dt else ""


def _detention_overdue_count(detentions) -> int:
    """Memos detained more than DETENTION_WINDOW_DAYS ago (``detentions`` is already scoped)."""
    cutoff = timezone.now() - timedelta(days=DETENTION_WINDOW_DAYS)
    overdue = 0
    for raw in detentions.exclude(date_time_detention="").values_list("date_time_detention", flat=True):
        dt = _parse_detention_datetime(raw)
        if dt is not None and dt < cutoff:
            overdue += 1
    return overdue


def _seizure_mgmt_overview(user) -> dict:
    today = timezone.localdate()
    note_sheets = scoped(NoteSheet.objects.filter(is_deleted=False), user)
    assessments = scoped(DetentionAssessment.objects.filter(is_deleted=False), user)
    recoveries = scoped(RecoveryMemo.objects.filter(is_deleted=False), user)
    reports = scoped(SeizureReport.objects.filter(is_deleted=False), user)
    detentions = scoped(DetentionMemo.objects.filter(is_deleted=False), user)

    recent: list[dict] = []
    for ns in note_sheets.order_by("-updated_at")[:8]:
        recent.append(
            {
                "kind": "note_sheet",
                "id": str(ns.id),
                "title": ns.note_sheet_no or ns.subject or "Note sheet",
                "subtitle": ns.case_no or ns.accused_name or "",
                "status": ns.status or "",
                "at": _iso(ns.updated_at),
            }
        )
    for dm in detentions.order_by("-updated_at")[:8]:
        recent.append(
            {
                "kind": "detention",
                "id": str(dm.id),
                "title": dm.case_no or "Detention memo",
                "subtitle": dm.place_of_detention or dm.owner_name or "",
                "status": dm.verification_status or dm.settlement_status or "",
                "at": _iso(dm.updated_at),
            }
        )
    for row in assessments.select_related("detention_memo").order_by("-updated_at")[:8]:
        memo = row.detention_memo
        recent.append(
            {
                "kind": "assessment",
                "id": str(row.id),
                "title": "Assessment",
                "subtitle": (memo.case_no if memo else "") or row.examining_officer or "",
                "status": row.status or "",
                "at": _iso(row.updated_at),
            }
        )
    for row in recoveries.select_related("detention_memo").order_by("-updated_at")[:8]:
        memo = row.detention_memo
        recent.append(
            {
                "kind": "recovery",
                "id": str(row.id),
                "title": "Recovery memo",
                "subtitle": (memo.case_no if memo else "") or row.category or "",
                "status": row.approval_status or "",
                "at": _iso(row.updated_at),
            }
        )
    for row in reports.select_related("detention_memo").order_by("-updated_at")[:8]:
        memo = row.detention_memo
        recent.append(
            {
                "kind": "seizure_report",
                "id": str(row.id),
                "title": "Seizure report",
                "subtitle": (memo.case_no if memo else "") or row.prepared_by or "",
                "status": row.status or "",
                "at": _iso(row.updated_at),
            }
        )
    recent.sort(key=lambda item: item.get("at") or "", reverse=True)

    return {
        "generatedAt": timezone.now().isoformat(),
        "detentionWindowDays": DETENTION_WINDOW_DAYS,
        "noteSheets": note_sheets.count(),
        "noteSheetsDraft": note_sheets.filter(status=NoteSheet.STATUS_DRAFT).count(),
        "noteSheetsPending": note_sheets.filter(status=NoteSheet.STATUS_SUBMITTED).count(),
        "noteSheetsApprovedAvailable": note_sheets.filter(
            status=NoteSheet.STATUS_APPROVED, detention_memo__isnull=True
        ).count(),
        "noteSheetsToday": note_sheets.filter(created_at__date=today).count(),
        "detentionMemos": detentions.count(),
        "detentionOverdue": _detention_overdue_count(detentions),
        "detentionsToday": detentions.filter(created_at__date=today).count(),
        "assessments": assessments.count(),
        "assessmentsPending": assessments.filter(status=DetentionAssessment.STATUS_SUBMITTED).count(),
        "assessmentsApproved": assessments.filter(status=DetentionAssessment.STATUS_APPROVED).count(),
        "assessmentsToday": assessments.filter(created_at__date=today).count(),
        "recoveryMemos": recoveries.count(),
        "recoveryPendingApproval": recoveries.filter(approval_status=RecoveryMemo.STATUS_PENDING).count(),
        "recoveryApproved": recoveries.filter(approval_status=RecoveryMemo.STATUS_APPROVED).count(),
        "recoveriesToday": recoveries.filter(created_at__date=today).count(),
        "seizureReports": reports.count(),
        "seizureReportsSubmitted": reports.filter(status=SeizureReport.STATUS_SUBMITTED).count(),
        "seizureReportsDraft": reports.filter(status=SeizureReport.STATUS_DRAFT).count(),
        "seizureReportsToday": reports.filter(created_at__date=today).count(),
        "recentActivity": recent[:12],
    }


class SeizureManagementOverviewAPIView(_SeizureAPIView):
    def get(self, request):
        return Response(_seizure_mgmt_overview(request.user))


# --------------------------------------------------------------------------- notifications


class NoteSheetNotificationListAPIView(_SeizureAPIView):
    """List in-app notifications for the logged-in official (note sheets + assessments + recovery)."""

    def get(self, request):
        unread_only = (request.query_params.get("unread") or "").strip() in ("1", "true", "yes")
        uid = request.user.id

        ns_qs = NoteSheetNotification.objects.filter(recipient_user_id=uid).select_related("note_sheet")
        as_qs = AssessmentNotification.objects.filter(recipient_user_id=uid).select_related(
            "assessment__detention_memo"
        )
        rc_qs = RecoveryNotification.objects.filter(recipient_user_id=uid).select_related(
            "recovery_memo__detention_memo"
        )
        if unread_only:
            ns_qs = ns_qs.filter(is_read=False)
            as_qs = as_qs.filter(is_read=False)
            rc_qs = rc_qs.filter(is_read=False)

        combined = (
            [_notification_to_dict(n) for n in ns_qs[:50]]
            + [_assessment_notification_to_dict(n) for n in as_qs[:50]]
            + [_recovery_notification_to_dict(n) for n in rc_qs[:50]]
        )
        combined.sort(key=lambda x: x.get("createdAt") or "", reverse=True)
        combined = combined[:50]

        unread_count = (
            NoteSheetNotification.objects.filter(recipient_user_id=uid, is_read=False).count()
            + AssessmentNotification.objects.filter(recipient_user_id=uid, is_read=False).count()
            + RecoveryNotification.objects.filter(recipient_user_id=uid, is_read=False).count()
        )
        return Response({"unreadCount": unread_count, "results": combined})


class NoteSheetNotificationMarkReadAPIView(_SeizureAPIView):
    def post(self, request, pk=None):
        mark_all = False
        if isinstance(getattr(request, "data", None), dict):
            mark_all = request.data.get("all") in (True, "1", "true", "yes")
        if not mark_all:
            mark_all = request.query_params.get("all") in ("1", "true", "yes")
        if mark_all:
            NoteSheetNotification.objects.filter(recipient_user_id=request.user.id, is_read=False).update(
                is_read=True
            )
            AssessmentNotification.objects.filter(recipient_user_id=request.user.id, is_read=False).update(
                is_read=True
            )
            RecoveryNotification.objects.filter(recipient_user_id=request.user.id, is_read=False).update(
                is_read=True
            )
            return Response({"detail": "All notifications marked as read."})
        if pk:
            ns = NoteSheetNotification.objects.filter(pk=pk, recipient_user_id=request.user.id).first()
            if ns:
                if not ns.is_read:
                    ns.is_read = True
                    ns.save(update_fields=["is_read"])
                return Response(_notification_to_dict(ns))
            an = AssessmentNotification.objects.filter(pk=pk, recipient_user_id=request.user.id).first()
            if an:
                if not an.is_read:
                    an.is_read = True
                    an.save(update_fields=["is_read"])
                return Response(_assessment_notification_to_dict(an))
            rn = get_object_or_404(RecoveryNotification, pk=pk, recipient_user_id=request.user.id)
            if not rn.is_read:
                rn.is_read = True
                rn.save(update_fields=["is_read"])
            return Response(_recovery_notification_to_dict(rn))
        return Response({"detail": "Provide notification id or {\"all\": true}."}, status=400)
