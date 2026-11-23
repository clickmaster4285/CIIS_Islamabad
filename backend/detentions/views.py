from __future__ import annotations

import json
import logging
from typing import Any

from django.conf import settings
from django.db import transaction
from django.db.models import Prefetch
from django.shortcuts import get_object_or_404
from rest_framework import permissions, status
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from .access import (
    DepositDeletePermission,
    DepositWritePermission,
    DocumentDeletePermission,
    DocumentWritePermission,
    MemoEditPermission,
    readonly_copy_response,
    scoped,
)
from .image_utils import compress_image
from .models import (
    DepositAccountEntry,
    DetentionMemo,
    DetentionMemoAttachment,
    DetentionMemoGoodsImage,
    DetentionMemoGoodsLine,
)
from .serializers import (
    DepositAccountWriteSerializer,
    DetentionMemoWriteSerializer,
    apply_partial_to_deposit_entry,
    apply_validated_to_memo,
    create_deposit_account_entry,
    deposit_entry_to_frontend_dict,
    memo_to_frontend_dict,
    replace_goods_lines,
)
from .uploads import KIND_DOCUMENT, KIND_IMAGE, KIND_VIDEO, validate_upload

logger = logging.getLogger(__name__)

GOODS_IMAGE_LIMIT = 10


def _detention_memo_scan_url(*, memo_id, client_origin: str = "") -> str:
    """Canonical frontend detail URL encoded in memo QR (not print mode)."""
    origin = (client_origin or "").strip().rstrip("/")
    if not origin:
        origin = (getattr(settings, "FRONTEND_ORIGIN", "") or "").strip().rstrip("/")
    if not origin:
        origin = ""
    return f"{origin}/seizure-management/detention-memo/{memo_id}"


def _actor_name(request) -> str:
    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return ""
    return (getattr(user, "full_name", "") or "").strip() or user.get_username() or ""


def _ensure_memo_qr(memo: DetentionMemo, client_origin: str = "") -> None:
    """QR number / payload are server-managed: fill when blank, keep the scan URL canonical."""
    update_fields = []
    base = (client_origin or "").strip() or (getattr(settings, "FRONTEND_ORIGIN", "") or "").strip()
    existing = (memo.memo_qr_code_payload or "").strip()
    if base or not existing:
        correct_payload = _detention_memo_scan_url(memo_id=memo.pk, client_origin=base)
        if existing != correct_payload:
            memo.memo_qr_code_payload = correct_payload
            update_fields.append("memo_qr_code_payload")
    if not (memo.memo_qr_code_number or "").strip():
        memo.memo_qr_code_number = f"DM-MEMO-{str(memo.pk).replace('-', '')[:12].upper()}"
        update_fields.append("memo_qr_code_number")
    if update_fields:
        memo.save(update_fields=update_fields)


def _body_from_request(request) -> dict[str, Any]:
    ct = (request.content_type or "").lower()
    if "multipart/form-data" in ct:
        raw = request.data.get("payload")
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}
    if isinstance(request.data, dict):
        return dict(request.data)
    try:
        return dict(request.data)
    except Exception:
        return {}


def _goods_image_files(request, goods_items: list | None) -> dict[str, list]:
    """Uploaded goods images keyed by client line id (goods_image_{id}_{n})."""
    out: dict[str, list] = {}
    for item in goods_items or []:
        if not isinstance(item, dict):
            continue
        client_line_id = str(item.get("id") or "").strip()
        if not client_line_id:
            continue
        prefix = f"goods_image_{client_line_id}_"
        files = [request.FILES[key] for key in request.FILES if key.startswith(prefix)]
        if files:
            out[client_line_id] = files
    return out


def _validate_memo_uploads(request, goods_items: list | None) -> None:
    """Check every upload before anything is written (whitelist + size cap)."""
    for key in ("owner_photo", "driver_photo"):
        f = request.FILES.get(key)
        if f:
            validate_upload(f, KIND_IMAGE, field=key)
    for f in request.FILES.getlist("documents"):
        validate_upload(f, KIND_DOCUMENT, field="documents")
    for f in request.FILES.getlist("videos"):
        validate_upload(f, KIND_VIDEO, field="videos")
    for files in _goods_image_files(request, goods_items).values():
        for f in files:
            validate_upload(f, KIND_IMAGE, field="goods_images")


def _save_memo_uploaded_media(request, memo: DetentionMemo) -> None:
    """Persist owner/driver FileFields and uploaded document/video attachments."""
    update_fields = []

    if request.FILES.get("owner_photo"):
        memo.owner_photo_upload = compress_image(
            request.FILES["owner_photo"], max_width=1920, max_height=1080, quality=85
        )
        update_fields.append("owner_photo_upload")

    if request.FILES.get("driver_photo"):
        memo.driver_photo_upload = compress_image(
            request.FILES["driver_photo"], max_width=1920, max_height=1080, quality=85
        )
        update_fields.append("driver_photo_upload")

    if update_fields:
        memo.save(update_fields=update_fields)

    for f in request.FILES.getlist("documents"):
        DetentionMemoAttachment.objects.create(
            memo=memo,
            kind=DetentionMemoAttachment.KIND_DOCUMENT,
            file=f,
            original_filename=f.name or "",
        )

    for f in request.FILES.getlist("videos"):
        DetentionMemoAttachment.objects.create(
            memo=memo,
            kind=DetentionMemoAttachment.KIND_VIDEO,
            file=f,
            original_filename=f.name or "",
        )


def _save_goods_images(request, memo: DetentionMemo, goods_items: list | None) -> None:
    """Save goods images uploaded for each goods line item (max 10 per line)."""
    for client_line_id, files in _goods_image_files(request, goods_items).items():
        goods_line = DetentionMemoGoodsLine.objects.filter(
            memo=memo, client_line_id=client_line_id, is_deleted=False
        ).first()
        if not goods_line:
            continue
        remaining_slots = GOODS_IMAGE_LIMIT - goods_line.images.filter(is_deleted=False).count()
        for img_file in files[: max(0, remaining_slots)]:
            DetentionMemoGoodsImage.objects.create(
                goods_line=goods_line,
                image=compress_image(img_file, max_width=1920, max_height=1080, quality=85),
            )


def get_memo_queryset():
    return DetentionMemo.objects.prefetch_related(
        Prefetch(
            "goods_lines",
            queryset=DetentionMemoGoodsLine.objects.filter(is_deleted=False).select_related(
                "located_camera__nvr__site",
                "located_camera__ml_server",
                "detection_event",
            ),
        ),
        Prefetch("goods_lines__images", queryset=DetentionMemoGoodsImage.objects.filter(is_deleted=False)),
        Prefetch("attachments", queryset=DetentionMemoAttachment.objects.filter(is_deleted=False)),
    ).filter(is_deleted=False)


def scoped_memo_queryset(user):
    """Memos this user may read (own location; global admin sees all)."""
    return scoped(get_memo_queryset(), user)


def _scoped_memo_lookup_qs(user):
    return scoped(DetentionMemo.objects.filter(is_deleted=False), user)


class DetentionMemoListAPIView(APIView):
    """GET /api/detention-memos/list/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        memos = scoped_memo_queryset(request.user).order_by("-created_at")
        data = [memo_to_frontend_dict(m, request) for m in memos]
        return Response(data)


class DetentionMemoCreateAPIView(APIView):
    """POST /api/detention-memos/create/ (JSON or multipart/form-data with `payload` + files)."""

    permission_classes = [DocumentWritePermission]

    def post(self, request):
        body = _body_from_request(request)
        ser = DetentionMemoWriteSerializer(data=body)
        ser.is_valid(raise_exception=True)
        validated = ser.validated_data
        _validate_memo_uploads(request, validated.get("goodsItems"))

        actor = _actor_name(request)
        with transaction.atomic():
            memo = DetentionMemo()
            apply_validated_to_memo(memo, validated)
            memo.created_by = actor
            memo.updated_by = actor
            memo.save()
            replace_goods_lines(memo, validated.get("goodsItems"))
            _save_memo_uploaded_media(request, memo)
            _save_goods_images(request, memo, validated.get("goodsItems"))
            _ensure_memo_qr(memo, validated.get("clientOrigin") or "")

        memo = get_memo_queryset().get(pk=memo.pk)
        return Response(memo_to_frontend_dict(memo, request), status=status.HTTP_201_CREATED)


class DetentionMemoReadAPIView(APIView):
    """GET /api/detention-memos/<uuid:pk>/read/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        memo = get_object_or_404(scoped_memo_queryset(request.user), pk=pk)
        return Response(memo_to_frontend_dict(memo, request))


class DetentionMemoUpdateAPIView(APIView):
    """PUT /api/detention-memos/<uuid:pk>/update/ — full replacement + optional extra files appended."""

    permission_classes = [MemoEditPermission]

    def put(self, request, pk):
        memo = get_object_or_404(_scoped_memo_lookup_qs(request.user), pk=pk)
        denied = readonly_copy_response(memo)
        if denied is not None:
            return denied

        body = _body_from_request(request)
        ser = DetentionMemoWriteSerializer(data=body)
        ser.is_valid(raise_exception=True)
        validated = ser.validated_data
        _validate_memo_uploads(request, validated.get("goodsItems"))

        with transaction.atomic():
            memo = DetentionMemo.objects.select_for_update().get(pk=memo.pk)
            apply_validated_to_memo(memo, validated)
            if not (memo.created_by or "").strip():
                memo.created_by = _actor_name(request)
            memo.updated_by = _actor_name(request)
            memo.save()
            replace_goods_lines(memo, validated.get("goodsItems"))
            _save_memo_uploaded_media(request, memo)
            _save_goods_images(request, memo, validated.get("goodsItems"))
            _ensure_memo_qr(memo, validated.get("clientOrigin") or "")

        memo = get_memo_queryset().get(pk=memo.pk)
        return Response(memo_to_frontend_dict(memo, request))


def _soft_delete(obj) -> None:
    obj.is_deleted = True
    obj.save()


class DetentionMemoDeleteAPIView(APIView):
    """DELETE /api/detention-memos/<uuid:pk>/delete/ — soft delete (syncs to HQ as a deletion)."""

    permission_classes = [DocumentDeletePermission]

    def delete(self, request, pk):
        from seizure_management.models import DetentionAssessment, NoteSheet, RecoveryMemo, SeizureReport

        memo = get_object_or_404(_scoped_memo_lookup_qs(request.user), pk=pk)
        denied = readonly_copy_response(memo)
        if denied is not None:
            return denied
        with transaction.atomic():
            memo = DetentionMemo.objects.select_for_update().get(pk=memo.pk)
            _soft_delete(memo)
            # Mirror the old cascade: the memo's assessment / recovery / report go with it,
            # and the note sheet is unlinked so it can be used again.
            for model in (DetentionAssessment, RecoveryMemo, SeizureReport):
                for row in model.objects.filter(detention_memo=memo, is_deleted=False):
                    _soft_delete(row)
            for sheet in NoteSheet.objects.filter(detention_memo=memo):
                sheet.detention_memo = None
                sheet.save(update_fields=["detention_memo", "updated_at"])
        return Response(status=status.HTTP_204_NO_CONTENT)


def _deposit_queryset(user):
    return scoped(
        DepositAccountEntry.objects.select_related("detention_memo").filter(is_deleted=False),
        user,
    )


class DepositAccountEntryListAPIView(APIView):
    """GET /api/deposit-accounts/list/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        qs = _deposit_queryset(request.user).order_by("-created_at")
        return Response([deposit_entry_to_frontend_dict(e) for e in qs])


class DepositAccountEntryCreateAPIView(APIView):
    """POST /api/deposit-accounts/create/"""

    permission_classes = [DepositWritePermission]

    def post(self, request):
        ser = DepositAccountWriteSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        memo_qs = _scoped_memo_lookup_qs(request.user)
        raw_memo = (ser.validated_data.get("detentionMemoId") or "").strip()
        if raw_memo:
            memo = memo_qs.filter(pk=raw_memo).first() if _is_uuid(raw_memo) else None
            if memo is None:
                raise DRFValidationError({"detentionMemoId": "Detention memo not found."})
            denied = readonly_copy_response(memo)
            if denied is not None:
                return denied
        entry = create_deposit_account_entry(ser.validated_data, memo_qs=memo_qs)
        entry.refresh_from_db()
        return Response(deposit_entry_to_frontend_dict(entry), status=status.HTTP_201_CREATED)


def _is_uuid(value: str) -> bool:
    import uuid

    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


class DepositAccountEntryReadAPIView(APIView):
    """GET /api/deposit-accounts/<uuid:pk>/read/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        entry = get_object_or_404(_deposit_queryset(request.user), pk=pk)
        return Response(deposit_entry_to_frontend_dict(entry))


class DepositAccountEntryUpdateAPIView(APIView):
    """PATCH /api/deposit-accounts/<uuid:pk>/update/ — partial update (camelCase)."""

    permission_classes = [DepositWritePermission]

    def patch(self, request, pk):
        entry = get_object_or_404(_deposit_queryset(request.user), pk=pk)
        denied = readonly_copy_response(entry)
        if denied is not None:
            return denied
        if not isinstance(request.data, dict):
            return Response({"detail": "Expected JSON object."}, status=status.HTTP_400_BAD_REQUEST)
        apply_partial_to_deposit_entry(
            entry, dict(request.data), memo_qs=_scoped_memo_lookup_qs(request.user)
        )
        entry.save()
        entry.refresh_from_db()
        return Response(deposit_entry_to_frontend_dict(entry))


class DepositAccountEntryDeleteAPIView(APIView):
    """DELETE /api/deposit-accounts/<uuid:pk>/delete/ — soft delete."""

    permission_classes = [DepositDeletePermission]

    def delete(self, request, pk):
        entry = get_object_or_404(_deposit_queryset(request.user), pk=pk)
        denied = readonly_copy_response(entry)
        if denied is not None:
            return denied
        _soft_delete(entry)
        return Response(status=status.HTTP_204_NO_CONTENT)
