import logging
import uuid

from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from cameras.models import Camera
from detentions.models import DetentionMemo
from users.models import User

from .distribution_service import (
    StockSyncError,
    complete_distribution,
    report_destruction_detection,
    start_distribution_recording,
    sync_stock_items,
)
from .models import DestructionAlert, FireSmokeDetectionLog, MemoDistribution, WarehouseStockItem
from .permissions import (
    DESTRUCTION_ALERT_ROLES,
    CanWriteWarehouse,
    cameras_in_scope,
    memo_in_scope,
    session_in_scope,
    session_scope_q,
    site_rows_q,
)
from .serializers import (
    DestructionAlertSerializer,
    FireSmokeDetectionLogSerializer,
    MemoDistributionSerializer,
    WarehouseStockItemSerializer,
)

logger = logging.getLogger(__name__)

WAREHOUSE_WRITE = [permissions.IsAuthenticated, CanWriteWarehouse]


def _int_list(raw) -> list[int] | None:
    """Parse camera ids; None when any id is not an integer."""
    values = raw if isinstance(raw, list) else [raw]
    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            out.append(int(str(value).strip()))
        except (TypeError, ValueError):
            return None
    return out


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _scoped_session(request, session_id):
    try:
        session = MemoDistribution.objects.select_related("camera").get(pk=session_id, is_deleted=False)
    except MemoDistribution.DoesNotExist:
        return None
    return session if session_in_scope(request.user, session) else None


class SyncWarehouseStockAPIView(APIView):
    permission_classes = WAREHOUSE_WRITE

    def post(self, request):
        rows = request.data.get("items") or request.data.get("rows") or []
        if not isinstance(rows, list):
            return Response({"detail": "items must be a list."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            created = sync_stock_items(rows)
        except StockSyncError as exc:
            return Response(
                {"detail": f"Stock sync rejected: {exc}", "errors": exc.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response({"synced": len(rows), "created": created})


class WarehouseStockListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        case_ref = (request.query_params.get("case_ref") or "").strip()
        memo_id = (request.query_params.get("detention_memo_id") or "").strip()
        qs = WarehouseStockItem.objects.filter(is_deleted=False).filter(site_rows_q(request.user))
        if memo_id:
            if not _is_uuid(memo_id):
                return Response({"detail": "detention_memo_id must be a UUID."}, status=status.HTTP_400_BAD_REQUEST)
            qs = qs.filter(detention_memo_id=memo_id)
        if case_ref:
            qs = qs.filter(case_ref__icontains=case_ref)
        return Response(WarehouseStockItemSerializer(qs[:200], many=True).data)


class MemoDistributionStartAPIView(APIView):
    permission_classes = WAREHOUSE_WRITE

    def post(self, request):
        memo_id = str(request.data.get("detention_memo_id") or "").strip()
        camera_id = request.data.get("camera_id")
        camera_ids = request.data.get("camera_ids") or []
        selected_items = request.data.get("selected_items") or []

        if not memo_id:
            return Response({"detail": "detention_memo_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not _is_uuid(memo_id):
            return Response({"detail": "detention_memo_id must be a UUID."}, status=status.HTTP_400_BAD_REQUEST)

        ids: list[int] | None = []
        if isinstance(camera_ids, list) and camera_ids:
            ids = _int_list(camera_ids)
        elif camera_id not in (None, ""):
            ids = _int_list(camera_id)
        if ids is None:
            return Response({"detail": "camera_ids must be numbers."}, status=status.HTTP_400_BAD_REQUEST)
        if not ids:
            return Response(
                {"detail": "camera_ids (or camera_id) is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        memo = DetentionMemo.objects.filter(pk=memo_id, is_deleted=False).first()
        if memo is None or not memo_in_scope(request.user, memo):
            return Response({"detail": "Detention memo not found."}, status=status.HTTP_404_NOT_FOUND)

        cameras = list(Camera.objects.filter(pk__in=ids, is_active=True).order_by("name"))
        if len(cameras) != len(set(ids)) or not cameras_in_scope(request.user, ids):
            return Response({"detail": "One or more cameras were not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            session = start_distribution_recording(
                detention_memo_id=memo_id,
                cameras=cameras,
                selected_items=selected_items if isinstance(selected_items, list) else [],
                performed_by=getattr(request.user, "username", "") or "",
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except FileNotFoundError:
            logger.exception("[distribution] ffmpeg not found while starting a recording")
            return Response(
                {"detail": "Recording service (ffmpeg) is not available on this server."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(
            MemoDistributionSerializer(session, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class MemoDistributionCompleteAPIView(APIView):
    permission_classes = WAREHOUSE_WRITE

    def post(self, request, session_id):
        if _scoped_session(request, session_id) is None:
            return Response({"detail": "Destruction record not found."}, status=status.HTTP_404_NOT_FOUND)
        try:
            session = complete_distribution(str(session_id))
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception("[distribution] completing session %s failed", session_id)
            return Response(
                {"detail": "Could not complete the destruction. Nothing was deducted; check the server log."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response(MemoDistributionSerializer(session, context={"request": request}).data)


class MemoDistributionDetectionAPIView(APIView):
    """Report smoke/fire ML detections during an active destruction session."""

    permission_classes = WAREHOUSE_WRITE

    def post(self, request, session_id):
        camera_id = request.data.get("camera_id")
        detections = request.data.get("detections") or []
        if camera_id in (None, ""):
            return Response({"detail": "camera_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        parsed = _int_list(camera_id)
        if not parsed:
            return Response({"detail": "camera_id must be a number."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(detections, list) or not all(isinstance(d, dict) for d in detections):
            return Response({"detail": "detections must be a list of objects."}, status=status.HTTP_400_BAD_REQUEST)
        if _scoped_session(request, session_id) is None:
            return Response({"detail": "Destruction record not found."}, status=status.HTTP_404_NOT_FOUND)
        try:
            log, alert = report_destruction_detection(
                str(session_id),
                camera_id=parsed[0],
                detections=detections,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if log is None:
            return Response({"log_created": False, "alert_created": False})
        payload = {
            "log_created": True,
            "log": FireSmokeDetectionLogSerializer(log, context={"request": request}).data,
            "alert_created": alert is not None,
        }
        if alert is not None:
            payload["alert"] = DestructionAlertSerializer(alert, context={"request": request}).data
        return Response(payload, status=status.HTTP_201_CREATED)


class MemoDistributionListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        memo_id = (request.query_params.get("detention_memo_id") or "").strip()
        case_ref = (request.query_params.get("case_ref") or "").strip()
        qs = (
            MemoDistribution.objects.select_related("camera")
            .filter(is_deleted=False)
            .filter(session_scope_q(request.user))
            .order_by("-created_at")
        )
        if memo_id:
            if not _is_uuid(memo_id):
                return Response({"detail": "detention_memo_id must be a UUID."}, status=status.HTTP_400_BAD_REQUEST)
            qs = qs.filter(detention_memo_id=memo_id)
        if case_ref:
            qs = qs.filter(detention_case_no__icontains=case_ref)
        return Response(
            MemoDistributionSerializer(qs[:100], many=True, context={"request": request}).data
        )


class MemoDistributionDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, session_id):
        session = _scoped_session(request, session_id)
        if session is None:
            return Response({"detail": "Destruction record not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(MemoDistributionSerializer(session, context={"request": request}).data)


class DestructionAlertListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user: User = request.user
        role = getattr(user, "role", "") or ""
        location = getattr(user, "location", "") or ""
        qs = DestructionAlert.objects.select_related("camera").filter(is_deleted=False).order_by("-created_at")

        if role == "ADMIN":
            pass
        elif role in DESTRUCTION_ALERT_ROLES:
            if location:
                qs = qs.filter(location_code=location)
            else:
                qs = qs.filter(notified_user_ids__contains=[user.pk])
        else:
            qs = qs.filter(notified_user_ids__contains=[user.pk])

        session_id = (request.query_params.get("distribution_id") or "").strip()
        if session_id:
            if not _is_uuid(session_id):
                return Response({"detail": "distribution_id must be a UUID."}, status=status.HTTP_400_BAD_REQUEST)
            qs = qs.filter(distribution_id=session_id)

        unack_only = (request.query_params.get("unacknowledged") or "").lower() in ("1", "true", "yes")
        if unack_only:
            qs = qs.filter(acknowledged=False)

        return Response(
            DestructionAlertSerializer(qs[:100], many=True, context={"request": request}).data
        )


class FireSmokeLogListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user: User = request.user
        role = getattr(user, "role", "") or ""
        location = getattr(user, "location", "") or ""
        qs = FireSmokeDetectionLog.objects.select_related("camera", "distribution").order_by("-created_at")

        if role == "ADMIN":
            pass
        elif role in DESTRUCTION_ALERT_ROLES:
            if location:
                qs = qs.filter(location_code=location)
        else:
            qs = qs.filter(performed_by=getattr(user, "username", "") or "")

        for param, field in (("distribution_id", "distribution_id"), ("detention_memo_id", "detention_memo_id")):
            value = (request.query_params.get(param) or "").strip()
            if value:
                if not _is_uuid(value):
                    return Response({"detail": f"{param} must be a UUID."}, status=status.HTTP_400_BAD_REQUEST)
                qs = qs.filter(**{field: value})

        camera_id = (request.query_params.get("camera_id") or "").strip()
        if camera_id:
            parsed = _int_list(camera_id)
            if not parsed:
                return Response({"detail": "camera_id must be a number."}, status=status.HTTP_400_BAD_REQUEST)
            qs = qs.filter(camera_id=parsed[0])

        location_code = (request.query_params.get("location_code") or "").strip()
        if location_code:
            qs = qs.filter(location_code=location_code)

        return Response(
            FireSmokeDetectionLogSerializer(qs[:200], many=True, context={"request": request}).data
        )


class DestructionAlertAcknowledgeAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, alert_id):
        user: User = request.user
        role = getattr(user, "role", "") or ""
        if role not in DESTRUCTION_ALERT_ROLES:
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        try:
            alert = DestructionAlert.objects.get(pk=alert_id, is_deleted=False)
        except DestructionAlert.DoesNotExist:
            return Response({"detail": "Alert not found."}, status=status.HTTP_404_NOT_FOUND)

        # Same visibility as the alert list: admins see all; others only their site's alerts
        # (or alerts they were notified about).
        location = (getattr(user, "location", "") or "").strip()
        if role != "ADMIN":
            same_site = bool(location) and (alert.location_code or "").strip().upper() == location.upper()
            notified = user.pk in (alert.notified_user_ids or [])
            if not (same_site or notified):
                return Response({"detail": "Alert not found."}, status=status.HTTP_404_NOT_FOUND)

        alert.acknowledged = True
        alert.acknowledged_at = timezone.now()
        alert.acknowledged_by = getattr(user, "username", "") or ""
        alert.save(update_fields=["acknowledged", "acknowledged_at", "acknowledged_by", "version"])
        return Response(DestructionAlertSerializer(alert, context={"request": request}).data)
