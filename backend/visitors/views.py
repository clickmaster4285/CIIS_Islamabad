from datetime import date, datetime, timedelta
import logging
import uuid

from django.db import IntegrityError, transaction
from django.db.models import F, Q
from django.utils import timezone
from rest_framework import generics, permissions, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.views import APIView
from rest_framework.response import Response

from .face_gallery import (
    active_face_count,
    deactivate_visitor_face,
    enroll_visitor_image,
    enroll_visitor_images,
    invalidate_gallery_cache,
    max_enrollment_images,
    min_enrollment_images,
    recognize_from_image,
    serialize_visitor_face,
    visitor_is_enrolled,
)
from .models import (
    Visitor,
    VisitorFace,
    ZoneAccessLog,
    SecurityAlert,
    Vehicle,
    VisitorNotification,
    VmsListRecord,
)
from .permissions import CanWriteVms, ReadAnyWriteVms, site_rows_q
from .serializers import (
    VisitorSerializer,
    VisitorWriteSerializer,
    FaceCaptureSerializer,
    ZoneScanSerializer,
    SecurityAlertSerializer,
    ApproveDenySerializer,
    NotifyHostSerializer,
    VehicleSerializer,
    VisitorNotificationSerializer,
    VmsListRecordSerializer,
    VmsListBulkSerializer,
)
from city_sync.identity import local_rows_q
from users.permissions import apply_location_filter, get_effective_location, resolve_location_for_write
from .payload_utils import merge_extra_into_response

logger = logging.getLogger(__name__)

VMS_OPERATOR = [permissions.IsAuthenticated, CanWriteVms]


def get_visitor_queryset():
    from .face_gallery import face_stats_qs

    return Visitor.objects.filter(is_deleted=False).annotate(face_count=face_stats_qs()).order_by("-created_at")


def filter_visitors_by_request(queryset, request):
    queryset = apply_location_filter(
        queryset,
        request.user,
        field="location",
        query_param=request.query_params.get("location"),
    )
    source = request.query_params.get("registration_source", "").strip()
    if source:
        queryset = queryset.filter(registration_source=source)
    status_param = request.query_params.get("registration_status", "").strip()
    if status_param:
        queryset = queryset.filter(registration_status=status_param)
    approval = request.query_params.get("approval_status", "").strip()
    if approval:
        queryset = queryset.filter(approval_status=approval)
    search = request.query_params.get("search", "").strip()
    if search:
        queryset = queryset.filter(
            Q(full_name__icontains=search)
            | Q(cnic_number__icontains=search)
            | Q(passport_number__icontains=search)
            | Q(qr_code_id__icontains=search)
        )
    return queryset


def scoped_visitors(request):
    """Visitors this user may act on: not deleted, inside their location scope (no list filters)."""
    return apply_location_filter(Visitor.objects.filter(is_deleted=False), request.user, field="location")


def get_scoped_visitor(request, pk):
    """Visitor by pk within the user's scope, or None (out-of-scope rows look like 404s)."""
    try:
        return scoped_visitors(request).get(pk=pk)
    except (Visitor.DoesNotExist, ValueError, TypeError):
        return None


def _effective_location(request):
    return get_effective_location(request.user, request.query_params.get("location"))


def _alert_scope_q(location):
    """Security alerts of one site: alerts on its visitors plus visitor-less alerts (bad QR) it raised."""
    if not location:
        return Q()
    return Q(visitor__location=location) | (Q(visitor__isnull=True) & site_rows_q(location))


def _limit_offset(request, default: int, maximum: int) -> tuple[int, int]:
    try:
        limit = int(request.query_params.get("limit", default))
    except (TypeError, ValueError):
        limit = default
    try:
        offset = int(request.query_params.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    return max(1, min(limit, maximum)), max(0, offset)


class CappedListMixin:
    """Plain-array list responses (the frontend expects arrays) with ?limit=&offset= paging.

    The full count is returned in the X-Total-Count header.
    """

    default_limit = 200
    max_limit = 1000

    def list(self, request, *args, **kwargs):
        qs = self.filter_queryset(self.get_queryset())
        limit, offset = _limit_offset(request, self.default_limit, self.max_limit)
        total = qs.count()
        data = self.get_serializer(qs[offset:offset + limit], many=True).data
        response = Response(data)
        response["X-Total-Count"] = str(total)
        return response


def serialize_visitor_list(queryset):
    from .serializers import VisitorListSerializer

    ser = VisitorListSerializer(queryset, many=True, context={"omit_blobs": True})
    return ser.data


def _actor_name(request) -> str:
    user = request.user
    return (getattr(user, "full_name", "") or getattr(user, "username", "") or "system").strip()[:150]


# ---------- Visitor CRUD ----------


class VisitorListAPIView(generics.ListAPIView):
    """GET /api/visitors/list/?registration_source=walk-in|pre-registration"""

    serializer_class = VisitorSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return filter_visitors_by_request(get_visitor_queryset(), self.request)

    def get_serializer_class(self):
        from .serializers import VisitorListSerializer

        return VisitorListSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context["omit_blobs"] = True
        return context


class VisitorCreateAPIView(APIView):
    """POST /api/visitors/create/ — body may include registration_source query or field."""

    permission_classes = VMS_OPERATOR

    def post(self, request):
        source = (
            request.data.get("registration_source")
            or request.query_params.get("registration_source")
            or ""
        )
        write_ser = VisitorWriteSerializer(
            data=request.data,
            context={"registration_source": str(source), "request": request},
        )
        if not write_ser.is_valid():
            return Response(write_ser.errors, status=status.HTTP_400_BAD_REQUEST)
        visitor = write_ser.save()
        return Response(VisitorSerializer(visitor).data, status=status.HTTP_201_CREATED)


class VisitorReadAPIView(generics.RetrieveAPIView):
    serializer_class = VisitorSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return filter_visitors_by_request(get_visitor_queryset(), self.request)


class VisitorProfileImageAPIView(APIView):
    """GET /api/visitors/<pk>/profile-image/ — visitor photo for list avatars (blobs omitted from list JSON)."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        from django.http import HttpResponse

        from .payload_utils import get_visitor_profile_photo_bytes

        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        try:
            visitor = qs.get(pk=pk)
        except Visitor.DoesNotExist:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)

        parsed = get_visitor_profile_photo_bytes(visitor)
        if not parsed:
            return Response({"detail": "No photo for this visitor."}, status=status.HTTP_404_NOT_FOUND)

        content_type, body = parsed
        response = HttpResponse(body, content_type=content_type)
        response["Cache-Control"] = "private, max-age=3600"
        return response


class VisitorUpdateAPIView(APIView):
    permission_classes = VMS_OPERATOR

    def put(self, request, pk):
        return self._update(request, pk, partial=False)

    def patch(self, request, pk):
        return self._update(request, pk, partial=True)

    def _update(self, request, pk, partial):
        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        try:
            visitor = qs.get(pk=pk)
        except Visitor.DoesNotExist:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        write_ser = VisitorWriteSerializer(
            visitor,
            data=request.data,
            partial=partial,
            context={"request": request},
        )
        if not write_ser.is_valid():
            return Response(write_ser.errors, status=status.HTTP_400_BAD_REQUEST)
        visitor = write_ser.save()
        return Response(VisitorSerializer(visitor).data)


class VisitorDeleteAPIView(APIView):
    """DELETE /api/visitors/<pk>/delete/ — soft delete within the user's location scope."""

    permission_classes = VMS_OPERATOR

    def delete(self, request, pk):
        visitor = get_scoped_visitor(request, pk)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        with transaction.atomic():
            visitor.is_deleted = True
            visitor.save(update_fields=["is_deleted", "updated_at", "version"])
            # Keep the enrolment rows (audit) but stop matching this face at gates.
            from city_sync.outbox import sync_update

            sync_update(VisitorFace.objects.filter(visitor=visitor, is_active=True), is_active=False)
        invalidate_gallery_cache()
        return Response(status=status.HTTP_204_NO_CONTENT)


class VisitorCnicCheckAPIView(APIView):
    """GET /api/visitors/check-cnic/?cnic=..."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        cnic = str(request.query_params.get("cnic", "")).strip().replace(" ", "").replace("-", "")
        digits = "".join(c for c in cnic if c.isdigit())
        # Too short to identify anyone (an empty/1-digit query would match every visitor).
        if len(digits) < 5:
            return Response({"exists": False})
        exists = scoped_visitors(request).filter(
            Q(cnic_number__icontains=digits) | Q(cnic_passport__icontains=digits)
        ).exists()
        return Response({"exists": exists})


# ---------- Active & Approval ----------


class ActiveVisitorsAPIView(APIView):
    """GET /api/visitors/active/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # Gate operations: this server's own visitors only (not HQ copies of city visitors).
        qs = Visitor.objects.filter(
            local_rows_q(),
            is_deleted=False,
            flow_stage="zone_checked_in",
            expiry_status="active",
        ).order_by("-updated_at")
        qs = filter_visitors_by_request(qs, request)
        return Response(serialize_visitor_list(qs))


class PendingApprovalsAPIView(APIView):
    """GET /api/approval/pending/"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        qs = Visitor.objects.filter(local_rows_q(), is_deleted=False, approval_status="pending").order_by("-created_at")
        qs = filter_visitors_by_request(qs, request)
        return Response(serialize_visitor_list(qs))


class VisitorApproveAPIView(APIView):
    """POST /api/visitors/<id>/approve/"""

    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        visitor = get_scoped_visitor(request, pk)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        ser = ApproveDenySerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        visitor.approval_status = "approved"
        # Recorded from the logged-in user; a body value cannot impersonate someone else.
        visitor.approved_by = _actor_name(request)
        visitor.registration_status = "approved"
        if visitor.flow_stage == "arrived":
            visitor.flow_stage = "registered"
        visitor.save()
        return Response(VisitorSerializer(visitor).data)


class VisitorDenyAPIView(APIView):
    """POST /api/visitors/<id>/deny/"""

    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        visitor = get_scoped_visitor(request, pk)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        ser = ApproveDenySerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        visitor.approval_status = "denied"
        visitor.denied_by = _actor_name(request)
        visitor.rejection_reason = ser.validated_data.get("rejection_reason") or ""
        visitor.expiry_status = "revoked"
        visitor.save()
        return Response(VisitorSerializer(visitor).data)


class VisitorNotifyHostAPIView(APIView):
    """POST /api/visitors/<id>/notify-host/

    There is no e-mail/SMS gateway wired into CIIS yet, so this only records a pending
    notification request; it does not claim the host was told.
    """

    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        visitor = get_scoped_visitor(request, pk)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        ser = NotifyHostSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        recipient = ser.validated_data.get("recipient") or visitor.host_email or visitor.host_contact_number
        if not recipient:
            return Response(
                {"detail": "No host email or phone on file."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        VisitorNotification.objects.create(
            visitor=visitor,
            notification_type="host_notify",
            recipient=recipient,
            message=(
                f"Visitor {visitor.full_name} is scheduled / on site. "
                f"[pending: not sent — no messaging gateway configured; requested by {_actor_name(request)}]"
            ),
            success=False,
        )
        return Response(
            {
                "host_notified_at": visitor.host_notified_at.isoformat() if visitor.host_notified_at else None,
                "recipient": recipient,
                "sent": False,
                "status": "pending",
                "detail": "Notification recorded but not sent: no e-mail/SMS gateway is configured.",
            },
            status=status.HTTP_200_OK,
        )


# ---------- Face Capture ----------


class VisitorFaceCaptureAPIView(APIView):
    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        visitor = get_scoped_visitor(request, pk)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        ser = FaceCaptureSerializer(data=request.data, partial=True)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        for key, value in ser.validated_data.items():
            setattr(visitor, key, value or "")
        visitor.flow_stage = "face_captured"
        visitor.save(update_fields=list(ser.validated_data.keys()) + ["flow_stage", "updated_at", "version"])
        return Response(VisitorSerializer(visitor).data)


class VisitorFaceEnrollAPIView(APIView):
    """POST /api/visitors/{id}/face/enroll/ — store 1–5 InsightFace embeddings. Not attendance."""

    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        try:
            visitor = qs.get(pk=pk)
        except Visitor.DoesNotExist:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)

        images = request.data.get("images")
        image = request.data.get("image")
        batch: list[str] = []
        if isinstance(images, list):
            batch = [str(item) for item in images if item]
        elif isinstance(image, str) and image.strip():
            batch = [image]

        if not batch:
            return Response(
                {"detail": "Provide image or images (base64 JPEG/PNG)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            if len(batch) == 1:
                result = enroll_visitor_image(visitor, batch[0])
            else:
                result = enroll_visitor_images(visitor, batch)
        except Exception:
            logger.exception("Visitor face enrollment failed for visitor %s", visitor.pk)
            return Response(
                {
                    "accepted": False,
                    "detail": "Face engine unavailable. Check InsightFace on the backend.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(result, status=status.HTTP_200_OK)


class VisitorFaceListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        try:
            visitor = qs.get(pk=pk)
        except Visitor.DoesNotExist:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        faces = VisitorFace.objects.filter(visitor=visitor, is_active=True)
        count = faces.count()
        return Response(
            {
                "visitor_id": visitor.pk,
                "faces": [serialize_visitor_face(face, request) for face in faces],
                "face_count": count,
                "images_required": min_enrollment_images(),
                "images_max": max_enrollment_images(),
                "is_enrolled": visitor_is_enrolled(visitor, count),
            }
        )


class VisitorFaceDeleteAPIView(APIView):
    permission_classes = VMS_OPERATOR

    def delete(self, request, pk, face_id):
        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        try:
            visitor = qs.get(pk=pk)
        except Visitor.DoesNotExist:
            return Response({"detail": "Visitor not found."}, status=status.HTTP_404_NOT_FOUND)
        if not deactivate_visitor_face(visitor, face_id):
            return Response({"detail": "Face not found."}, status=status.HTTP_404_NOT_FOUND)
        count = active_face_count(visitor)
        return Response(
            {
                "deleted": True,
                "face_count": count,
                "is_enrolled": visitor_is_enrolled(visitor, count),
            }
        )


class VisitorFaceRecognizeAPIView(APIView):
    """Identify staff then visitor. Never creates an attendance record. Gate / VMS operators only."""

    permission_classes = VMS_OPERATOR

    def post(self, request):
        image = request.data.get("image")
        if not isinstance(image, str) or not image.strip():
            return Response({"detail": "Provide image (base64)."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = recognize_from_image(image)
        except Exception:
            logger.exception("Visitor face recognize failed")
            return Response(
                {"matched": False, "identity_type": "unknown", "detail": "Face engine unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(result)


# ---------- Zone Scan ----------


def _parse_hhmm(raw):
    text = str(raw or "").strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _pass_window_problem(visitor) -> str:
    """'' when now is inside the pass's visit date / time window, else the reason it is not."""
    now = timezone.localtime()
    visit_day = visitor.visit_date or visitor.preferred_visit_date
    if visit_day and visit_day != now.date():
        return f"Pass is valid for {visit_day.isoformat()}, not today."
    start = _parse_hhmm(visitor.time_validity_start)
    end = _parse_hhmm(visitor.time_validity_end)
    current = now.time().replace(second=0, microsecond=0)
    if start and current < start:
        return f"Pass is not valid until {start.strftime('%H:%M')}."
    if end and current > end:
        return f"Pass expired at {end.strftime('%H:%M')}."
    return ""


class ZoneScanAPIView(APIView):
    permission_classes = VMS_OPERATOR

    def post(self, request):
        ser = ZoneScanSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        qr_code_id = ser.validated_data["qr_code_id"].strip()
        zone = ser.validated_data["zone"].strip()
        gate = ser.validated_data.get("gate") or ""
        scan_type = ser.validated_data.get("scan_type") or "entry"
        scanner_id = ser.validated_data.get("scanner_id") or ""

        def _alert(visitor, alert_type, severity, message):
            SecurityAlert.objects.create(
                visitor=visitor,
                alert_type=alert_type,
                severity=severity,
                message=message,
                zone=zone,
                gate=gate,
            )

        candidates = scoped_visitors(request).filter(local_rows_q())
        try:
            visitor = candidates.get(qr_code_id=qr_code_id)
        except Visitor.DoesNotExist:
            _alert(None, "invalid_qr", "high", f"Scan with unknown QR code: {qr_code_id} at zone={zone} gate={gate}")
            return Response(
                {
                    "allowed": False,
                    "message": "Invalid or unknown QR code.",
                    "alert_created": True,
                },
                status=status.HTTP_200_OK,
            )
        except Visitor.MultipleObjectsReturned:
            # qr_code_id is not unique in the database (legacy rows / synced copies); never guess who it is.
            _alert(None, "invalid_qr", "high", f"QR code {qr_code_id} is assigned to more than one visitor (zone={zone} gate={gate}).")
            return Response(
                {
                    "allowed": False,
                    "message": "This QR code is assigned to more than one visitor. Re-issue the pass.",
                    "alert_created": True,
                },
                status=status.HTTP_200_OK,
            )

        allowed = True
        msg = "Access granted."

        from .screening_utils import apply_visitor_screening, visitor_blocks_entry

        apply_visitor_screening(visitor, create_alert=False)
        blocked, block_msg = visitor_blocks_entry(visitor)
        if blocked:
            allowed = False
            msg = block_msg
            _alert(visitor, "watchlist_hit", "critical", msg)

        if scan_type == "entry":
            if allowed and visitor.approval_status != "approved":
                allowed = False
                msg = (
                    "Visitor approval was denied."
                    if visitor.approval_status == "denied"
                    else "Visitor is not approved yet."
                )
                _alert(visitor, "other", "high", msg)

            if allowed and str(visitor.registration_status or "").strip().lower() == "draft":
                allowed = False
                msg = "Registration is still a draft."
                _alert(visitor, "other", "medium", msg)

            if allowed and visitor.expiry_status in ("expired", "revoked"):
                allowed = False
                msg = f"QR code is {visitor.expiry_status}."
                _alert(visitor, "expired_qr", "high", msg)

            if allowed:
                problem = _pass_window_problem(visitor)
                if problem:
                    allowed = False
                    msg = problem
                    _alert(visitor, "expired_qr", "high", msg)

            if allowed and visitor.access_zone and visitor.access_zone != "all":
                if zone != visitor.access_zone:
                    allowed = False
                    msg = f"Visitor not allowed in zone: {zone}."
                    _alert(visitor, "mismatch_zone", "medium", msg)

            if allowed:
                recent_entry = ZoneAccessLog.objects.filter(
                    visitor=visitor,
                    scan_type="entry",
                    allowed=True,
                    scanned_at__gte=timezone.now() - timedelta(minutes=5),
                ).exists()
                if recent_entry and visitor.flow_stage == "zone_checked_in":
                    allowed = False
                    msg = "Duplicate entry scan detected."
                    _alert(visitor, "duplicate_entry", "medium", msg)

        ZoneAccessLog.objects.create(
            visitor=visitor,
            zone=zone,
            gate=gate,
            scan_type=scan_type,
            allowed=allowed,
            message=msg,
            scanner_id=scanner_id,
        )

        if allowed:
            # Atomic increment: concurrent scans at two gates must not lose a count.
            from city_sync.outbox import sync_update

            sync_update(Visitor.objects.filter(pk=visitor.pk), scan_count=F("scan_count") + 1)
            visitor.refresh_from_db(fields=["scan_count"])
            if scan_type == "exit":
                visitor.flow_stage = "exited"
            elif visitor.flow_stage in ("arrived", "registered", "face_captured", "qr_printed"):
                visitor.flow_stage = "zone_checked_in"
            visitor.save(update_fields=["flow_stage", "updated_at", "version"])

        return Response(
            {
                "allowed": allowed,
                "message": msg,
                "visitor_id": visitor.id,
                "visitor_name": visitor.full_name,
                "flow_stage": visitor.flow_stage,
            },
            status=status.HTTP_200_OK,
        )


# ---------- Security ----------


class SecurityAlertListAPIView(CappedListMixin, generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SecurityAlertSerializer
    default_limit = 100
    max_limit = 500

    def get_queryset(self):
        qs = SecurityAlert.objects.filter(is_deleted=False).select_related("visitor")
        qs = qs.filter(_alert_scope_q(_effective_location(self.request)))
        acknowledged = self.request.query_params.get("acknowledged")
        if acknowledged is not None:
            if acknowledged.lower() in ("false", "0", "no"):
                qs = qs.filter(acknowledged=False)
            elif acknowledged.lower() in ("true", "1", "yes"):
                qs = qs.filter(acknowledged=True)
        severity = self.request.query_params.get("severity")
        if severity:
            qs = qs.filter(severity=severity)
        return qs


class SecurityAlertAcknowledgeAPIView(APIView):
    """POST /api/security/alerts/<id>/acknowledge/"""

    permission_classes = VMS_OPERATOR

    def post(self, request, pk):
        scope = get_effective_location(request.user, None)
        qs = SecurityAlert.objects.filter(is_deleted=False).filter(_alert_scope_q(scope))
        try:
            alert = qs.get(pk=pk)
        except SecurityAlert.DoesNotExist:
            return Response({"detail": "Alert not found."}, status=status.HTTP_404_NOT_FOUND)
        alert.acknowledged = True
        alert.acknowledged_at = timezone.now()
        alert.acknowledged_by = _actor_name(request)[:100]
        alert.save()
        return Response(SecurityAlertSerializer(alert).data)


class SecurityDashboardAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        today = timezone.localdate()
        location = _effective_location(request)
        visitors = apply_location_filter(
            Visitor.objects.filter(is_deleted=False), request.user, field="location",
            query_param=request.query_params.get("location"),
        )
        alerts = SecurityAlert.objects.filter(is_deleted=False).filter(_alert_scope_q(location))
        visitors_today = visitors.filter(created_at__date=today).count()
        in_building = visitors.filter(flow_stage="zone_checked_in").count()
        alerts_unack = alerts.filter(acknowledged=False).count()
        alerts_today = alerts.filter(created_at__date=today).count()
        recent_alerts = alerts.filter(acknowledged=False).select_related("visitor")[:10]
        return Response(
            {
                "visitors_today": visitors_today,
                "in_building": in_building,
                "alerts_unacknowledged": alerts_unack,
                "alerts_today": alerts_today,
                "recent_alerts": SecurityAlertSerializer(recent_alerts, many=True).data,
            },
            status=status.HTTP_200_OK,
        )


# ---------- Vehicles ----------


class VehicleListAPIView(CappedListMixin, generics.ListAPIView):
    serializer_class = VehicleSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = Vehicle.objects.select_related("visitor").filter(is_deleted=False)
        qs = apply_location_filter(
            qs, self.request.user, field="visitor__location",
            query_param=self.request.query_params.get("location"),
        )
        visitor_id = self.request.query_params.get("visitor_id")
        if visitor_id:
            qs = qs.filter(visitor_id=visitor_id)
        return qs


class VehicleCreateAPIView(generics.CreateAPIView):
    serializer_class = VehicleSerializer
    permission_classes = VMS_OPERATOR
    queryset = Vehicle.objects.all()

    def perform_create(self, serializer):
        visitor = serializer.validated_data.get("visitor")
        if visitor is None or not scoped_visitors(self.request).filter(pk=visitor.pk).exists():
            raise PermissionDenied("That visitor is not in your location.")
        serializer.save()


# ---------- Notifications ----------


class NotificationListAPIView(CappedListMixin, generics.ListAPIView):
    serializer_class = VisitorNotificationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = VisitorNotification.objects.select_related("visitor").all()
        qs = apply_location_filter(
            qs, self.request.user, field="visitor__location",
            query_param=self.request.query_params.get("location"),
        )
        visitor_id = self.request.query_params.get("visitor_id")
        if visitor_id:
            qs = qs.filter(visitor_id=visitor_id)
        return qs


# ---------- VMS Analytics ----------


class VmsAnalyticsAPIView(APIView):
    """GET /api/vms/analytics/?from=YYYY-MM-DD&to=YYYY-MM-DD"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        today = timezone.localdate()
        from_str = request.query_params.get("from") or str(today)
        to_str = request.query_params.get("to") or str(today)
        try:
            from_date = date.fromisoformat(from_str[:10])
            to_date = date.fromisoformat(to_str[:10])
        except ValueError:
            from_date = today
            to_date = today

        location = _effective_location(request)
        visitors = Visitor.objects.filter(is_deleted=False)
        if location:
            visitors = visitors.filter(location=location)
        logs = ZoneAccessLog.objects.filter(is_deleted=False)
        if location:
            logs = logs.filter(visitor__location=location)
        alerts = SecurityAlert.objects.filter(is_deleted=False).filter(_alert_scope_q(location))

        visitors_registered = visitors.filter(
            created_at__date__gte=from_date,
            created_at__date__lte=to_date,
        ).count()
        in_building_now = visitors.filter(flow_stage="zone_checked_in").count()
        zone_scans = logs.filter(
            scanned_at__date__gte=from_date,
            scanned_at__date__lte=to_date,
        ).count()
        alerts_in_range = alerts.filter(
            created_at__date__gte=from_date,
            created_at__date__lte=to_date,
        ).count()
        alerts_unacknowledged = alerts.filter(acknowledged=False).count()
        pending_approvals = visitors.filter(approval_status="pending").count()
        approved_in_range = visitors.filter(
            approval_status="approved",
            updated_at__date__gte=from_date,
            updated_at__date__lte=to_date,
        ).count()
        denied_in_range = visitors.filter(
            approval_status="denied",
            updated_at__date__gte=from_date,
            updated_at__date__lte=to_date,
        ).count()

        return Response(
            {
                "from_date": str(from_date),
                "to_date": str(to_date),
                "visitors_registered": visitors_registered,
                "in_building_now": in_building_now,
                "zone_scans_in_range": zone_scans,
                "alerts_in_range": alerts_in_range,
                "alerts_unacknowledged": alerts_unacknowledged,
                "pending_approvals": pending_approvals,
                "approved_in_range": approved_in_range,
                "denied_in_range": denied_in_range,
            }
        )


def _visitor_clearance_label(visitor):
    status = (visitor.watchlist_check_status or "").lower()
    if "blacklist" in status or "flagged" in status:
        return "Blacklisted"
    if visitor.approval_status == "denied":
        return "Rejected"
    if visitor.approval_status == "pending":
        return "Pending Approval"
    if visitor.registration_status == "draft":
        return "Pending Docs"
    return "Cleared"


def _visitor_priority_label(visitor):
    view = (visitor.preferred_view_visit or "").lower()
    if view == "high-security":
        return "High"
    if view == "logins":
        return "Urgent"
    return "Normal"


def _visitor_datetime_label(visitor):
    date_part = ""
    if visitor.preferred_visit_date:
        date_part = visitor.preferred_visit_date.strftime("%d-%m-%Y")
    elif visitor.visit_date:
        date_part = visitor.visit_date.strftime("%d-%m-%Y")
    elif visitor.created_at:
        date_part = timezone.localtime(visitor.created_at).strftime("%d-%m-%Y")
    slot = visitor.preferred_time_slot or visitor.preferred_time_slot_walkin or ""
    if slot:
        return f"{date_part} | {slot}" if date_part else slot
    return date_part or "—"


def _serialize_overview_visitor(visitor):
    vehicles = list(visitor.vehicles.all()[:1])
    plate = vehicles[0].plate_number if vehicles else "-"
    clearance = _visitor_clearance_label(visitor)
    approval = visitor.approval_status or "pending"
    return {
        "id": visitor.id,
        "reg_id": str(visitor.id),
        "date": timezone.localtime(visitor.created_at).strftime("%d-%m-%Y") if visitor.created_at else "—",
        "visitor_name": visitor.full_name,
        "organization": visitor.organization_name or "—",
        "vehicle_id": plate,
        "status": clearance,
        "host_name": visitor.host_full_name or visitor.host_officer_name or "—",
        "date_time": _visitor_datetime_label(visitor),
        "priority": _visitor_priority_label(visitor),
        "approval_status": approval,
        "registration_source": visitor.registration_source or "",
    }


class VmsOverviewAPIView(APIView):
    """GET /api/vms/overview/ — dashboard stats and recent visitor rows."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        today = timezone.localdate()
        qs = filter_visitors_by_request(get_visitor_queryset(), request)

        expected_today = qs.filter(
            Q(preferred_visit_date=today) | Q(visit_date=today)
        ).count()
        checked_in = qs.filter(
            flow_stage="zone_checked_in", expiry_status="active"
        ).count()
        pending_docs = qs.filter(registration_status="draft").count()
        pending_approval = qs.filter(approval_status="pending").count()
        rejected_requests = qs.filter(approval_status="denied").count()
        active_passes = qs.filter(
            approval_status="approved",
            expiry_status="active",
        ).exclude(flow_stage="exited").count()

        location = _effective_location(request)
        bl_qs = VmsListRecord.objects.filter(module="vms_blacklist_management_rows", is_deleted=False)
        if location:
            bl_qs = bl_qs.filter(location=location)
        blacklisted_visitors = bl_qs.count()
        if blacklisted_visitors == 0:
            blacklisted_visitors = qs.filter(
                watchlist_check_status__icontains="blacklist"
            ).count()

        blacklisted_vehicles = 0
        vehicle_rows = VmsListRecord.objects.filter(module="vms_vehicle_entries", is_deleted=False)
        if location:
            vehicle_rows = vehicle_rows.filter(location=location)
        for data in vehicle_rows.values_list("data", flat=True).iterator():
            data = data or {}
            status_text = str(data.get("status") or data.get("vehicle_status") or "").lower()
            if "blacklist" in status_text:
                blacklisted_vehicles += 1

        recent_qs = qs.prefetch_related("vehicles")[:10]
        registered_qs = (
            qs.filter(approval_status="approved")
            .prefetch_related("vehicles")
            .order_by("-created_at")[:10]
        )

        return Response(
            {
                "expected_today": expected_today,
                "checked_in": checked_in,
                "pending_docs": pending_docs,
                "pending_approval": pending_approval,
                "blacklisted_visitors": blacklisted_visitors,
                "blacklisted_vehicles": blacklisted_vehicles,
                "rejected_requests": rejected_requests,
                "active_passes": active_passes,
                "visitors_registered_today": qs.filter(created_at__date=today).count(),
                "recent_registrations": [
                    _serialize_overview_visitor(v) for v in recent_qs
                ],
                "registered_visitors": [
                    _serialize_overview_visitor(v) for v in registered_qs
                ],
            }
        )


# ---------- VMS List storage (blacklist, watchlist, etc.) ----------


SCREENING_MODULES = ("vms_watchlist_screening_rows", "vms_blacklist_management_rows")


class VmsListRecordsAPIView(APIView):
    """
    GET /api/vms/lists/?module=vms_blacklist_management_rows
    PUT /api/vms/lists/  body: { module, rows: [{ id, ...fields }] }
    """

    permission_classes = [permissions.IsAuthenticated, ReadAnyWriteVms]

    def get(self, request):
        module = request.query_params.get("module", "").strip()
        if not module:
            return Response({"detail": "module query param required."}, status=400)
        # Whole-list editor: only this server's rows, so a save never rewrites city copies.
        records = VmsListRecord.objects.filter(local_rows_q(), module=module, is_deleted=False)
        location = get_effective_location(
            request.user, request.query_params.get("location")
        )
        if location:
            records = records.filter(location=location)
        rows = [{"id": r.record_id, **(r.data or {})} for r in records]
        return Response(rows)

    def put(self, request):
        ser = VmsListBulkSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        module = ser.validated_data["module"]
        rows = ser.validated_data["rows"]
        location = resolve_location_for_write(
            request.user, ser.validated_data.get("location") or ""
        )
        own = VmsListRecord.objects.filter(local_rows_q(), module=module)
        if location:
            replace_qs = own.filter(location=location)
        else:
            # Unscoped editor saving without a location: replace only unassigned rows, rows of the
            # locations present in this payload, and the rows being rewritten — never every site's list.
            row_locations = {str(r.get("location") or "").strip() for r in rows} - {""}
            row_ids = {str(r.get("id") or "").strip() for r in rows} - {""}
            replace_qs = own.filter(
                Q(location="") | Q(location__in=row_locations) | Q(record_id__in=row_ids)
            )

        try:
            with transaction.atomic():
                replace_qs.delete()
                taken = set(
                    VmsListRecord.objects.filter(module=module).values_list("record_id", flat=True)
                )
                bulk = []
                for row in rows:
                    record_id = str(row.get("id") or "").strip()[:80]
                    if not record_id or record_id in taken:
                        record_id = f"row-{uuid.uuid4().hex}"
                    taken.add(record_id)
                    data = {k: v for k, v in row.items() if k != "id"}
                    bulk.append(
                        VmsListRecord(
                            module=module,
                            record_id=record_id,
                            data=data,
                            location=location or str(row.get("location") or "")[:30],
                        )
                    )
                if bulk:
                    from city_sync.outbox import sync_bulk_create

                    sync_bulk_create(VmsListRecord, bulk)
        except IntegrityError:
            logger.exception("VMS list save failed for module %s", module)
            return Response(
                {"detail": "Could not save the list (duplicate record id). Reload and try again."},
                status=status.HTTP_409_CONFLICT,
            )
        if module in SCREENING_MODULES:
            from .screening_utils import rescreen_all_visitors

            rescreen_all_visitors(location=location or "")
        return Response({"module": module, "count": len(bulk)})


class VmsScreeningSummaryAPIView(APIView):
    """GET /api/vms/screening/summary/ — visitor screening status counts."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        qs = filter_visitors_by_request(get_visitor_queryset(), request)
        return Response(
            {
                "total": qs.count(),
                "cleared": qs.filter(watchlist_check_status="cleared").count(),
                "flagged": qs.filter(watchlist_check_status="flagged").count(),
                "potential": qs.filter(watchlist_check_status="potential").count(),
                "blacklisted": qs.filter(
                    watchlist_check_status__icontains="blacklist"
                ).count(),
                "not_checked": qs.filter(
                    Q(watchlist_check_status="") | Q(watchlist_check_status__isnull=True)
                ).count(),
            }
        )


class VmsScreeningRescreenAPIView(APIView):
    """POST /api/vms/screening/rescreen/ — re-run screening for one or all visitors."""

    permission_classes = VMS_OPERATOR

    def post(self, request):
        from .screening_utils import apply_visitor_screening, rescreen_all_visitors

        visitor_id = request.data.get("visitor_id")
        if visitor_id:
            visitor = get_scoped_visitor(request, visitor_id)
            if visitor is None:
                return Response({"detail": "Visitor not found."}, status=404)
            apply_visitor_screening(visitor)
            return Response({"screened": 1})

        location = get_effective_location(
            request.user, request.data.get("location")
        )
        count = rescreen_all_visitors(location=location or "")
        return Response({"screened": count})


class VmsScreeningMarkAPIView(APIView):
    """POST /api/vms/screening/mark/ — flag or blacklist a registered visitor."""

    permission_classes = VMS_OPERATOR

    def post(self, request):
        from .screening_utils import mark_visitor_blacklisted, mark_visitor_flagged

        visitor_id = request.data.get("visitor_id")
        action = str(request.data.get("action") or "").strip().lower()
        remarks = str(request.data.get("remarks") or "").strip()
        marked_by = _actor_name(request) or "security"

        if not visitor_id:
            return Response({"detail": "visitor_id is required."}, status=400)
        if action not in ("flagged", "blacklisted"):
            return Response(
                {"detail": "action must be 'flagged' or 'blacklisted'."},
                status=400,
            )

        visitor = get_scoped_visitor(request, visitor_id)
        if visitor is None:
            return Response({"detail": "Visitor not found."}, status=404)

        if action == "flagged":
            result = mark_visitor_flagged(visitor, remarks=remarks, marked_by=marked_by)
        else:
            result = mark_visitor_blacklisted(visitor, reason=remarks, marked_by=marked_by)

        return Response(
            {
                "visitor_id": visitor.pk,
                "watchlist_check_status": visitor.watchlist_check_status,
                "screening": result.as_extra(),
            }
        )
