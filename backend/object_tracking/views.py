from __future__ import annotations

from datetime import date, datetime, timedelta
from datetime import time as dt_time

from django.db.models import Count, Prefetch, Q
from django.utils import timezone
from rest_framework import generics, permissions
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import GlobalObject, MotionClip, ObjectMovement, ObjectVisit, VisitStatus
from .serializers import (
    GlobalObjectDetailSerializer,
    GlobalObjectListSerializer,
    MotionClipSerializer,
    ObjectVisitSerializer,
)
from .services import object_type_for_class
from .services import finalize_stale_visits_globally


class ObjectTrackingPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100

    def get_paginated_response(self, data):
        page_size = self.get_page_size(self.request) or self.page_size
        return Response(
            {
                "count": self.page.paginator.count,
                "page": self.page.number,
                "page_size": page_size,
                "total_pages": self.page.paginator.num_pages or 1,
                "next": self.get_next_link(),
                "previous": self.get_previous_link(),
                "results": data,
            }
        )


def _parse_page_params(request, default_size: int = 25, max_size: int = 100) -> tuple[int, int]:
    try:
        page = int(request.query_params.get("page") or 1)
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(
            request.query_params.get("page_size")
            or request.query_params.get("limit")
            or default_size
        )
    except (TypeError, ValueError):
        page_size = default_size
    page = max(1, page)
    page_size = min(max(page_size, 1), max_size)
    return page, page_size


class ObjectTrackingSummaryAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        finalize_stale_visits_globally(limit=100)
        now = timezone.now()
        day_ago = now - timedelta(hours=24)
        objects = GlobalObject.objects.all()
        visits = ObjectVisit.objects.all()
        payload = {
            "objects_total": objects.count(),
            "present_now": objects.filter(exit_at__isnull=True).count(),
            "active_visits": visits.filter(status=VisitStatus.ACTIVE).count(),
            "visits_24h": visits.filter(entry_at__gte=day_ago).count(),
            "exits_24h": visits.filter(exit_at__gte=day_ago).count(),
            "by_type": {
                row["object_type"]: row["c"]
                for row in objects.values("object_type").annotate(c=Count("id"))
            },
        }
        return Response(payload)


class GlobalObjectListAPIView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = GlobalObjectListSerializer
    pagination_class = ObjectTrackingPagination

    def get_queryset(self):
        finalize_stale_visits_globally(limit=50)
        qs = GlobalObject.objects.select_related("latest_camera").annotate(
            visit_count=Count("visits")
        )
        object_type = self.request.query_params.get("object_type", "").strip()
        if object_type:
            qs = qs.filter(object_type=object_type)
        present = self.request.query_params.get("present")
        if present and str(present).lower() in ("1", "true", "yes"):
            qs = qs.filter(exit_at__isnull=True)
        elif present and str(present).lower() in ("0", "false", "no"):
            qs = qs.filter(exit_at__isnull=False)
        q = self.request.query_params.get("q", "").strip()
        if q:
            qs = qs.filter(
                Q(code__icontains=q)
                | Q(class_name__icontains=q)
                | Q(label__icontains=q)
            )
        return qs.order_by("-last_seen_at", "-created_at")


class GlobalObjectDetailAPIView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = GlobalObjectDetailSerializer
    lookup_field = "uuid"

    def get_queryset(self):
        return GlobalObject.objects.select_related("latest_camera").prefetch_related(
            Prefetch("visits", queryset=ObjectVisit.objects.select_related("camera").order_by("-entry_at")),
            "tracks",
        )


class ObjectVisitListAPIView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ObjectVisitSerializer
    pagination_class = ObjectTrackingPagination

    def get_queryset(self):
        finalize_stale_visits_globally(limit=50)
        qs = ObjectVisit.objects.select_related("global_object", "camera")
        status_param = self.request.query_params.get("status", "").strip()
        if status_param:
            qs = qs.filter(status=status_param)
        object_type = self.request.query_params.get("object_type", "").strip()
        if object_type:
            qs = qs.filter(global_object__object_type=object_type)
        code = self.request.query_params.get("code", "").strip()
        if code:
            qs = qs.filter(global_object__code__iexact=code)
        q = self.request.query_params.get("q", "").strip()
        if q:
            qs = qs.filter(
                Q(global_object__code__icontains=q)
                | Q(global_object__class_name__icontains=q)
                | Q(global_object__label__icontains=q)
                | Q(camera__name__icontains=q)
            )
        return qs.order_by("-entry_at")


class ObjectTrackingLiveAPIView(APIView):
    """Active visits currently present (professional live board)."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        finalize_stale_visits_globally(limit=100)
        page, page_size = _parse_page_params(request, default_size=15, max_size=100)
        qs = (
            ObjectVisit.objects.filter(status=VisitStatus.ACTIVE)
            .select_related("global_object", "camera")
            .order_by("-last_seen_at")
        )
        total = qs.count()
        total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
        if page > total_pages:
            page = total_pages
        offset = (page - 1) * page_size
        visits = qs[offset : offset + page_size]
        return Response(
            {
                "count": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "results": ObjectVisitSerializer(
                    visits, many=True, context={"request": request}
                ).data,
            }
        )


def _parse_time(value: str | None) -> dt_time | None:
    try:
        return dt_time.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _filter_date_time(qs, params, field: str):
    """date_from / date_to (YYYY-MM-DD) + time_from / time_to (HH:MM), local time.

    With a date, the time sets the exact start / end moment (19 Nov 10:00 → 19 Nov 12:30).
    Without a date, the time is a window on every day (09:00–17:00 any day; 22:00–06:00
    crosses midnight).
    """
    t_from, t_to = _parse_time(params.get("time_from")), _parse_time(params.get("time_to"))
    no_dates = not (params.get("date_from") or "").strip() and not (params.get("date_to") or "").strip()
    if no_dates and t_from and t_to and t_from > t_to:
        end = t_to.replace(second=59, microsecond=999999)
        return qs.filter(Q(**{f"{field}__time__gte": t_from}) | Q(**{f"{field}__time__lte": end}))
    for side in ("from", "to"):
        try:
            day = date.fromisoformat((params.get(f"date_{side}") or "").strip())
        except ValueError:
            day = None
        t = _parse_time(params.get(f"time_{side}"))
        if day is not None:
            if t is None:
                t = dt_time.min if side == "from" else dt_time.max
            elif side == "to":
                t = t.replace(second=59, microsecond=999999)
            moment = timezone.make_aware(datetime.combine(day, t), timezone.get_current_timezone())
            qs = qs.filter(**{f"{field}__gte" if side == "from" else f"{field}__lte": moment})
        elif t is not None:
            if side == "to":
                t = t.replace(second=59, microsecond=999999)
            qs = qs.filter(**{f"{field}__time__gte" if side == "from" else f"{field}__time__lte": t})
    return qs


class MotionClipListAPIView(generics.ListAPIView):
    """Motion clips (newest first). Filters: camera, object_type, class_name, object (uuid), q,
    date_from, date_to, time_from, time_to."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = MotionClipSerializer
    pagination_class = ObjectTrackingPagination

    def get_queryset(self):
        params = self.request.query_params
        qs = MotionClip.objects.select_related("camera").prefetch_related(
            Prefetch("movements", queryset=ObjectMovement.objects.select_related("global_object").order_by("started_at"))
        )
        camera = (params.get("camera") or "").strip()
        if camera.isdigit():
            qs = qs.filter(camera_id=int(camera))
        object_type = (params.get("object_type") or "").strip().lower()
        if object_type in ("person", "vehicle", "object"):
            # Unlinked movements only have a class name, so match on that too
            classes = set(
                ObjectMovement.objects.values_list("class_name", flat=True).distinct()
            )
            matching = [c for c in classes if object_type_for_class(c) == object_type]
            qs = qs.filter(
                Q(movements__global_object__object_type=object_type) | Q(movements__class_name__in=matching)
            )
        class_name = (params.get("class_name") or "").strip()
        if class_name:
            qs = qs.filter(movements__class_name__iexact=class_name)
        obj = (params.get("object") or "").strip()
        if obj:
            qs = qs.filter(movements__global_object__uuid=obj)
        q = (params.get("q") or "").strip()
        if q:
            qs = qs.filter(
                Q(movements__global_object__code__icontains=q)
                | Q(movements__ml_object_id__icontains=q)
                | Q(movements__class_name__icontains=q)
                | Q(camera__name__icontains=q)
            )
        qs = _filter_date_time(qs, params, "started_at")
        return qs.distinct().order_by("-started_at")


class MotionClipDetailAPIView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = MotionClipSerializer

    def get_queryset(self):
        return MotionClip.objects.select_related("camera").prefetch_related(
            Prefetch("movements", queryset=ObjectMovement.objects.select_related("global_object").order_by("started_at"))
        )
