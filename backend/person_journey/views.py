from __future__ import annotations

import hmac
from datetime import timedelta

from django.conf import settings
from django.db.models import Count, Prefetch, Q
from django.http import Http404
from django.utils import timezone
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from city_sync.identity import local_rows_q
from users.permissions import apply_source_location_filter

from .access import CanViewPersonJourney, can_edit_journeys, person_scope_q
from .models import CameraTrack, JourneyEvent, JourneyPerson, PersonStatus, PersonType
from .serializers import (
    IngestObservationSerializer,
    JourneyEventSerializer,
    JourneyPersonDetailSerializer,
    JourneyPersonListSerializer,
    JourneyPersonNameUpdateSerializer,
    MergeVisitorSerializer,
)
from .services import ingest_track_observation, merge_person_to_visitor
from .snapshot_utils import annotate_person_list_extras, person_camera_captures, serializer_context_for_events

_VISIBLE_TYPES = [PersonType.STAFF, PersonType.VISITOR, PersonType.UNKNOWN]


def _clip_context_for_events(events, person=None) -> dict:
    return serializer_context_for_events(list(events), person=person)


def _detail_limit() -> int:
    return max(20, int(getattr(settings, "JOURNEY_DETAIL_MAX_ROWS", 200)))


def visible_persons(user):
    """Journey persons this user may open (location scope applied)."""
    return JourneyPerson.objects.filter(person_scope_q(user))


def visible_events(user):
    return JourneyEvent.objects.filter(person_scope_q(user, "journey_person__"))


def _surviving_person(person: JourneyPerson) -> JourneyPerson:
    """A merged duplicate's events live on the person it was merged into — show that journey."""
    seen = {person.pk}
    for _ in range(10):
        if person.status != PersonStatus.MERGED or not person.merged_into_id:
            break
        nxt = JourneyPerson.objects.filter(pk=person.merged_into_id).first()
        if nxt is None or nxt.pk in seen:
            # Survivor was tombstoned (city sync) or the chain loops: nothing to show.
            raise Http404("Person not found.")
        seen.add(nxt.pk)
        person = nxt
    return person


def _get_visible_person(request, uuid) -> JourneyPerson:
    person = visible_persons(request.user).filter(uuid=uuid).first()
    if person is None:
        raise Http404("Person not found.")
    return _surviving_person(person)


class JourneyIngestAPIView(APIView):
    """POST /api/person-journey/ingest/ — ML pipeline observation intake."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        expected = getattr(settings, "PERSON_JOURNEY_INGEST_TOKEN", "").strip()
        token = (
            request.headers.get("X-Journey-Ingest-Token")
            or request.META.get("HTTP_X_JOURNEY_INGEST_TOKEN")
            or ""
        ).strip()
        if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
            return Response({"detail": "Invalid ingest token."}, status=status.HTTP_403_FORBIDDEN)

        ser = IngestObservationSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        result = ingest_track_observation(ser.validated_data)
        if result.get("ignored"):
            # Stale ML pipelines after camera delete — do not 500.
            return Response(result, status=status.HTTP_200_OK)
        return Response(result, status=status.HTTP_201_CREATED)


class JourneyPersonListAPIView(generics.ListAPIView):
    """List all journey persons (paginated). Default excludes merged duplicates."""

    permission_classes = [CanViewPersonJourney]
    serializer_class = JourneyPersonListSerializer

    def get_queryset(self):
        qs = (
            visible_persons(self.request.user)
            .select_related("staff", "visitor", "latest_camera")
            .filter(person_type__in=_VISIBLE_TYPES)
        )
        qs = apply_source_location_filter(qs, self.request.user, self.request.query_params.get("source_location"))
        person_type = self.request.query_params.get("person_type")
        if person_type:
            qs = qs.filter(person_type=person_type.strip())
        status_param = self.request.query_params.get("status")
        if status_param:
            qs = qs.filter(status=status_param.strip())
        else:
            # Hide merged duplicates unless explicitly requested.
            qs = qs.exclude(status=PersonStatus.MERGED)
        active_only = self.request.query_params.get("active_only")
        if active_only and str(active_only).lower() in ("1", "true", "yes"):
            qs = qs.filter(status=PersonStatus.ACTIVE)
        q = self.request.query_params.get("q", "").strip()
        if q:
            qs = qs.filter(
                Q(code__icontains=q)
                | Q(display_name__icontains=q)
                | Q(latest_zone__icontains=q)
                | Q(latest_camera__name__icontains=q)
            )
        return qs.order_by("-latest_seen_at", "-created_at")

    def list(self, request, *args, **kwargs):
        qs = self.filter_queryset(self.get_queryset())
        try:
            page = max(1, int(request.query_params.get("page") or 1))
        except (TypeError, ValueError):
            page = 1
        try:
            page_size = int(request.query_params.get("page_size") or 20)
        except (TypeError, ValueError):
            page_size = 20
        page_size = max(5, min(100, page_size))

        total = qs.count()
        start = (page - 1) * page_size
        end = start + page_size
        items = annotate_person_list_extras(qs[start:end])
        serializer = self.get_serializer(items, many=True)
        return Response(
            {
                "count": total,
                "page": page,
                "page_size": page_size,
                "total_pages": max(1, (total + page_size - 1) // page_size) if total else 1,
                "results": serializer.data,
            }
        )


class JourneyPersonDetailAPIView(generics.RetrieveAPIView):
    permission_classes = [CanViewPersonJourney]
    serializer_class = JourneyPersonDetailSerializer
    lookup_field = "uuid"

    def get_queryset(self):
        limit = _detail_limit()
        # Newest rows only: a long-lived person can have thousands of events / tracklets.
        return visible_persons(self.request.user).select_related(
            "staff", "visitor", "latest_camera"
        ).prefetch_related(
            Prefetch(
                "events",
                queryset=JourneyEvent.objects.select_related("camera", "track", "journey_person").order_by(
                    "-created_at"
                )[:limit],
            ),
            Prefetch(
                "tracks",
                queryset=CameraTrack.objects.select_related("camera", "journey_person").order_by("-started_at")[
                    :limit
                ],
            ),
        )

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        survivor = _surviving_person(instance)
        if survivor.pk != instance.pk:
            instance = self.get_queryset().filter(pk=survivor.pk).first()
            if instance is None:
                raise Http404("Person not found.")
        annotate_person_list_extras([instance])
        events = list(instance.events.all())
        context = self.get_serializer_context()
        context.update(_clip_context_for_events(events, person=instance))
        serializer = self.get_serializer(instance, context=context)
        return Response(serializer.data)

    def patch(self, request, *args, **kwargs):
        """Let administrators give this journey person a clear name."""
        if not can_edit_journeys(request.user):
            return Response({"detail": "Only administrators may rename journey persons."}, status=status.HTTP_403_FORBIDDEN)
        instance = _surviving_person(self.get_object())
        if not JourneyPerson.objects.filter(local_rows_q(), pk=instance.pk).exists():
            return Response(
                {"detail": "This person was recorded on another site; rename it there."},
                status=status.HTTP_403_FORBIDDEN,
            )
        if instance.person_type == PersonType.STAFF or instance.staff_id:
            return Response(
                {"detail": "Staff names come from the HR record and cannot be changed here."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        ser = JourneyPersonNameUpdateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        name = ser.validated_data["display_name"].strip()
        if not name:
            return Response({"display_name": ["Name cannot be empty."]}, status=status.HTTP_400_BAD_REQUEST)
        from .services import merge_person_metadata

        instance.display_name = name[:200]
        instance.save(update_fields=["display_name", "updated_at"])
        merge_person_metadata(instance.pk, {"name_locked": True, "named_at": timezone.now().isoformat()})
        instance.refresh_from_db()
        return Response(JourneyPersonListSerializer(instance, context=self.get_serializer_context()).data)


class JourneyPersonTimelineAPIView(APIView):
    permission_classes = [CanViewPersonJourney]

    def get(self, request, uuid):
        person = _get_visible_person(request, uuid)

        from .snapshot_utils import backfill_snapshot_paths_for_person

        backfill_snapshot_paths_for_person(person)

        refresh = str(request.query_params.get("refresh", "")).lower() in ("1", "true", "yes")
        if refresh:
            from .snapshot_capture import capture_missing_for_person

            # Queued in the background; the next poll shows the new images.
            capture_missing_for_person(person, since=None, limit=15)

        events = JourneyEvent.objects.filter(journey_person=person).select_related("camera", "track", "journey_person")
        date_from = request.query_params.get("date_from")
        date_to = request.query_params.get("date_to")
        try:
            if date_from:
                events = events.filter(created_at__date__gte=date_from)
            if date_to:
                events = events.filter(created_at__date__lte=date_to)
            events = events.order_by("created_at")
            events_list = list(events)
        except Exception:
            return Response({"detail": "date_from / date_to must be YYYY-MM-DD."}, status=status.HTTP_400_BAD_REQUEST)
        ctx = _clip_context_for_events(events_list, person=person)
        return Response(
            {
                "person": JourneyPersonListSerializer(annotate_person_list_extras([person])[0]).data,
                "events": JourneyEventSerializer(events_list, many=True, context=ctx).data,
            }
        )


class JourneyLiveAPIView(APIView):
    """Active persons seen in the last N minutes."""

    permission_classes = [CanViewPersonJourney]

    def get(self, request):
        minutes = 30
        try:
            minutes = int(request.query_params.get("minutes", 30))
        except (TypeError, ValueError):
            pass
        minutes = max(1, min(minutes, 24 * 60))
        since = timezone.now() - timedelta(minutes=minutes)
        qs = (
            visible_persons(request.user)
            .filter(latest_seen_at__gte=since, person_type__in=_VISIBLE_TYPES)
            .select_related("latest_camera", "staff", "visitor")
            .order_by("-latest_seen_at")[:100]
        )
        results = annotate_person_list_extras(qs)
        try:
            from .snapshot_capture import enqueue_latest_crops_for_persons

            enqueue_latest_crops_for_persons(results)
        except Exception:
            pass
        return Response(
            {
                "count": len(results),
                "results": JourneyPersonListSerializer(results, many=True).data,
            }
        )


class JourneySummaryAPIView(APIView):
    permission_classes = [CanViewPersonJourney]

    def get(self, request):
        today = timezone.localdate()
        since = timezone.now() - timedelta(hours=24)
        persons = visible_persons(request.user)
        events = visible_events(request.user)
        return Response(
            {
                "active_now": persons.filter(status=PersonStatus.ACTIVE, person_type__in=_VISIBLE_TYPES).count(),
                "unknown_today": persons.filter(person_type=PersonType.UNKNOWN, created_at__date=today).count(),
                "visitors_today": persons.filter(person_type=PersonType.VISITOR, latest_seen_at__date=today).count(),
                "staff_recognized_24h": events.filter(event_type="staff_recognized", created_at__gte=since).count(),
                "events_24h": events.filter(created_at__gte=since).count(),
                "by_type": {
                    row["person_type"]: row["c"]
                    for row in persons.filter(latest_seen_at__gte=since).values("person_type").annotate(c=Count("id"))
                },
            }
        )


class JourneyMergeVisitorAPIView(APIView):
    permission_classes = [CanViewPersonJourney]

    def post(self, request):
        if not can_edit_journeys(request.user):
            return Response({"detail": "Only administrators may link persons to visitors."}, status=status.HTTP_403_FORBIDDEN)
        ser = MergeVisitorSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        uuid = str(ser.validated_data["person_uuid"])
        if not visible_persons(request.user).filter(local_rows_q(), uuid=uuid).exists():
            return Response({"detail": "Person not found on this site."}, status=status.HTTP_404_NOT_FOUND)
        try:
            person = merge_person_to_visitor(
                uuid,
                ser.validated_data["visitor_id"],
                face_match_score=ser.validated_data.get("face_match_score"),
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if person is None:
            return Response({"detail": "Merge failed."}, status=status.HTTP_400_BAD_REQUEST)
        return Response(JourneyPersonListSerializer(annotate_person_list_extras([person])[0]).data)


class JourneyPersonCameraCapturesAPIView(APIView):
    """Latest capture image per camera for one journey person."""

    permission_classes = [CanViewPersonJourney]

    def get(self, request, uuid):
        person = _get_visible_person(request, uuid)

        hours = 0  # 0 = all history
        try:
            raw_hours = request.query_params.get("hours")
            if raw_hours is not None and str(raw_hours).strip() != "":
                hours = int(raw_hours)
        except (TypeError, ValueError):
            hours = 0
        since = None
        if hours > 0:
            since = timezone.now() - timedelta(hours=hours)

        from .snapshot_utils import backfill_snapshot_paths_for_person

        backfill_snapshot_paths_for_person(person)

        refresh = str(request.query_params.get("refresh", "")).lower() in ("1", "true", "yes")
        if refresh:
            from .snapshot_capture import capture_missing_for_person

            capture_missing_for_person(person, since=since, limit=12)

        captures = person_camera_captures(person, since=since)

        return Response(
            {
                "person": JourneyPersonListSerializer(annotate_person_list_extras([person])[0]).data,
                "count": len(captures),
                "results": captures,
            }
        )


class JourneyCameraSightingsAPIView(APIView):
    """Latest captured sighting per camera (with snapshot)."""

    permission_classes = [CanViewPersonJourney]

    def get(self, request):
        # Missing hours -> last 24 h. hours=0 ("all history") is capped at JOURNEY_SIGHTINGS_MAX_HOURS.
        max_hours = max(1, int(getattr(settings, "JOURNEY_SIGHTINGS_MAX_HOURS", 24 * 7)))
        hours = 24
        try:
            raw_hours = request.query_params.get("hours")
            if raw_hours is not None and str(raw_hours).strip() != "":
                hours = int(raw_hours)
        except (TypeError, ValueError):
            hours = 24
        if hours <= 0 or hours > max_hours:
            hours = max_hours
        since = timezone.now() - timedelta(hours=hours)
        # One query: Postgres DISTINCT ON (camera_id) keeps the newest event per camera.
        latest_per_camera = list(
            visible_events(request.user)
            .filter(camera__isnull=False, created_at__gte=since)
            .select_related("camera", "journey_person", "track")
            .order_by("camera_id", "-created_at")
            .distinct("camera_id")
        )

        ctx = _clip_context_for_events(latest_per_camera)
        return Response(
            {
                "count": len(latest_per_camera),
                "results": JourneyEventSerializer(latest_per_camera, many=True, context=ctx).data,
            }
        )


class JourneyRecentEventsAPIView(APIView):
    """Recent journey events with snapshots (all cameras)."""

    permission_classes = [CanViewPersonJourney]

    def get(self, request):
        limit = 40
        try:
            limit = int(request.query_params.get("limit", 40))
        except (TypeError, ValueError):
            pass
        limit = min(max(limit, 1), 100)
        events = list(
            visible_events(request.user)
            .filter(
                camera__isnull=False,
                detection_event_id__isnull=False,
            )
            .select_related("camera", "journey_person", "track")
            .order_by("-created_at")[:limit]
        )
        ctx = _clip_context_for_events(events)
        return Response(
            {
                "count": len(events),
                "results": JourneyEventSerializer(events, many=True, context=ctx).data,
            }
        )
