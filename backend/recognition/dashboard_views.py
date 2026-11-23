from datetime import datetime

from django.db.models import Count, Q
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from recognition.models import FaceEnrollment
from recognition.scoping import staff_scope_q
from users.models import Attendance, Staff
from users.permissions import IsAdminOrHR, get_location_scope
from users.serializers import AttendanceSerializer


def _scoped_staff(user):
    """Active (not deleted) staff the viewer may see. Not limited to local rows: HQ dashboards include cities."""
    return Staff.objects.filter(staff_scope_q(user, local_only=False))


def _scoped_attendance(user):
    qs = Attendance.objects.filter(is_deleted=False)
    scope = get_location_scope(user)
    if scope:
        qs = qs.filter(Q(user__location=scope) | Q(staff__user__location=scope))
    return qs


class DashboardSummaryView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        today = timezone.localdate()
        staff_qs = _scoped_staff(request.user)
        total_staff = staff_qs.count()
        trained_faces = FaceEnrollment.objects.filter(
            staff_scope_q(request.user, "staff__", local_only=False), is_trained=True
        ).count()
        attendance = _scoped_attendance(request.user)
        today_records = attendance.filter(date=today).select_related("staff", "user")

        present = today_records.filter(
            Q(status=Attendance.STATUS_PRESENT) | Q(status=Attendance.STATUS_LATE) | Q(check_in__isnull=False)
        ).distinct().count()
        late = today_records.filter(status=Attendance.STATUS_LATE).count()
        checked_out = today_records.filter(check_out__isnull=False).count()
        # Count unique staff present; fall back to records with check_in
        present_staff_ids = set(
            today_records.exclude(staff_id=None).values_list("staff_id", flat=True)
        )
        absent = max(total_staff - len(present_staff_ids), 0)

        department_stats = (
            today_records.filter(staff__isnull=False)
            .values("staff__department")
            .annotate(
                total=Count("id"),
                present_count=Count("id", filter=Q(status=Attendance.STATUS_PRESENT)),
                late_count=Count("id", filter=Q(status=Attendance.STATUS_LATE)),
            )
            .order_by("staff__department")
        )

        recent = attendance.select_related("staff", "user").order_by("-check_in", "-id")[:10]

        return Response({
            "date": str(today),
            "summary": {
                "total_employees": total_staff,
                "trained_faces": trained_faces,
                "present_today": present,
                "late_today": late,
                "absent_today": absent,
                "checked_out_today": checked_out,
            },
            "department_stats": [
                {
                    "department": row["staff__department"] or "Unassigned",
                    "total": row["total"],
                    "present_count": row["present_count"],
                    "late_count": row["late_count"],
                }
                for row in department_stats
            ],
            "recent_activity": AttendanceSerializer(recent, many=True).data,
        })


class DailyReportView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        date_str = (request.query_params.get("date") or "").strip()
        if date_str:
            try:
                report_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return Response({"error": "date must be YYYY-MM-DD"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            report_date = timezone.localdate()

        records = _scoped_attendance(request.user).filter(date=report_date).select_related("staff", "user")
        present_staff_ids = set(records.exclude(staff_id=None).values_list("staff_id", flat=True))
        # Records punched by user account only: map those users to their staff rows in one query.
        user_ids = set(records.filter(staff_id=None).exclude(user_id=None).values_list("user_id", flat=True))
        if user_ids:
            present_staff_ids.update(
                Staff.objects.filter(user_id__in=user_ids).values_list("pk", flat=True)
            )

        absent_list = [
            {
                "staff_id": emp["pk"],
                "employee_id": emp["employee_id"],
                "name": emp["full_name"],
                "department": emp["department"],
            }
            for emp in _scoped_staff(request.user)
            .exclude(pk__in=present_staff_ids)
            .order_by("full_name")
            .values("pk", "employee_id", "full_name", "department")
        ]

        return Response({
            "date": str(report_date),
            "attendance": AttendanceSerializer(records, many=True).data,
            "absent": absent_list,
            "totals": {
                "present": records.filter(status=Attendance.STATUS_PRESENT).count()
                or records.filter(check_in__isnull=False, status=Attendance.STATUS_PRESENT).count()
                or records.filter(check_in__isnull=False).exclude(status=Attendance.STATUS_LATE).count(),
                "late": records.filter(status=Attendance.STATUS_LATE).count(),
                "absent": len(absent_list),
            },
        })
