"""Create (or activate) a starter monitoring policy so the app has something to show.

Edit the text to match the organization's approved policy before using in production.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from employee_monitoring.models import MonitoringPolicy
from employee_monitoring.services import deployment_id, retention_days


def default_categories(ret):
    """The canonical category list, shared by the seeder and the enable/disable command."""
    return [
            {
                "key": "work_session_location",
                "label": "Work-session location",
                "description": "Your GPS location while an authorized work session is active, for check-ins and duty verification.",
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "work_session_status",
                "label": "Work-session status",
                "description": "When you start and end a work session, and app foreground/locked time on your work phone.",
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "attendance",
                "label": "Attendance",
                "description": "Your check-in and check-out times.",
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "call_metadata",
                "label": "Company phone-system calls",
                "description": "Direction, masked number, time and duration of calls handled by the approved company phone system.",
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "android_call_history",
                "label": "Android call history",
                "description": (
                    "On an Android work phone, after you acknowledge this policy and allow call history, "
                    "CIIS reads incoming, outgoing and missed call numbers, times and durations and uploads "
                    "them when you open the app. Calls from before you acknowledge are not uploaded. "
                    "iPhone cellular call history is not available. The app does not record call audio."
                ),
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "call_recording",
                "label": "Work call recordings",
                "description": "Recordings of work calls captured by the approved company telephony system, where that system records them. The app does not record calls.",
                "collected": True,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "android_call_recording",
                "label": "Android call recordings (your phone's recorder)",
                "description": (
                    "On an Android work phone whose own dialer records calls, after you opt in to this "
                    "category CIIS uploads the audio files your phone saved for work calls, so supervisors "
                    "can review them. CIIS does not record the calls itself — it uploads recordings your "
                    "phone already made — and only for calls placed after you opt in. Playback is limited to "
                    "authorized staff and every play is logged. You can withdraw this one category at any "
                    "time, and it is not available on iPhone."
                ),
                "collected": False,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "app_usage",
                "label": "App usage (Android)",
                "description": (
                    "On Android, after you opt in to this category and turn on Usage access for CIIS, "
                    "the app records which apps were in the foreground and for how long during your work "
                    "sessions, and uploads that when you open CIIS. It records app names and durations only "
                    "— never what is shown inside those apps, your messages or your keystrokes. Usage from "
                    "before you opt in is not uploaded. You can withdraw this one category at any time, and "
                    "it is not available on iPhone."
                ),
                "collected": False,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "in_app_screens",
                "label": "Screenshots of the CIIS app",
                "description": (
                    "After you opt in, CIIS may capture screenshots of its own screens — what you did inside "
                    "the CIIS app — and upload them. This does not capture any other app on your phone."
                ),
                "collected": False,
                "retention_days": ret,
                "measurement": "measured",
            },
            {
                "key": "screen_share",
                "label": "Screen share (you start it)",
                "description": (
                    "On Android, after you opt in, you can start a screen-share session with the control room. "
                    "You start each session yourself and Android shows a recording indicator the whole time; "
                    "CIIS cannot start it silently or hide that indicator. You can stop it at any time. Not "
                    "available on iPhone outside the CIIS app."
                ),
                "collected": False,
                "retention_days": ret,
                "measurement": "measured",
            },
        ]


class Command(BaseCommand):
    help = "Create and activate a default employee-monitoring policy."

    def add_arguments(self, parser):
        parser.add_argument("--policy-version", default="2026-10", help="Policy version string.")

    @transaction.atomic
    def handle(self, *args, **options):
        version = options["policy_version"]
        ret = retention_days() or None
        categories = default_categories(ret)
        MonitoringPolicy.objects.filter(is_active=True).update(is_active=False)
        policy, created = MonitoringPolicy.objects.update_or_create(
            version=version,
            defaults={
                "effective_at": timezone.now(),
                "summary": (
                    "CIIS collects the data listed below so supervisors can review duty records. "
                    "Records are visible to your supervisors, HR and authorized administrators for this deployment. "
                    "On an Android work phone, call history is uploaded only after you acknowledge this policy "
                    "and allow it on the phone. Calls placed before that acknowledgment are not uploaded. "
                    "The app does not record call audio. iPhone cellular call history cannot be read; "
                    "business calls on iPhone are reported only when they go through the company phone system."
                ),
                "work_hours": "During authorized work sessions only.",
                "contact": "Contact your HR or IT administrator for questions, data requests, or withdrawal.",
                "categories": categories,
                "is_active": True,
            },
        )
        self.stdout.write(self.style.SUCCESS(
            f"{'Created' if created else 'Updated'} and activated policy {version} for {deployment_id()}."
        ))
