import os
import re

from django.conf import settings
from rest_framework import serializers

from .models import RecoveryStage, VideoRecoveryJob

# Keys that carry server file-system locations; never sent to clients.
_PATH_KEY_RE = re.compile(r"(^|_)(path|paths|dir|file)$")


def _server_roots() -> list[str]:
    roots = {str(getattr(settings, "MEDIA_ROOT", "") or ""), str(getattr(settings, "BASE_DIR", "") or "")}
    out = []
    for root in roots:
        if root:
            out.extend({root, root.replace("\\", "/"), os.path.abspath(root)})
    return sorted({r for r in out if r}, key=len, reverse=True)


def scrub(value, roots=None):
    """Drop path-like keys and strip server directories out of report JSON (recursively)."""
    roots = _server_roots() if roots is None else roots
    if isinstance(value, dict):
        return {k: scrub(v, roots) for k, v in value.items() if not _PATH_KEY_RE.search(str(k))}
    if isinstance(value, list):
        return [scrub(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            if root in value:
                value = value.replace(root, "")
        return value
    return value


def damage_map_summary(damage_map: dict) -> dict:
    """Damage map without the per-frame list (counts, segments and strip only)."""
    return scrub({k: v for k, v in (damage_map or {}).items() if k != "frames"})


PIPELINE_STAGES = [
    {"key": RecoveryStage.UPLOAD, "label": "Video Upload", "phase": "intake"},
    {"key": RecoveryStage.VALIDATE, "label": "Upload Validation", "phase": "intake"},
    {"key": RecoveryStage.PRESERVE, "label": "Preserve Original", "phase": "intake"},
    {"key": RecoveryStage.DAMAGE_ANALYSIS, "label": "Deep Damage Analysis", "phase": "analysis"},
    {"key": RecoveryStage.DAMAGE_MAP, "label": "Damage Map", "phase": "analysis"},
    {"key": RecoveryStage.RECOVERY, "label": "Recovery Engine", "phase": "recovery"},
    {"key": RecoveryStage.REGENERATION, "label": "Regeneration Engine", "phase": "regeneration"},
    {"key": RecoveryStage.HYBRID_MERGE, "label": "Hybrid Merger", "phase": "merge"},
    {"key": RecoveryStage.TEMPORAL, "label": "Temporal Consistency", "phase": "merge"},
    {"key": RecoveryStage.AUDIO, "label": "Audio Recovery", "phase": "audio"},
    {"key": RecoveryStage.RECONSTRUCT, "label": "Video Encoding", "phase": "output"},
    {"key": RecoveryStage.VALIDATE_OUT, "label": "Quality Validation", "phase": "output"},
    {"key": RecoveryStage.COMPLETED, "label": "Final Output", "phase": "output"},
]


class VideoRecoveryJobSerializer(serializers.ModelSerializer):
    original_url = serializers.SerializerMethodField()
    recovered_url = serializers.SerializerMethodField()
    pipeline_stages = serializers.SerializerMethodField()

    class Meta:
        model = VideoRecoveryJob
        fields = [
            "id",
            "original_filename",
            "original_url",
            "recovered_url",
            "original_sha256",
            "status",
            "current_stage",
            "stage_logs",
            "forensic_report",
            "damage_map",
            "hybrid_report",
            "quality_report",
            "error_message",
            "pipeline_stages",
            "created_at",
            "updated_at",
            "completed_at",
        ]
        read_only_fields = fields

    def _media_url(self, file_field) -> str | None:
        if not file_field:
            return None
        request = self.context.get("request")
        url = file_field.url
        if request and url.startswith("/"):
            return request.build_absolute_uri(url)
        return url

    def to_representation(self, instance):
        data = super().to_representation(instance)
        roots = _server_roots()
        for key in ("stage_logs", "forensic_report", "damage_map", "hybrid_report", "quality_report", "error_message"):
            if key in data:
                data[key] = scrub(data[key], roots)
        return data

    def get_original_url(self, obj: VideoRecoveryJob) -> str | None:
        return self._media_url(obj.original_file)

    def get_recovered_url(self, obj: VideoRecoveryJob) -> str | None:
        return self._media_url(obj.recovered_file)

    def get_pipeline_stages(self, obj: VideoRecoveryJob) -> list[dict]:
        stage_order = [s["key"] for s in PIPELINE_STAGES]
        try:
            current_idx = stage_order.index(obj.current_stage)
        except ValueError:
            current_idx = 0
        completed_stages = {
            log.get("stage")
            for log in (obj.stage_logs or [])
            if log.get("status") != "failed"
        }
        logged_stages = {log.get("stage") for log in (obj.stage_logs or [])}

        out = []
        for idx, stage in enumerate(PIPELINE_STAGES):
            key = stage["key"]
            if obj.status == "completed" and key == RecoveryStage.COMPLETED:
                state = "completed"
            elif obj.status == "failed" and key == obj.current_stage:
                state = "failed"
            elif key in logged_stages and key != obj.current_stage:
                state = "completed"
            elif idx < current_idx:
                state = "completed"
            elif key == obj.current_stage and obj.status == "processing":
                state = "active"
            else:
                state = "pending"
            out.append({**stage, "state": state})
        return out


class VideoRecoveryJobListSerializer(VideoRecoveryJobSerializer):
    """List rows: status and summary only. Full reports come from the detail endpoint."""

    damage_map = serializers.SerializerMethodField()

    class Meta(VideoRecoveryJobSerializer.Meta):
        fields = [
            "id",
            "original_filename",
            "original_url",
            "recovered_url",
            "original_sha256",
            "status",
            "current_stage",
            "damage_map",
            "error_message",
            "created_at",
            "updated_at",
            "completed_at",
        ]
        read_only_fields = fields

    def get_damage_map(self, obj: VideoRecoveryJob) -> dict:
        return damage_map_summary(obj.damage_map or {})
