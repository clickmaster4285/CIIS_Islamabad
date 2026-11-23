from urllib.parse import urlsplit, urlunsplit

from rest_framework import serializers

from .models import (
    DeviceType,
    InfraAlert,
    InfraAlertRule,
    InfraDevice,
    InfraEvent,
    InfraSite,
)

# Shown instead of a stored secret. Sending it back unchanged keeps the stored value.
SECRET_MASK = "********"


def _split_url_password(url: str):
    """(parts, username, password) for a URL with user:pass@, else (None, '', '')."""
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return None, "", ""
    if parts.password is None:
        return None, parts.username or "", ""
    return parts, parts.username or "", parts.password or ""


def _with_url_password(parts, username: str, password: str) -> str:
    host = parts.netloc.rpartition("@")[2]  # host[:port] exactly as stored
    netloc = f"{username}:{password}@{host}" if username or password else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def mask_url_password(url: str) -> str:
    parts, user, password = _split_url_password(url)
    if parts is None or not password:
        return url or ""
    return _with_url_password(parts, user, SECRET_MASK)


class InfraSiteSerializer(serializers.ModelSerializer):
    device_count = serializers.SerializerMethodField()

    class Meta:
        model = InfraSite
        fields = [
            "id",
            "code",
            "name",
            "location",
            "description",
            "is_active",
            "device_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]

    def get_device_count(self, obj):
        # Annotated by InfraSiteViewSet; fall back to a query for other callers.
        annotated = getattr(obj, "active_device_count", None)
        if annotated is not None:
            return annotated
        return obj.devices.filter(is_active=True).count()


class InfraDeviceSerializer(serializers.ModelSerializer):
    site_code = serializers.CharField(source="site.code", read_only=True, default="")
    site_name = serializers.CharField(source="site.name", read_only=True, default="")
    device_type_label = serializers.CharField(
        source="get_device_type_display", read_only=True
    )
    primary_protocol_label = serializers.CharField(
        source="get_primary_protocol_display", read_only=True
    )
    status_label = serializers.CharField(source="get_status_display", read_only=True)
    password_set = serializers.SerializerMethodField()
    snmp_community_set = serializers.SerializerMethodField()
    rtsp_url_set = serializers.SerializerMethodField()

    class Meta:
        model = InfraDevice
        fields = [
            "id",
            "site",
            "site_code",
            "site_name",
            "device_type",
            "device_type_label",
            "name",
            "manufacturer",
            "model_number",
            "serial_number",
            "asset_tag",
            "ip_address",
            "port",
            "mac_address",
            "install_location",
            "primary_protocol",
            "primary_protocol_label",
            "secondary_protocol",
            "username",
            "password",
            "password_set",
            "snmp_version",
            "snmp_community",
            "snmp_community_set",
            "snmp_port",
            "snmp_oid_map",
            "modbus_unit_id",
            "modbus_port",
            "modbus_register_map",
            "onvif_port",
            "rtsp_url",
            "rtsp_url_set",
            "channel_count",
            "supports_onvif",
            "supports_rtsp",
            "supports_snmp",
            "supports_modbus",
            "rated_capacity_kva",
            "rated_power_kw",
            "network_role",
            "poll_interval_sec",
            "status",
            "status_label",
            "last_seen_at",
            "last_polled_at",
            "last_error",
            "last_metrics",
            "source_key",
            "is_active",
            "notes",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "status",
            "last_seen_at",
            "last_polled_at",
            "last_error",
            "last_metrics",
            "source_key",
            "created_at",
            "updated_at",
            "password_set",
            "snmp_community_set",
            "rtsp_url_set",
        ]
        extra_kwargs = {
            "password": {"write_only": True, "required": False, "allow_blank": True},
        }

    def get_password_set(self, obj):
        return bool(obj.password)

    def get_snmp_community_set(self, obj):
        return bool(obj.snmp_community)

    def get_rtsp_url_set(self, obj):
        return bool(obj.rtsp_url)

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Never echo secrets: the edit form gets a mask it can send back unchanged.
        data["snmp_community"] = SECRET_MASK if instance.snmp_community else ""
        data["rtsp_url"] = mask_url_password(instance.rtsp_url)
        return data

    def validate(self, attrs):
        attrs = super().validate(attrs)
        instance = self.instance
        community = attrs.get("snmp_community")
        if community == SECRET_MASK:
            if instance is None:
                raise serializers.ValidationError({"snmp_community": "Enter the SNMP community."})
            attrs.pop("snmp_community")
        url = attrs.get("rtsp_url")
        if url:
            parts, user, password = _split_url_password(url)
            if parts is not None and password == SECRET_MASK:
                if instance is None:
                    raise serializers.ValidationError({"rtsp_url": "Enter the RTSP password."})
                # Masked password sent back: keep the stored one (host/path may have changed).
                _old_parts, _old_user, old_password = _split_url_password(instance.rtsp_url)
                attrs["rtsp_url"] = _with_url_password(parts, user, old_password)
        return attrs

    @staticmethod
    def _flag_camera_own_creds(validated_data, instance=None) -> None:
        """A camera only gets direct ISAPI calls when an admin saved its own login."""
        device_type = validated_data.get("device_type") or getattr(instance, "device_type", "")
        if device_type != DeviceType.CAMERA or not validated_data.get("password"):
            return
        meta = validated_data.get("metadata")
        if meta is None:
            meta = dict(instance.metadata) if instance is not None and isinstance(instance.metadata, dict) else {}
        meta = {**meta, "camera_has_own_creds": True}
        validated_data["metadata"] = meta

    def create(self, validated_data):
        self._flag_camera_own_creds(validated_data)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Keep existing password when client omits or sends blank
        if "password" in validated_data and not validated_data["password"]:
            validated_data.pop("password")
        self._flag_camera_own_creds(validated_data, instance)
        return super().update(instance, validated_data)


class InfraAlertRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = InfraAlertRule
        fields = [
            "id",
            "name",
            "device_type",
            "metric_key",
            "operator",
            "threshold_value",
            "severity",
            "message_template",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]


class InfraAlertSerializer(serializers.ModelSerializer):
    device_name = serializers.CharField(source="device.name", read_only=True, default="")
    device_type = serializers.CharField(
        source="device.device_type", read_only=True, default=""
    )
    severity_label = serializers.CharField(
        source="get_severity_display", read_only=True
    )

    class Meta:
        model = InfraAlert
        fields = [
            "id",
            "device",
            "device_name",
            "device_type",
            "alert_type",
            "severity",
            "severity_label",
            "title",
            "message",
            "metric_key",
            "metric_value",
            "acknowledged",
            "acknowledged_at",
            "resolved",
            "resolved_at",
            "first_detected",
            "last_detected",
            "details",
        ]
        read_only_fields = [
            "first_detected",
            "last_detected",
            "acknowledged_at",
            "resolved_at",
        ]


class InfraEventSerializer(serializers.ModelSerializer):
    device_name = serializers.CharField(source="device.name", read_only=True, default="")

    class Meta:
        model = InfraEvent
        fields = [
            "id",
            "device",
            "device_name",
            "event_type",
            "title",
            "message",
            "actor",
            "payload",
            "created_at",
        ]
        read_only_fields = ["created_at"]
