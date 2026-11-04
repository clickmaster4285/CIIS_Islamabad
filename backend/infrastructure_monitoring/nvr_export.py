"""Excel export: NVR inventory + every camera connected to each NVR.

Sheets:
  1. NVRs      — one row per NVR (device info, health, channels, storage totals)
  2. Cameras   — one row per camera connected to the NVR, with the full device
                 information Hikvision reports (identity, MAC, network, system,
                 streams, image settings, smart capabilities)
  3. Storage   — one row per hard disk inside each NVR

Camera data is read live at export time:
  * NVR bulk calls  — InputProxy channel list / status / streaming proxy
  * per camera      — Hikvision ISAPI pages fetched THROUGH the NVR using the
                      channel's devIndex (``?devIndex=``), so cameras on a
                      private NVR PoE network still answer. The NVR login is
                      only ever sent to the NVR itself (never to camera IPs).
All requests are read-only GETs with HTTP Digest auth only (no Basic auth, so
the password is never sent in clear text). A camera that stops answering is
skipped after its first failed request, and the whole live read has a time
budget so the download always finishes. If an NVR cannot be reached, the last saved
poll data is used and the "Data Source" column says so.

Credentials (passwords, RTSP URLs, SNMP community) are never written.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import requests
from django.utils import timezone
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from requests.auth import HTTPDigestAuth

from .models import InfraDevice
from .nvr_health import (
    _find_direct,
    _find_text,
    _nvr_http_creds,
    _tag,
    _truthy,
    compute_nvr_health,
    normalize_hdd_list,
    resolve_cameras_nvr_id,
)

logger = logging.getLogger(__name__)

LIVE_TIMEOUT_SEC = 8
CAMERA_TIMEOUT_SEC = 5
MAX_PARALLEL_NVRS = 4
MAX_PARALLEL_CAMERAS_PER_NVR = 6
EXPORT_LIVE_BUDGET_SEC = 90  # total time for all live reads in one export

HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")
SECTION_FONT = Font(bold=True, color="FFFFFF")
GROUP_FILL = PatternFill("solid", fgColor="DCE6F1")
GROUP_FONT = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")

LEVEL_FILLS = {
    "healthy": PatternFill("solid", fgColor="C6EFCE"),
    "warning": PatternFill("solid", fgColor="FFEB9C"),
    "critical": PatternFill("solid", fgColor="FFC7CE"),
}
STATUS_FILLS = {
    "Online": LEVEL_FILLS["healthy"],
    "Video loss": LEVEL_FILLS["warning"],
    "Offline": LEVEL_FILLS["critical"],
}
CHANNEL_STATUS_LABELS = {
    "online": "Online",
    "offline": "Offline",
    "video_loss": "Video loss",
}


# ---------------------------------------------------------------- helpers


def _get(ip: str, path: str, *, username: str, password: str, port: int, timeout: float,
         deadline: float | None = None) -> tuple[int | None, str]:
    """Read-only GET, Digest auth only. (None, err) on network error/timeout/budget."""
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            return None, "time budget exceeded"
        timeout = min(timeout, remaining)
    try:
        res = requests.get(
            f"http://{ip}:{port}{path}",
            auth=HTTPDigestAuth(username, password or ""),
            timeout=timeout,
        )
        return res.status_code, res.text or ""
    except requests.RequestException as exc:
        logger.debug("NVR export GET %s%s failed: %s", ip, path, exc)
        return None, str(exc)[:120]


def _fmt_dt(value) -> str:
    if not value:
        return ""
    try:
        return timezone.localtime(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def _gb(mb) -> float | str:
    try:
        mb = float(mb)
    except (TypeError, ValueError):
        return ""
    return round(mb / 1024.0, 1) if mb >= 0 else ""


def _val(*values) -> Any:
    """First value that is not None/'' — keeps real zeros (0 °C, 0 %, 0 disks)."""
    for v in values:
        if v is not None and v != "":
            return v
    return ""


def _num(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fps(raw) -> float | str:
    """Hikvision maxFrameRate is hundredths of fps (1200 → 12.0)."""
    v = _num(raw)
    if v is None or v < 0:
        return ""
    return round(v / 100.0, 1) if v > 120 else round(v, 1)


def _uptime(raw) -> str:
    secs = _num(raw)
    if secs is None:
        return ""
    secs = int(secs)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"


def _yes_no(raw) -> str:
    if raw in (None, ""):
        return ""
    return "Yes" if _truthy(str(raw)) else "No"


def _metrics(device: InfraDevice) -> dict[str, Any]:
    return device.last_metrics if isinstance(device.last_metrics, dict) else {}


def _logical_channel(channel) -> int | None:
    # Hikvision stream IDs (101, 201, …) → logical channel 1, 2, …
    try:
        ch = int(float(str(channel).strip()))
    except (TypeError, ValueError):
        return None
    return ch // 100 if ch >= 100 else ch


def _parse_xml(body: str | None):
    if not body:
        return None
    try:
        return ET.fromstring(body)
    except ET.ParseError:
        return None


def _child(root, *path: str):
    """First descendant matching the tag path, e.g. _child(root, "Link")."""
    node = root
    for name in path:
        want = name.lower()
        found = None
        if node is None:
            return None
        for el in node.iter():
            if el is not node and _tag(el).lower() == want:
                found = el
                break
        node = found
    return node


def _txt(root, *path: str) -> str:
    """Text of the element at a tag path (last name is the leaf)."""
    if root is None:
        return ""
    *parents, leaf = path
    node = _child(root, *parents) if parents else root
    if node is None:
        return ""
    return _find_text(node, leaf)


# ------------------------------------------------------- live NVR channels


def fetch_live_channels(
    device: InfraDevice, deadline: float | None = None
) -> tuple[dict[int, dict[str, Any]], str]:
    """Bulk channel list from the NVR (3 GETs). Returns ({channel: info}, error)."""
    ip, user, password, port = _nvr_http_creds(device)
    if not ip or not user:
        return {}, "NVR IP or username missing"
    kw = {
        "username": user, "password": password, "port": port,
        "timeout": LIVE_TIMEOUT_SEC, "deadline": deadline,
    }

    code, body = _get(ip, "/ISAPI/ContentMgmt/InputProxy/channels", **kw)
    root = _parse_xml(body) if code == 200 else None
    if root is None:
        return {}, f"NVR not reachable (HTTP {code})" if code else "NVR not reachable"

    out: dict[int, dict[str, Any]] = {}
    for ch in root.iter():
        if _tag(ch).lower() != "inputproxychannel":
            continue
        cid = _logical_channel(_find_direct(ch, "id"))
        if cid is None:
            continue
        out[cid] = {
            "channel_name": _find_direct(ch, "name"),
            "camera_ip": _find_text(ch, "ipAddress", "ipv4Address", "hostName"),
            "manage_port": _find_text(ch, "managePortNo"),
            "protocol": _find_text(ch, "proxyProtocol"),
            "model": _find_text(ch, "model"),
            "serial_number": _find_text(ch, "serialNumber"),
            "firmware": _find_text(ch, "firmwareVersion"),
            "dev_index": _find_direct(ch, "devIndex"),
        }

    code, body = _get(ip, "/ISAPI/ContentMgmt/InputProxy/channels/status", **kw)
    root = _parse_xml(body) if code == 200 else None
    if root is not None:
        for st in root.iter():
            if _tag(st).lower() != "inputproxychannelstatus":
                continue
            cid = _logical_channel(_find_direct(st, "id"))
            if cid is None:
                continue
            info = out.setdefault(cid, {})
            online_raw = _find_direct(st, "online")
            detect = _find_direct(st, "chanDetectResult")
            online = _truthy(online_raw) if online_raw else False
            if online and detect.lower() in ("videoloss", "video_loss", "loss", "abnormal"):
                status = "Video loss"
            elif online:
                status = "Online"
            else:
                status = "Offline"
            info["status"] = status
            info["connection_detail"] = detect
            info["password_strength"] = _find_text(st, "PasswordStatus")

    code, body = _get(ip, "/ISAPI/ContentMgmt/StreamingProxy/channels", **kw)
    root = _parse_xml(body) if code == 200 else None
    if root is not None:
        for sc in root.iter():
            if _tag(sc).lower() != "streamingchannel":
                continue
            raw_id = _num(_find_direct(sc, "id"))
            if raw_id is None or raw_id < 100:
                continue
            cid, stream_no = int(raw_id) // 100, int(raw_id) % 100
            prefix = {1: "nvr_main", 2: "nvr_sub"}.get(stream_no)
            if prefix:
                out.setdefault(cid, {})[prefix] = _parse_stream(sc)

    return out, ""


# ------------------------------------------------- per-camera Hikvision info


def _parse_stream(sc) -> dict[str, Any]:
    video = _child(sc, "Video")
    audio = _child(sc, "Audio")
    w = _txt(video, "videoResolutionWidth")
    h = _txt(video, "videoResolutionHeight")
    mode = _txt(video, "videoQualityControlType")
    bitrate = (
        _txt(video, "constantBitRate") if mode.upper() == "CBR" else _txt(video, "vbrUpperCap")
    ) or _txt(video, "constantBitRate")
    smart = _child(video, "SmartCodec") if video is not None else None
    return {
        "enabled": _yes_no(_find_direct(sc, "enabled")),
        "resolution": f"{w}x{h}" if w and h else "",
        "codec": _txt(video, "videoCodecType"),
        "profile": _txt(video, "H265Profile") or _txt(video, "H264Profile"),
        "fps": _fps(_txt(video, "maxFrameRate")),
        "bitrate_mode": mode,
        "bitrate": bitrate,
        "avg_bitrate": _txt(video, "vbrAverageCap"),
        "quality": _txt(video, "fixedQuality"),
        "gop": _txt(video, "GovLength"),
        "smart_codec": _yes_no(_find_direct(smart, "enabled")) if smart is not None else "",
        "scan": _txt(video, "videoScanType"),
        "transport": _txt(sc, "rtpTransportType"),
        "audio": (
            _txt(audio, "audioCompressionType")
            if audio is not None and _truthy(_find_direct(audio, "enabled") or "false")
            else ("Off" if audio is not None else "")
        ),
    }


def _camera_getter(nvr: InfraDevice, info: dict[str, Any], deadline: float | None):
    """get(path) -> root|None for one camera, through the NVR (devIndex passthrough).

    Stops after the first network failure/timeout, so a dead camera costs one
    timeout instead of one per page.
    """
    ip, user, password, port = _nvr_http_creds(nvr)
    dev_index = (info.get("dev_index") or "").strip()
    state = {"dead": not (ip and user and dev_index)}

    def get(path: str):
        if state["dead"]:
            return None
        sep = "&" if "?" in path else "?"
        code, body = _get(
            ip, f"{path}{sep}devIndex={dev_index}",
            username=user, password=password, port=port,
            timeout=CAMERA_TIMEOUT_SEC, deadline=deadline,
        )
        if code is None:
            state["dead"] = True
            return None
        return _parse_xml(body) if code == 200 else None

    return get


def fetch_camera_details(
    nvr: InfraDevice, channel: int, info: dict[str, Any], deadline: float | None = None
) -> dict[str, Any]:
    """Full Hikvision ISAPI detail for one camera (read-only, via the NVR)."""
    out: dict[str, Any] = {}
    get = _camera_getter(nvr, info, deadline)
    dev_root = get("/ISAPI/System/deviceInfo")
    if dev_root is not None and _tag(dev_root).lower() != "deviceinfo":
        dev_root = None

    if dev_root is None:
        # Fallback: NVR's own per-channel device summary (has MAC on most models)
        ip, user, password, port = _nvr_http_creds(nvr)
        code, body = _get(
            ip, f"/ISAPI/ContentMgmt/InputProxy/channels/{channel}/deviceInfo",
            username=user, password=password, port=port,
            timeout=CAMERA_TIMEOUT_SEC, deadline=deadline,
        )
        root = _parse_xml(body) if code == 200 else None
        if root is not None:
            out.update(
                device_name=_find_text(root, "deviceName"),
                model=_find_text(root, "model"),
                serial_number=_find_text(root, "serialNumber"),
                mac=_find_text(root, "macAddress"),
                firmware=_find_text(root, "firmwareVersion"),
                via="NVR channel summary",
            )
        return out

    out["via"] = "Live via NVR"
    for key, tag in (
        ("device_name", "deviceName"),
        ("device_id", "deviceID"),
        ("device_description", "deviceDescription"),
        ("device_location", "deviceLocation"),
        ("system_contact", "systemContact"),
        ("manufacturer", "manufacturer"),
        ("model", "model"),
        ("serial_number", "serialNumber"),
        ("sub_serial", "subSerialNumber"),
        ("mac", "macAddress"),
        ("firmware", "firmwareVersion"),
        ("firmware_date", "firmwareReleasedDate"),
        ("firmware_info", "firmwareVersionInfo"),
        ("encoder", "encoderVersion"),
        ("encoder_date", "encoderReleasedDate"),
        ("boot", "bootVersion"),
        ("boot_date", "bootReleasedDate"),
        ("hardware", "hardwareVersion"),
        ("device_type", "deviceType"),
        ("telecontrol_id", "telecontrolID"),
        ("oem_code", "OEMCode"),
    ):
        out[key] = _find_direct(dev_root, tag)

    net = get("/ISAPI/System/Network/interfaces")
    if net is not None:
        iface = _child(net, "NetworkInterface")
        if iface is None:
            iface = net
        addr = _child(iface, "IPAddress")
        link = _child(iface, "Link")
        out.update(
            net_ip=_find_direct(addr, "ipAddress") if addr is not None else "",
            net_addressing=_txt(addr, "addressingType"),
            net_ip_version=_txt(addr, "ipVersion"),
            net_mask=_txt(addr, "subnetMask"),
            net_gateway=_txt(addr, "DefaultGateway", "ipAddress"),
            net_dns1=_txt(addr, "PrimaryDNS", "ipAddress"),
            net_dns2=_txt(addr, "SecondaryDNS", "ipAddress"),
            net_ipv6=_txt(addr, "ipv6Address"),
            net_link_mac=_txt(link, "MACAddress"),
            net_speed=_txt(link, "speed"),
            net_duplex=_txt(link, "duplex"),
            net_autoneg=_yes_no(_txt(link, "autoNegotiation")),
            net_mtu=_txt(link, "MTU"),
            net_upnp=_yes_no(_txt(iface, "UPnP", "enabled")),
        )
        if not out.get("mac"):
            out["mac"] = out["net_link_mac"]

    st = get("/ISAPI/System/status")
    if st is not None:
        out.update(
            sys_time=_find_text(st, "currentDeviceTime"),
            sys_uptime=_uptime(_find_text(st, "deviceUpTime")),
            sys_cpu=_find_text(st, "cpuUtilization"),
            sys_cpu_desc=_find_text(st, "cpuDescription"),
            sys_mem_usage=_find_text(st, "memoryUsage"),
            sys_mem_avail=_find_text(st, "memoryAvailable"),
        )

    tm = get("/ISAPI/System/time")
    if tm is not None:
        out.update(time_mode=_find_text(tm, "timeMode"), time_zone=_find_text(tm, "timeZone"))
    ntp = get("/ISAPI/System/time/ntpServers")
    if ntp is not None:
        out["ntp"] = _find_text(ntp, "hostName", "ipAddress")

    for stream_no, prefix in ((1, "main"), (2, "sub")):
        sc = get(f"/ISAPI/Streaming/channels/1{stream_no:02d}")
        if sc is not None:
            out[prefix] = _parse_stream(sc)

    img = get("/ISAPI/Image/channels/1")
    if img is not None:
        out.update(
            img_daynight=_txt(img, "IrcutFilter", "IrcutFilterType"),
            img_exposure=_txt(img, "Exposure", "ExposureType"),
            img_shutter=_txt(img, "Shutter", "ShutterLevel"),
            img_gain=_txt(img, "Gain", "GainLevel"),
            img_wdr=_txt(img, "WDR", "mode"),
            img_blc=_yes_no(_txt(img, "BLC", "enabled")),
            img_hlc=_yes_no(_txt(img, "HLC", "enabled")),
            img_noise=_txt(img, "NoiseReduce", "mode"),
            img_wb=_txt(img, "WhiteBalance", "WhiteBalanceStyle"),
            img_brightness=_txt(img, "Color", "brightnessLevel"),
            img_contrast=_txt(img, "Color", "contrastLevel"),
            img_saturation=_txt(img, "Color", "saturationLevel"),
            img_sharpness=_txt(img, "Sharpness", "SharpnessLevel"),
            img_flip=_yes_no(_txt(img, "ImageFlip", "enabled")),
            img_powerline=_txt(img, "powerLineFrequency", "powerLineFrequencyMode"),
            img_light=_txt(img, "SupplementLight", "supplementLightMode"),
        )

    smart = get("/ISAPI/Smart/capabilities")
    if smart is not None:
        names = []
        for el in smart:
            tag = _tag(el)
            if tag.startswith("isSupport") and _truthy((el.text or "").strip()):
                names.append(tag[len("isSupport"):])
        out["smart"] = ", ".join(names) or "None"

    return out


def _fetch_nvr_cameras(
    device: InfraDevice, deadline: float | None = None
) -> tuple[dict[int, dict], dict[int, dict], str]:
    """Live channels + per-camera details for one NVR."""
    channels, error = fetch_live_channels(device, deadline)
    if not channels:
        return {}, {}, error

    def run(item):
        ch, info = item
        # Offline channels: the NVR answers "channel is offline" and the camera
        # cannot be reached — skip to keep the export fast.
        if info.get("status") == "Offline":
            return ch, {}
        try:
            return ch, fetch_camera_details(device, ch, info, deadline)
        except Exception as exc:
            logger.warning("NVR export camera %s ch%s failed: %s", device.pk, ch, exc)
            return ch, {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CAMERAS_PER_NVR) as pool:
        details = dict(pool.map(run, sorted(channels.items())))
    return channels, details, ""


def _fetch_all_live(devices: list[InfraDevice]) -> dict[int, tuple[dict, dict, str]]:
    if not devices:
        return {}
    deadline = time.monotonic() + EXPORT_LIVE_BUDGET_SEC

    def run(device):
        try:
            return device.id, _fetch_nvr_cameras(device, deadline)
        except Exception as exc:  # never break the export on one bad NVR
            logger.warning("NVR export live fetch failed for %s: %s", device.pk, exc)
            return device.id, ({}, {}, "Live read failed")

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_NVRS, len(devices))) as pool:
        return dict(pool.map(run, devices))


def _registered_counts(devices: list[InfraDevice]) -> dict[int, int]:
    """InfraDevice.id → number of cameras added on the linked NVR (NVR sheet only)."""
    from django.db.models import Count

    from cameras.models import Camera

    link = {d.id: resolve_cameras_nvr_id(d) for d in devices}
    ids = {v for v in link.values() if v}
    if not ids:
        return {}
    counts = dict(
        Camera.objects.filter(nvr_id__in=ids).values_list("nvr_id").annotate(c=Count("id"))
    )
    return {dev_id: counts.get(nvr_id, 0) for dev_id, nvr_id in link.items() if nvr_id}


# ----------------------------------------------------------------- sheets


def _style_header(ws, headers: list[str]) -> None:
    ws.append(headers)
    for cell in ws[ws.max_row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER


def _append_row(ws, values: list[Any]) -> None:
    ws.append(values)
    for cell in ws[ws.max_row]:
        cell.border = BORDER


def _autosize(ws, max_width: int = 45, skip_rows: set[int] | None = None) -> None:
    widths: dict[int, int] = {}
    for row in ws.iter_rows():
        for cell in row:
            if skip_rows and cell.row in skip_rows:
                continue
            if cell.value is None:
                continue
            widths[cell.column] = max(widths.get(cell.column, 0), len(str(cell.value)))
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = min(max(width + 2, 8), max_width)


NVR_HEADERS = [
    "Site",
    "NVR Name",
    "IP Address",
    "Manufacturer",
    "Model",
    "Serial Number",
    "Firmware",
    "MAC Address",
    "Install Location",
    "Asset Tag",
    "Status",
    "Overall Health",
    "Last Seen",
    "Last Polled",
    "Uptime",
    "Cameras Connected",
    "Cameras Online",
    "Cameras Offline",
    "Cameras Video Loss",
    "Cameras Recording",
    "Registered in TekEye",
    "HDD Count",
    "Storage Health",
    "Storage Detail",
    "Total Capacity (GB)",
    "Free (GB)",
    "Temperature (°C)",
    "CPU %",
    "Camera Data Source",
    "Last Error",
    "Notes",
]


def _stream_cols(label: str) -> list[tuple[str, Any]]:
    return [
        (f"{label} Enabled", lambda r, k=label: r["_s"][k].get("enabled")),
        (f"{label} Resolution", lambda r, k=label: r["_s"][k].get("resolution")),
        (f"{label} Codec", lambda r, k=label: r["_s"][k].get("codec")),
        (f"{label} Profile", lambda r, k=label: r["_s"][k].get("profile")),
        (f"{label} FPS", lambda r, k=label: r["_s"][k].get("fps")),
        (f"{label} Bitrate Mode", lambda r, k=label: r["_s"][k].get("bitrate_mode")),
        (f"{label} Max Bitrate (kbps)", lambda r, k=label: r["_s"][k].get("bitrate")),
        (f"{label} Avg Bitrate (kbps)", lambda r, k=label: r["_s"][k].get("avg_bitrate")),
        (f"{label} Quality", lambda r, k=label: r["_s"][k].get("quality")),
        (f"{label} I-Frame Interval (GOP)", lambda r, k=label: r["_s"][k].get("gop")),
        (f"{label} Smart Codec", lambda r, k=label: r["_s"][k].get("smart_codec")),
        (f"{label} Scan Type", lambda r, k=label: r["_s"][k].get("scan")),
        (f"{label} Transport", lambda r, k=label: r["_s"][k].get("transport")),
        (f"{label} Audio", lambda r, k=label: r["_s"][k].get("audio")),
    ]


def _d(key: str):
    return lambda r: r["d"].get(key)


def _i(key: str):
    return lambda r: r["i"].get(key)


# (section, [(header, getter(row_ctx))])
CAMERA_SECTIONS: list[tuple[str, list[tuple[str, Any]]]] = [
    (
        "NVR / Channel",
        [
            ("Site", lambda r: r["site"]),
            ("NVR Name", lambda r: r["nvr"].name),
            ("NVR IP", lambda r: r["nvr"].ip_address or ""),
            ("Channel", lambda r: r["ch"]),
            ("Channel Name", lambda r: r["i"].get("channel_name") or r["sv"].get("name")),
            ("Status", lambda r: r["status"]),
            ("Connection Detail", _i("connection_detail")),
            ("Password Strength", _i("password_strength")),
            ("Recording", lambda r: ("Yes" if r["sv"].get("recording") else "No") if r["sv"] else ""),
            ("Protocol", _i("protocol")),
            ("Manage Port", _i("manage_port")),
        ],
    ),
    (
        "Device Information",
        [
            ("Camera IP", lambda r: r["d"].get("net_ip") or r["i"].get("camera_ip") or r["sv"].get("code")),
            ("MAC Address", _d("mac")),
            ("Device Name", _d("device_name")),
            ("Device ID", _d("device_id")),
            ("Device Type", _d("device_type")),
            ("Description", _d("device_description")),
            ("Location", _d("device_location")),
            ("Manufacturer", _d("manufacturer")),
            ("System Contact", _d("system_contact")),
            ("Model", lambda r: r["d"].get("model") or r["i"].get("model")),
            ("Serial Number", lambda r: r["d"].get("serial_number") or r["i"].get("serial_number")),
            ("Sub Serial Number", _d("sub_serial")),
            ("Firmware Version", lambda r: r["d"].get("firmware") or r["i"].get("firmware")),
            ("Firmware Build", _d("firmware_date")),
            ("Firmware Info", _d("firmware_info")),
            ("Encoder Version", _d("encoder")),
            ("Encoder Build", _d("encoder_date")),
            ("Boot Version", _d("boot")),
            ("Boot Build", _d("boot_date")),
            ("Hardware Version", _d("hardware")),
            ("Telecontrol ID", _d("telecontrol_id")),
            ("OEM Code", _d("oem_code")),
        ],
    ),
    (
        "Network",
        [
            ("Addressing", _d("net_addressing")),
            ("IP Version", _d("net_ip_version")),
            ("Subnet Mask", _d("net_mask")),
            ("Default Gateway", _d("net_gateway")),
            ("Primary DNS", _d("net_dns1")),
            ("Secondary DNS", _d("net_dns2")),
            ("IPv6 Address", _d("net_ipv6")),
            ("Link Speed (Mbps)", _d("net_speed")),
            ("Duplex", _d("net_duplex")),
            ("Auto Negotiation", _d("net_autoneg")),
            ("MTU", _d("net_mtu")),
            ("UPnP", _d("net_upnp")),
        ],
    ),
    (
        "System",
        [
            ("Device Time", _d("sys_time")),
            ("Uptime", _d("sys_uptime")),
            ("CPU %", _d("sys_cpu")),
            ("CPU", _d("sys_cpu_desc")),
            ("Memory Usage", _d("sys_mem_usage")),
            ("Memory Available", _d("sys_mem_avail")),
            ("Time Mode", _d("time_mode")),
            ("Time Zone", _d("time_zone")),
            ("NTP Server", _d("ntp")),
        ],
    ),
    ("Main Stream", _stream_cols("Main")),
    ("Sub Stream", _stream_cols("Sub")),
    (
        "Image Settings",
        [
            ("Day/Night (IR-Cut)", _d("img_daynight")),
            ("Exposure", _d("img_exposure")),
            ("Shutter", _d("img_shutter")),
            ("Gain", _d("img_gain")),
            ("WDR", _d("img_wdr")),
            ("BLC", _d("img_blc")),
            ("HLC", _d("img_hlc")),
            ("Noise Reduction", _d("img_noise")),
            ("White Balance", _d("img_wb")),
            ("Brightness", _d("img_brightness")),
            ("Contrast", _d("img_contrast")),
            ("Saturation", _d("img_saturation")),
            ("Sharpness", _d("img_sharpness")),
            ("Image Flip", _d("img_flip")),
            ("Power Line Frequency", _d("img_powerline")),
            ("Supplement Light", _d("img_light")),
        ],
    ),
    (
        "Other",
        [
            ("Smart Features Supported", _d("smart")),
            ("Data Source", lambda r: r["source"]),
        ],
    ),
]

CAMERA_COLUMNS = [col for _section, cols in CAMERA_SECTIONS for col in cols]
CAMERA_HEADERS = [h for h, _g in CAMERA_COLUMNS]
SECTION_FILLS = [
    "1F3A5F", "2E5E4E", "5B3A70", "7A4B1E", "1E5A7A", "35607A", "6B3A3A", "4A4A4A",
]

HDD_HEADERS = [
    "Site",
    "NVR Name",
    "Disk",
    "Status",
    "Capacity (GB)",
    "Free (GB)",
    "Usage",
    "Type",
    "Attribute",
    "Disk Model",
    "Disk Serial",
]


def _camera_header(ws) -> int:
    """Two-row header: section band + column names. Returns header row count."""
    col = 1
    section_row: list[Any] = []
    spans = []
    for idx, (section, cols) in enumerate(CAMERA_SECTIONS):
        section_row.extend([section] + [None] * (len(cols) - 1))
        spans.append((col, col + len(cols) - 1, SECTION_FILLS[idx % len(SECTION_FILLS)]))
        col += len(cols)
    ws.append(section_row)
    for start, end, color in spans:
        fill = PatternFill("solid", fgColor=color)
        for c in range(start, end + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = fill
            cell.font = SECTION_FONT
            cell.alignment = CENTER
            cell.border = BORDER
        if end > start:
            ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=end)
    _style_header(ws, CAMERA_HEADERS)
    for start, end, color in spans:
        for c in range(start, end + 1):
            ws.cell(row=2, column=c).fill = PatternFill("solid", fgColor=color)
    ws.freeze_panes = "F3"  # keep NVR / channel / name / status visible
    return 2


def _saved_channel_items(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    channels = metrics.get("channels") if isinstance(metrics.get("channels"), dict) else {}
    out: dict[int, dict[str, Any]] = {}
    for item in channels.get("items") or []:
        if not isinstance(item, dict):
            continue
        ch = _logical_channel(item.get("channel") or item.get("id"))
        if ch is not None:
            out[ch] = item
    return out


def _build_camera_rows(device, site_name, metrics, live, details, live_error):
    saved = _saved_channel_items(metrics)
    # Live list is the truth; saved poll only when the NVR could not be read
    channel_ids = sorted(live) if live else sorted(saved)
    rows: list[list[Any]] = []
    status_by_ch: dict[int, str] = {}
    empty_stream: dict[str, Any] = {}

    for ch in channel_ids:
        info = live.get(ch, {})
        sv = saved.get(ch, {})
        d = details.get(ch, {})
        status = info.get("status") or CHANNEL_STATUS_LABELS.get(str(sv.get("status") or ""), "")
        status_by_ch[ch] = status
        if not live:
            source = f"Saved data ({live_error or 'not live'})"
        elif d.get("via"):
            source = d["via"]
        elif status == "Offline":
            source = "Camera offline — NVR data only"
        else:
            source = "Camera did not answer — NVR data only"
        ctx = {
            "site": site_name,
            "nvr": device,
            "ch": ch,
            "i": info,
            "sv": sv,
            "d": d,
            "status": status,
            "source": source,
            "_s": {
                "Main": d.get("main") or info.get("nvr_main") or empty_stream,
                "Sub": d.get("sub") or info.get("nvr_sub") or empty_stream,
            },
        }
        rows.append([("" if (v := getter(ctx)) is None else v) for _h, getter in CAMERA_COLUMNS])

    counts = list(status_by_ch.values())
    summary = {
        "connected": len(channel_ids),
        "online": counts.count("Online"),
        "offline": counts.count("Offline"),
        "video_loss": counts.count("Video loss"),
        "recording": sum(1 for it in saved.values() if it.get("recording")),
        "source": "Live from NVR" if live else f"Saved data ({live_error or 'not live'})",
    }
    return rows, summary


def build_nvr_export(devices: list[InfraDevice], *, live: bool = True) -> bytes:
    registered = _registered_counts(devices)
    live_data = _fetch_all_live(devices) if live else {}

    wb = Workbook()
    ws_nvr = wb.active
    ws_nvr.title = "NVRs"
    ws_cam = wb.create_sheet("Cameras")
    ws_hdd = wb.create_sheet("Storage")

    _style_header(ws_nvr, NVR_HEADERS)
    ws_nvr.freeze_panes = "A2"
    cam_header_rows = _camera_header(ws_cam)
    _style_header(ws_hdd, HDD_HEADERS)
    ws_hdd.freeze_panes = "A2"

    health_col = NVR_HEADERS.index("Overall Health") + 1
    cam_status_col = CAMERA_HEADERS.index("Status") + 1
    cam_skip_rows: set[int] = set(range(1, cam_header_rows))

    for device in devices:
        metrics = _metrics(device)
        health = compute_nvr_health(metrics, device_status=device.status)
        hdds = normalize_hdd_list(
            metrics.get("hdd_list") if isinstance(metrics.get("hdd_list"), list) else []
        )
        site_name = device.site.name if device.site_id else ""
        live_channels, details, live_error = live_data.get(device.id, ({}, {}, "live read skipped"))

        cam_rows, summary = _build_camera_rows(
            device, site_name, metrics, live_channels, details, live_error
        )

        _append_row(
            ws_nvr,
            [
                site_name,
                device.name,
                device.ip_address or "",
                device.manufacturer,
                device.model_number or metrics.get("device_model") or "",
                device.serial_number or metrics.get("serial_number") or "",
                metrics.get("firmware_version") or "",
                device.mac_address or metrics.get("mac_address") or "",
                device.install_location,
                device.asset_tag,
                device.get_status_display(),
                health.get("overall_label", ""),
                _fmt_dt(device.last_seen_at),
                _fmt_dt(device.last_polled_at),
                metrics.get("uptime") or "",
                summary["connected"],
                summary["online"],
                summary["offline"],
                summary["video_loss"],
                summary["recording"],
                registered.get(device.id, 0),
                len(hdds) if hdds else _val(metrics.get("hdd_count")),
                health.get("storage", {}).get("label", ""),
                health.get("storage", {}).get("detail", ""),
                _gb(metrics.get("hdd_capacity_mb")),
                _gb(metrics.get("hdd_free_mb")),
                _val(metrics.get("temperature_c")),
                _val(metrics.get("cpu_percent")),
                summary["source"],
                device.last_error,
                device.notes,
            ],
        )
        fill = LEVEL_FILLS.get(health.get("overall", ""))
        if fill:
            ws_nvr.cell(row=ws_nvr.max_row, column=health_col).fill = fill

        # Cameras sheet — group title row per NVR
        ws_cam.append(
            [
                f"{device.name}  ({device.ip_address or 'no IP'})   —   "
                f"{summary['connected']} camera(s) connected · {summary['online']} online · "
                f"{summary['offline']} offline · {summary['video_loss']} video loss"
            ]
        )
        cam_skip_rows.add(ws_cam.max_row)
        for cell in ws_cam[ws_cam.max_row]:
            cell.fill = GROUP_FILL
            cell.font = GROUP_FONT
        ws_cam.merge_cells(
            start_row=ws_cam.max_row,
            start_column=1,
            end_row=ws_cam.max_row,
            end_column=min(len(CAMERA_HEADERS), 20),
        )
        if not cam_rows:
            _append_row(ws_cam, [site_name, device.name, device.ip_address or "", "", "No cameras found"])
        for row in cam_rows:
            _append_row(ws_cam, row)
            fill = STATUS_FILLS.get(row[cam_status_col - 1])
            if fill:
                ws_cam.cell(row=ws_cam.max_row, column=cam_status_col).fill = fill

        for hdd in hdds:
            _append_row(
                ws_hdd,
                [
                    site_name,
                    device.name,
                    hdd.get("name", ""),
                    hdd.get("status", ""),
                    _val(hdd.get("capacity_gb")),
                    hdd.get("remaining_gb") if hdd.get("remaining_gb") is not None else "",
                    hdd.get("usage_display", ""),
                    hdd.get("disk_type", ""),
                    hdd.get("attribute", ""),
                    hdd.get("disk_model", ""),
                    hdd.get("disk_serial", ""),
                ],
            )
            fill = LEVEL_FILLS.get(hdd.get("level", ""))
            if fill:
                ws_hdd.cell(row=ws_hdd.max_row, column=4).fill = fill

    if ws_hdd.max_row == 1:
        _append_row(ws_hdd, ["No storage data reported"])

    _autosize(ws_nvr)
    _autosize(ws_cam, max_width=40, skip_rows=cam_skip_rows)
    _autosize(ws_hdd)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
