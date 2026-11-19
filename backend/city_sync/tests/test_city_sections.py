"""City Control Grid left-menu screens (ops_central/city_sections.py)."""

from __future__ import annotations

from datetime import timedelta
from unittest import mock

from django.test import TestCase, override_settings
from django.utils import timezone

from city_sync.models import CityCamera, HqCityStatus, SyncSite
from city_sync.tests.test_city_board import HQ_CAMS, city_event
from city_sync.tests.test_sites import HQ
from detentions.models import DetentionMemo, DetentionMemoGoodsLine
from ops_central.city_sections import UnknownCity, build_city_section
from seizure_management.models import NoteSheet, NoteSheetItem
from visitors.models import SecurityAlert
from warehouse.models import WarehouseStockItem


@override_settings(**HQ)
@mock.patch("cameras.live_status.camera_counts", return_value=HQ_CAMS)
class CitySectionTests(TestCase):
    def setUp(self):
        SyncSite.objects.create(location_code="HAYATABAD", name="Hayatabad", base_url="http://x")
        HqCityStatus.objects.create(
            location_code="HAYATABAD",
            last_heartbeat_at=timezone.now(),
            raw={
                "cameras_total": 2,
                "cameras_online": 1,
                "vitals": {
                    "uptime_sec": 7200,
                    "disk_percent": 41.0,
                    "ml_source": "ml",
                    "infra": {
                        "total": 3,
                        "online": 2,
                        "issues": 1,
                        "nvrs_total": 1,
                        "nvrs_online": 1,
                        "by_type": {"nvr": {"total": 1, "online": 1}, "ups": {"total": 2, "online": 1}},
                    },
                },
            },
        )
        CityCamera.objects.create(location_code="HAYATABAD", origin_pk=7, name="Gate 1", zone="Entry")
        CityCamera.objects.create(location_code="HAYATABAD", origin_pk=8, name="Yard", zone="Store")

    def section(self, name, **params):
        return build_city_section("hayatabad", name, params)

    def test_unknown_city_and_section(self, _cams):
        with self.assertRaises(UnknownCity):
            build_city_section("nowhere", "ai", {})
        with self.assertRaises(ValueError):
            build_city_section("hayatabad", "secrets", {})

    def test_cameras_list_with_last_detection(self, _cams):
        city_event("HAYATABAD", "person", cam=7)
        data = self.section("cameras")
        self.assertEqual((data["summary"]["online"], data["summary"]["total"]), (1, 2))
        self.assertFalse(data["summary"]["per_camera_status"])
        cams = {c["name"]: c for c in data["cameras"]}
        self.assertEqual(cams["Gate 1"]["detections_24h"], 1)
        self.assertIsNone(cams["Yard"]["last_detection"])
        self.assertEqual([c["name"] for c in self.section("cameras", q="yard")["cameras"]], ["Yard"])

    def test_ai_counts_filters_and_pages(self, _cams):
        city_event("HAYATABAD", "person", cam=7)
        city_event("HAYATABAD", "knife", cam=8, alert=True)
        city_event("HAYATABAD", "car", cam=7, minutes_ago=60 * 24 * 3)  # 3 days ago
        city_event("PESHAWAR", "person")  # another location
        week = self.section("ai", days="7")
        self.assertEqual(week["summary"]["total"], 3)
        self.assertEqual(week["summary"]["by_type"]["Weapon"], 1)
        self.assertEqual(week["summary"]["alerts"], 1)
        self.assertEqual(week["top_cameras"][0], {"key": 7, "camera": "Gate 1", "count": 2})
        self.assertEqual(self.section("ai", days="1")["summary"]["total"], 2)
        weapons = self.section("ai", days="7", type="Weapon")
        self.assertEqual([i["camera"] for i in weapons["items"]], ["Yard"])
        self.assertEqual(self.section("ai", days="7", camera="7")["matched"], 2)

    def test_seizure_lists_have_case_details_only(self, _cams):
        sheet = NoteSheet.objects.create(
            location_code="HAYATABAD", note_sheet_no="NS-1", case_no="C-9", prepared_by="Officer A",
            accused_name="Secret Person", accused_cnic="12345-1234567-1",
        )
        NoteSheetItem.objects.create(note_sheet=sheet, product="Cloth", estimated_value="1,500", location_code="HAYATABAD")
        memo = DetentionMemo.objects.create(
            location_code="HAYATABAD", case_no="D-3", owner_name="Owner X", owner_cnic="11111-1111111-1",
            driver_name="Driver Y",
        )
        DetentionMemoGoodsLine.objects.create(memo=memo, description="Tyres", assessable_value_pkr="2000", location_code="HAYATABAD")
        data = self.section("seizures", days="30")
        self.assertEqual((data["summary"]["seizures"], data["summary"]["detentions"]), (1, 1))
        self.assertEqual(data["note_sheets"][0]["value_pkr"], 1500.0)
        self.assertEqual((data["summary"]["seizure_value_pkr"], data["summary"]["detention_value_pkr"]), (1500.0, 2000.0))
        self.assertEqual(data["detention_memos"][0]["goods"], "Tyres")
        text = str(data)
        for secret in ("Secret Person", "12345-1234567-1", "Owner X", "11111-1111111-1", "Driver Y"):
            self.assertNotIn(secret, text)
        self.assertEqual(len(self.section("seizures", days="30", q="zzz")["note_sheets"]), 0)

    def test_customs_operations(self, _cams):
        WarehouseStockItem.objects.create(location_code="HAYATABAD", case_ref="C-9", description="Cloth", quantity=5)
        SecurityAlert.objects.create(location_code="HAYATABAD", alert_type="watchlist_hit", severity="high", message="Ali at gate")
        data = self.section("customs", days="7")
        self.assertEqual(data["warehouse"]["items"], 1)
        self.assertEqual(data["security_alerts"]["open"], 1)
        self.assertNotIn("Ali at gate", str(data))  # alert text can name a visitor
        self.assertIn("gate_today", data)

    def test_infrastructure_from_heartbeat_vitals(self, _cams):
        data = self.section("infra")
        self.assertEqual(data["health"]["uptime"], "2h 0m")
        self.assertEqual(data["health"]["infra_health"], "Degraded")
        self.assertEqual({t["type"] for t in data["by_type"]}, {"NVRs", "UPS"})
        self.assertFalse(data["devices_listed"])

    def test_charts_have_daily_series_and_comparison(self, _cams):
        city_event("HAYATABAD", "person")
        data = self.section("charts", days="30")
        self.assertEqual(len(data["trend"]), 30)
        self.assertEqual(data["trend"][-1]["people"], 1)
        cities = {c["id"]: c for c in data["comparison"]}
        self.assertEqual(set(cities), {"peshawar", "hayatabad"})
        self.assertEqual(cities["hayatabad"]["detections"], 1)

    def test_offline_city_infra_is_critical(self, _cams):
        HqCityStatus.objects.filter(location_code="HAYATABAD").update(
            last_heartbeat_at=timezone.now() - timedelta(hours=3)
        )
        data = self.section("infra")
        self.assertEqual(data["health"]["api_connection"], "Disconnected")
        self.assertEqual(data["health"]["infra_health"], "Critical")
