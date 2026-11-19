import { API_BASE_URL, getAuthHeaders } from "@/lib/api"

const API = `${API_BASE_URL}/api`

/** Left-menu screens of /city-control-grid/<city> (backend: ops_central/city_sections.py). */
export type CitySectionId = "cameras" | "ai" | "seizures" | "customs" | "infra" | "charts"
export type CityRangeDays = 1 | 7 | 30 | 90

export const CITY_RANGES: { days: CityRangeDays; label: string }[] = [
  { days: 1, label: "Today" },
  { days: 7, label: "7 days" },
  { days: 30, label: "30 days" },
  { days: 90, label: "90 days" },
]

export type When = { at: string; time: string; ago: string }

type SectionBase = {
  city: { id: string; location_code: string; name: string; is_hq: boolean }
  section: CitySectionId
  days: CityRangeDays
  range_label: string
  generated_at: string
}

export type CityCameraItem = {
  key: number
  code: string
  name: string
  zone: string
  location: string
  channel: number | null
  site: string
  /** "online" | "offline" | "inactive" for HQ; null for a city (only totals are sent). */
  status: "online" | "offline" | "inactive" | null
  detections_24h: number
  last_detection: When | null
}

export type CityCamerasSection = SectionBase & {
  summary: {
    online: number
    offline: number
    total: number
    listed: number
    active_24h: number
    per_camera_status: boolean
  }
  zones: string[]
  cameras: CityCameraItem[]
}

export type CityDetectionGroup = "Person" | "Vehicle" | "Weapon" | "License Plate"

export type CityAiItem = {
  id: number
  time: string
  date: string
  at: string
  type: CityDetectionGroup | "Object"
  object: string
  camera: string
  image: string
  is_alert: boolean
  confidence: number
}

export type CityAiSection = SectionBase & {
  summary: {
    total: number
    delta: string
    alerts: number
    by_type: Record<CityDetectionGroup, number>
  }
  by_hour: { hour: string; count: number }[]
  top_cameras: { key: number; camera: string; count: number }[]
  items: CityAiItem[]
  page: number
  pages: number
  matched: number
}

export type CitySectionParams = {
  days?: CityRangeDays
  page?: number
  type?: string
  camera?: string | number
  alerts?: boolean
  q?: string
}

export async function fetchCitySection<T extends SectionBase>(
  cityId: string,
  section: CitySectionId,
  params: CitySectionParams = {},
  signal?: AbortSignal
): Promise<T> {
  const qs = new URLSearchParams()
  if (params.days) qs.set("days", String(params.days))
  if (params.page && params.page > 1) qs.set("page", String(params.page))
  if (params.type) qs.set("type", params.type)
  if (params.camera !== undefined && params.camera !== "") qs.set("camera", String(params.camera))
  if (params.alerts) qs.set("alerts", "1")
  if (params.q?.trim()) qs.set("q", params.q.trim())
  const query = qs.toString()
  const res = await fetch(
    `${API}/ops/city-control/${encodeURIComponent(cityId)}/${section}/${query ? `?${query}` : ""}`,
    { headers: getAuthHeaders(), signal }
  )
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error((data && data.detail) || "Could not load this screen")
  return data as T
}

/** Seizures & Detentions — case details only (no accused / owner / driver personal data). */
export type CityNoteSheetRow = When & {
  id: string
  note_sheet_no: string
  case_no: string
  reference_number: string
  subject: string
  office: string
  officer: string
  designation: string
  status: string
  priority: string
  items: number
  value_pkr: number
  goods: string
}

export type CityDetentionMemoRow = When & {
  id: string
  case_no: string
  reference_number: string
  fir_number: string
  detention_type: string
  place_of_detention: string
  officer: string
  status: string
  items: number
  value_pkr: number
  goods: string
}

export type CityStatusCount = { status: string; count: number }

export type CitySeizuresSection = SectionBase & {
  summary: {
    seizures: number
    seizure_delta: string
    detentions: number
    detention_delta: string
    seizure_value_pkr: number
    detention_value_pkr: number
    seizure_status: CityStatusCount[]
    detention_status: CityStatusCount[]
  }
  daily: { day: string; seizures: number; detentions: number }[]
  note_sheets: CityNoteSheetRow[]
  detention_memos: CityDetentionMemoRow[]
  list_limit: number
}

export type CityCustomsSection = SectionBase & {
  warehouse: {
    items: number
    by_status: CityStatusCount[]
    recent: (When & {
      case_ref: string
      qr_code: string
      description: string
      quantity: number
      unit: string
      godown: string
      status: string
    })[]
  }
  distributions: {
    total: number
    destructed: number
    inventory: number
    recent: (When & { case_no: string; outcome: string; status: string; performed_by: string; smoke_fire: boolean })[]
  }
  destruction_alerts: {
    open: number
    in_range: number
    recent: (When & { case_no: string; type: string; severity: string; message: string; acknowledged: boolean })[]
  }
  gate_today: {
    visitors: number
    on_premises: number
    exited: number
    vehicles: number
    scans_allowed: number
    scans_denied: number
    by_stage: { stage: string; count: number }[]
  }
  security_alerts: {
    open: number
    in_range: number
    recent: (When & { type: string; severity: string; zone: string; gate: string; acknowledged: boolean })[]
  }
  daily_visitors: { day: string; visitors: number; vehicles: number }[]
}

export type CityHealth = {
  api_connection: string
  api_detail: string
  ai_service: string
  ai_detail: string
  uptime: string | null
  disk_percent: number | null
  sync_queue: { pending_records: number; pending_media: number; failed: number; oldest_pending_sec: number } | null
  infra: { total: number; online: number; issues: number; nvrs_total: number; nvrs_online: number } | null
  infra_health: "Good" | "Degraded" | "Critical" | null
  last_detection: string | null
  vitals_reported: boolean
}

export type CityInfraSection = SectionBase & {
  health: CityHealth
  state: string
  heartbeat: When | null
  by_type: { type: string; total: number; online: number }[]
  devices: { name: string; type: string; status: string; location: string; site: string; last_seen: When | null }[]
  devices_listed: boolean
}

export type CityComparisonRow = {
  id: string
  city: string
  is_hq: boolean
  status: string
  cameras_online: number
  cameras_total: number
  detections: number
  seizures: number
  detentions: number
  staff_total: number
}

export type CityChartsSection = SectionBase & {
  trend: { day: string; people: number; vehicles: number; weapons: number; plates: number }[]
  ops: { day: string; seizures: number; detentions: number; visitors: number }[]
  comparison: CityComparisonRow[]
}
