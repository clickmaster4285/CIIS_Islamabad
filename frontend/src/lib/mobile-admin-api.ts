import { API_BASE_URL, getAuthHeaders } from "@/lib/api"

export type MobileDeviceRecord = {
  id: number
  user_id: number
  staff_name: string
  username: string
  device_uuid: string
  device_name: string
  platform: string
  browser: string
  os_version: string
  app_version: string
  pwa_version: string
  is_active: boolean
  is_revoked: boolean
  mobile_access_enabled: boolean
  status: string
  registered_at: string | null
  last_seen_at: string | null
  last_gps_at: string | null
  last_heartbeat_at: string | null
  last_latitude: number | null
  last_longitude: number | null
  battery_level: number | null
  last_gps_status: string
  session_status: string | null
  session_id: string | null
  logged_in: boolean
  installed: boolean
}

export type MobileAlertRecord = {
  id: number
  alert_type: string
  severity: "INFO" | "WARNING" | "CRITICAL" | string
  message: string
  staff_name: string
  user_id: number
  staff_id: number | null
  device_id: number | null
  device_uuid: string | null
  device_name: string | null
  created_at: string
  detected_at: string
  is_active: boolean
  last_gps_at: string | null
  last_heartbeat_at: string | null
  last_latitude: number | null
  last_longitude: number | null
  battery_level: number | null
  device_status: string | null
  session_status: string | null
}

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.detail === "string") return data.detail
  } catch {
    /* ignore */
  }
  return fallback
}

export async function fetchMobileDevices(params?: { userId?: number; username?: string }): Promise<MobileDeviceRecord[]> {
  const sp = new URLSearchParams()
  if (params?.userId) sp.set("user_id", String(params.userId))
  if (params?.username) sp.set("username", params.username)
  const q = sp.toString() ? `?${sp}` : ""
  const res = await fetch(`${API_BASE_URL}/api/mobile-devices/${q}`, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(await readError(res, "Failed to load devices"))
  const data = await res.json()
  return Array.isArray(data?.results) ? data.results : []
}

export async function fetchMobileDevice(id: number): Promise<MobileDeviceRecord> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-devices/${id}/`, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(await readError(res, "Failed to load device"))
  return res.json()
}

export async function fetchMobileAlerts(): Promise<MobileAlertRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-alerts/?active=1`, { headers: getAuthHeaders() })
  if (!res.ok) throw new Error(await readError(res, "Failed to load alerts"))
  const data = await res.json()
  return Array.isArray(data?.results) ? data.results : []
}

export async function acknowledgeMobileAlert(id: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-alerts/${id}/acknowledge/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: "{}",
  })
  if (!res.ok) throw new Error(await readError(res, "Failed to acknowledge alert"))
}

export async function revokeMobileDevice(id: number, reason = ""): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-devices/${id}/revoke/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({ reason }),
  })
  if (!res.ok) throw new Error(await readError(res, "Failed to revoke device"))
}

export async function terminateMobileDeviceSession(id: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-devices/${id}/terminate-session/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: "{}",
  })
  if (!res.ok) throw new Error(await readError(res, "Failed to terminate session"))
}

export async function disableMobileAccess(id: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/mobile-devices/${id}/disable-access/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: "{}",
  })
  if (!res.ok) throw new Error(await readError(res, "Failed to disable mobile access"))
}

export type PhoneCallRecord = {
  id: string
  username: string | null
  name: string | null
  direction: string
  masked_number: string | null
  /** Full number; only sent to overview roles (each view is audited server-side). */
  number?: string | null
  started_at: string | null
  ended_at: string | null
  duration_seconds: number | null
  source: string
  source_label: string
  sim_slot: string | null
  recording_status: string
  recording_id: string | null
}

export type AppUsageRecord = {
  id: string
  username: string | null
  name: string | null
  package_name: string
  app_label: string
  foreground_seconds: number
  launch_count: number
  window_start: string | null
  window_end: string | null
  last_used_at: string | null
  /** data:image/png;base64 URI for the app icon, or "" if the phone hasn't uploaded it yet. */
  icon?: string
}

export type PhoneActivityFilters = { date?: string; username?: string }

/** One page of results. `count` is the total across all pages (server-side). */
export type Paged<T> = {
  results: T[]
  count: number
  page: number
  page_size: number
  has_next: boolean
  has_prev: boolean
}

/** Page request: 1-based page and a page size the server clamps (<=100). `signal` lets React
 *  Query abort an in-flight request when filters/page change, so stale requests never land. */
export type PageParams = { page?: number; pageSize?: number; signal?: AbortSignal }

const DEFAULT_PAGE_SIZE = 50

async function readMonitorError(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.error?.message === "string") return data.error.message
    if (typeof data?.detail === "string") return data.detail
  } catch {
    /* ignore */
  }
  return fallback
}

function monitorQuery(filters: PhoneActivityFilters, params: PageParams = {}): string {
  const sp = new URLSearchParams()
  if (filters.date) sp.set("date", filters.date)
  if (filters.username) sp.set("username", filters.username.trim())
  sp.set("page", String(Math.max(1, params.page ?? 1)))
  sp.set("page_size", String(params.pageSize ?? DEFAULT_PAGE_SIZE))
  return `?${sp}`
}

type PageMetaSource = {
  count?: unknown
  page?: unknown
  page_size?: unknown
  has_next?: unknown
  has_prev?: unknown
} | null | undefined

function pageMeta<T>(data: PageMetaSource, results: T[], params: PageParams): Paged<T> {
  return {
    results,
    count: typeof data?.count === "number" ? data.count : results.length,
    page: typeof data?.page === "number" ? data.page : params.page ?? 1,
    page_size: typeof data?.page_size === "number" ? data.page_size : params.pageSize ?? DEFAULT_PAGE_SIZE,
    has_next: !!data?.has_next,
    has_prev: !!data?.has_prev,
  }
}

export async function fetchPhoneCalls(
  filters: PhoneActivityFilters = {},
  params: PageParams = {},
): Promise<Paged<PhoneCallRecord> & { total_duration_seconds?: number }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/calls/${monitorQuery(filters, params)}`, {
    headers: getAuthHeaders(),
    signal: params.signal,
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load call logs"))
  const data = await res.json()
  const results = Array.isArray(data?.results) ? data.results : []
  return { ...pageMeta(data, results, params), total_duration_seconds: data?.total_duration_seconds }
}

export async function fetchAppUsage(
  filters: PhoneActivityFilters = {},
  params: PageParams = {},
): Promise<Paged<AppUsageRecord> & { total_foreground_seconds?: number }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/app-usage/${monitorQuery(filters, params)}`, {
    headers: getAuthHeaders(),
    signal: params.signal,
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load app usage"))
  const data = await res.json()
  const results = Array.isArray(data?.results) ? data.results : []
  return { ...pageMeta(data, results, params), total_foreground_seconds: data?.total_foreground_seconds }
}

export type CallRecordingRecord = {
  id: string
  call_id: string | null
  /** Whose call this recording is. */
  username: string | null
  name: string | null
  /** Linked call details (null when the recording isn't tied to a call row). */
  direction: string | null
  masked_number: string | null
  number?: string | null
  call_started_at: string | null
  source: string
  source_label: string
  recorded_at: string | null
  duration_seconds: number | null
  status: string
  can_play: boolean
}

export async function fetchRecordings(
  filters: PhoneActivityFilters = {},
  params: PageParams = {},
): Promise<Paged<CallRecordingRecord>> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/recordings/${monitorQuery(filters, params)}`, {
    headers: getAuthHeaders(),
    signal: params.signal,
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load recordings"))
  const data = await res.json()
  const results = Array.isArray(data?.results) ? data.results : []
  return pageMeta(data, results, params)
}

/**
 * Resolve a recording to a playable blob URL. Two audited server hops: first an
 * authorized short-lived signed stream URL, then the stream itself (fetched with the
 * auth header, since a bare <audio src> cannot carry the bearer token). The caller
 * owns the returned object URL and must revoke it when done.
 */
export async function fetchRecordingAudioUrl(id: string): Promise<string> {
  const auth = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/recordings/${id}/playback/`, {
    headers: getAuthHeaders(),
  })
  if (!auth.ok) throw new Error(await readMonitorError(auth, "Could not authorize playback"))
  const { url } = await auth.json()
  if (typeof url !== "string" || !url) throw new Error("Playback URL missing from server response")
  const stream = await fetch(url, { headers: getAuthHeaders() })
  if (!stream.ok) throw new Error(await readMonitorError(stream, "Could not stream recording"))
  return URL.createObjectURL(await stream.blob())
}
