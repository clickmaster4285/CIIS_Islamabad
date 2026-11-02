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
}

export type PhoneActivityFilters = { date?: string; username?: string }

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

function monitorQuery(filters: PhoneActivityFilters): string {
  const sp = new URLSearchParams()
  if (filters.date) sp.set("date", filters.date)
  if (filters.username) sp.set("username", filters.username.trim())
  return sp.toString() ? `?${sp}` : ""
}

export async function fetchPhoneCalls(filters: PhoneActivityFilters = {}): Promise<PhoneCallRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/calls/${monitorQuery(filters)}`, {
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load call logs"))
  const data = await res.json()
  return Array.isArray(data?.results) ? data.results : []
}

export async function fetchAppUsage(filters: PhoneActivityFilters = {}): Promise<AppUsageRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/app-usage/${monitorQuery(filters)}`, {
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load app usage"))
  const data = await res.json()
  return Array.isArray(data?.results) ? data.results : []
}

export type CallRecordingRecord = {
  id: string
  call_id: string | null
  source: string
  source_label: string
  recorded_at: string | null
  duration_seconds: number | null
  status: string
  can_play: boolean
}

export async function fetchRecordings(filters: PhoneActivityFilters = {}): Promise<CallRecordingRecord[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/employee-monitor/recordings/${monitorQuery(filters)}`, {
    headers: getAuthHeaders(),
  })
  if (!res.ok) throw new Error(await readMonitorError(res, "Failed to load recordings"))
  const data = await res.json()
  return Array.isArray(data?.results) ? data.results : []
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
