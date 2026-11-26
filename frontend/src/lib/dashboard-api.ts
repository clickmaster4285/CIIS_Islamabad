/**
 * Dashboard "At a Glance" stats: warehouse, AI analytics and HR numbers.
 */
import { API_BASE_URL, getAuthHeaders } from "@/lib/api"

export type DashboardGlance = {
  warehouse: { active_operations: number; pending_items: number }
  analytics: {
    /** Busiest 4-hour window of visitor gate entries, e.g. "10AM - 2PM"; null when no scans */
    peak_hours: string | null
    /** Visitors who checked out ÷ visitors who checked in, 0–100; null when no check-ins */
    efficiency_pct: number | null
    window_days: number
  }
  hr: { employees: number; present_today: number }
  generated_at: string
}

export async function fetchDashboardGlance(): Promise<DashboardGlance> {
  const res = await fetch(`${API_BASE_URL}/api/dashboard/glance/`, {
    headers: getAuthHeaders(),
    cache: "no-store",
  })
  if (!res.ok) throw new Error(`Failed to load dashboard stats (${res.status})`)
  return res.json()
}
