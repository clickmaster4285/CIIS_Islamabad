import { useState } from "react"
import { BarChart3, Download, FileSpreadsheet, FileText, Loader2 } from "lucide-react"
import { downloadCityExcel, downloadComparativePdf, downloadDailySnapshotPdf } from "@/lib/city-reports"
import type { CityRangeDays } from "@/lib/city-section-api"

type ReportId = "snapshot" | "excel" | "compare"

/** The three History & Reports downloads. `days` sets the Excel / comparative period. */
export function ReportButtons({ cityId, days = 30 }: { cityId: string; days?: CityRangeDays }) {
  const [busy, setBusy] = useState<ReportId | null>(null)
  const [error, setError] = useState("")
  const period = days === 1 ? "today" : `last ${days} days`

  const run = async (id: ReportId) => {
    if (busy) return
    setBusy(id)
    setError("")
    try {
      if (id === "snapshot") await downloadDailySnapshotPdf(cityId)
      else if (id === "excel") await downloadCityExcel(cityId, days)
      else await downloadComparativePdf(cityId, days)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not make the report")
    } finally {
      setBusy(null)
    }
  }

  const items: { id: ReportId; title: string; note: string; icon: typeof FileText }[] = [
    { id: "snapshot", title: "Daily Snapshot (PDF)", note: "Today's status and records", icon: FileText },
    { id: "excel", title: "Export Data (Excel)", note: `All screens, ${period}`, icon: FileSpreadsheet },
    { id: "compare", title: "Comparative Report", note: `All cities, ${period} (PDF)`, icon: BarChart3 },
  ]

  return (
    <div>
      <ul className="space-y-2">
        {items.map((item) => {
          const Icon = item.icon
          return (
            <li key={item.id}>
              <button
                type="button"
                disabled={Boolean(busy)}
                onClick={() => void run(item.id)}
                className="flex w-full items-center gap-2.5 rounded-lg border border-[#D9E6F5] px-3 py-2.5 text-left transition hover:border-[#0875E1]/40 hover:bg-[#F7FBFF] disabled:opacity-60"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#EFF6FF]">
                  <Icon className="h-4 w-4 text-[#0875E1]" />
                </div>
                <span className="flex-1">
                  <span className="block text-[12px] font-semibold text-[#102C55]">{item.title}</span>
                  <span className="block text-[11px] text-[#90A7C4]">{busy === item.id ? "Preparing…" : item.note}</span>
                </span>
                {busy === item.id ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-[#0875E1]" />
                ) : (
                  <Download className="h-3.5 w-3.5 text-[#607DA5]" />
                )}
              </button>
            </li>
          )
        })}
      </ul>
      {error ? <p className="mt-2 text-[12px] text-[#FF4D59]">{error}</p> : null}
    </div>
  )
}
