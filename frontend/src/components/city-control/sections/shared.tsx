import { useEffect, useRef, useState, type Dispatch, type ReactNode } from "react"
import { RefreshCw } from "lucide-react"
import {
  CITY_RANGES,
  fetchCitySection,
  type CityRangeDays,
  type CitySectionId,
  type CitySectionParams,
} from "@/lib/city-section-api"
import { CITY_CONTROL_REFRESH_MS } from "@/data/city-control-grid-data"
import { cn } from "@/lib/utils"

/** Loads one menu screen and refreshes it every 30s; keeps the last good data on a failed refresh. */
export function useCitySection<T>(cityId: string, section: CitySectionId, params: CitySectionParams) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(true)
  const key = JSON.stringify(params)
  const dataRef = useRef<T | null>(null)
  dataRef.current = data

  useEffect(() => {
    const ac = new AbortController()
    setLoading(true)
    const load = () =>
      fetchCitySection(cityId, section, params, ac.signal)
        .then((d) => {
          if (ac.signal.aborted) return
          setData(d as T)
          setError("")
        })
        .catch((e: unknown) => {
          if (ac.signal.aborted) return
          if (!dataRef.current) setError(e instanceof Error ? e.message : "Could not load this screen")
        })
        .finally(() => {
          if (!ac.signal.aborted) setLoading(false)
        })
    void load()
    const timer = window.setInterval(() => void load(), CITY_CONTROL_REFRESH_MS)
    return () => {
      window.clearInterval(timer)
      ac.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cityId, section, key])

  return { data, error, loading }
}

export function RangePicker({ value, onChange }: { value: CityRangeDays; onChange: Dispatch<CityRangeDays> }) {
  return (
    <div className="inline-flex rounded-lg border border-[#D9E6F5] bg-white p-0.5">
      {CITY_RANGES.map((r) => (
        <button
          key={r.days}
          type="button"
          onClick={() => onChange(r.days)}
          className={cn(
            "rounded-md px-2.5 py-1 text-[12px] font-semibold transition",
            value === r.days ? "bg-[#0875E1] text-white" : "text-[#607DA5] hover:text-[#102C55]"
          )}
        >
          {r.label}
        </button>
      ))}
    </div>
  )
}

export function SectionHeader({
  title,
  subtitle,
  loading,
  children,
}: {
  title: string
  subtitle?: string
  loading?: boolean
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 className="flex items-center gap-2 text-[16px] font-bold text-[#102C55]">
          {title}
          {loading ? <RefreshCw className="h-3.5 w-3.5 animate-spin text-[#90A7C4]" /> : null}
        </h3>
        {subtitle ? <p className="text-[12px] text-[#607DA5]">{subtitle}</p> : null}
      </div>
      {children ? <div className="flex flex-wrap items-center gap-2">{children}</div> : null}
    </div>
  )
}

export function Card({ title, right, children, className }: { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-[10px] border border-[#D9E6F5] bg-white p-4", className)}>
      {title || right ? (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title ? <h4 className="text-[14px] font-semibold text-[#102C55]">{title}</h4> : <span />}
          {right}
        </div>
      ) : null}
      {children}
    </div>
  )
}

const TONES = {
  blue: ["#BFDBFE", "#2563EB"],
  purple: ["#DDD6FE", "#7C3AED"],
  red: ["#FECACA", "#EF4444"],
  orange: ["#FED7AA", "#F59E0B"],
  green: ["#A7F3D0", "#10B981"],
  gray: ["#D9E6F5", "#607DA5"],
} as const

export function StatTile({
  label,
  value,
  note,
  tone = "blue",
  delta,
}: {
  label: string
  value: ReactNode
  note?: ReactNode
  tone?: keyof typeof TONES
  delta?: string
}) {
  const [border, color] = TONES[tone]
  return (
    <div className="rounded-[10px] border bg-white p-3" style={{ borderColor: border }}>
      <p className="text-[12px] font-medium text-[#607DA5]">{label}</p>
      <div className="mt-1 flex items-end gap-2">
        <p className="text-[22px] font-bold" style={{ color: tone === "gray" ? "#102C55" : color }}>
          {value}
        </p>
        {delta ? <DeltaBadge delta={delta} /> : null}
      </div>
      {note ? <p className="mt-0.5 text-[11px] text-[#607DA5]">{note}</p> : null}
    </div>
  )
}

/** "+12%" green, "-8%" red, "new" blue, "—" grey. */
export function DeltaBadge({ delta }: { delta: string }) {
  const color = delta.startsWith("+") ? "#12A66A" : delta.startsWith("-") ? "#FF4D59" : delta === "new" ? "#0875E1" : "#90A7C4"
  const text = delta === "—" ? "no change data" : delta === "new" ? "new" : `${delta} vs before`
  return (
    <span className="mb-1 text-[11px] font-semibold" style={{ color }}>
      {text}
    </span>
  )
}

export function LoadState({ error, loading }: { error?: string; loading?: boolean }) {
  return (
    <div className="rounded-[10px] border border-[#D9E6F5] bg-white p-6 text-center text-[13px] text-[#607DA5]">
      {error ? <span className="text-[#FF4D59]">{error}</span> : loading ? "Loading…" : "No data"}
    </div>
  )
}

export function EmptyRow({ cols, text }: { cols: number; text: string }) {
  return (
    <tr>
      <td colSpan={cols} className="py-6 text-center text-[13px] text-[#607DA5]">
        {text}
      </td>
    </tr>
  )
}

export function Pager({ page, pages, onPage }: { page: number; pages: number; onPage: Dispatch<number> }) {
  if (pages <= 1) return null
  return (
    <div className="mt-3 flex items-center justify-end gap-2 text-[12px]">
      <button
        type="button"
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
        className="rounded-md border border-[#D9E6F5] px-2.5 py-1 font-semibold text-[#102C55] disabled:opacity-40"
      >
        Previous
      </button>
      <span className="text-[#607DA5]">
        Page {page} of {pages}
      </span>
      <button
        type="button"
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
        className="rounded-md border border-[#D9E6F5] px-2.5 py-1 font-semibold text-[#102C55] disabled:opacity-40"
      >
        Next
      </button>
    </div>
  )
}

export const inputClass =
  "h-8 rounded-lg border border-[#D9E6F5] bg-white px-2.5 text-[12px] text-[#102C55] outline-none placeholder:text-[#90A7C4] focus:border-[#0875E1] focus:ring-2 focus:ring-[#0875E1]/20"
