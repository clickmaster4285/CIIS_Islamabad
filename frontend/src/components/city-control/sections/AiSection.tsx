import { useState } from "react"
import { AlertTriangle, Car, UserRound } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { DetectionThumb } from "@/components/city-control/detection-thumb"
import type { CityAiSection, CityDetectionGroup, CityRangeDays } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { Card, EmptyRow, LoadState, Pager, RangePicker, SectionHeader, StatTile, inputClass, useCitySection } from "./shared"

const TYPES: { id: CityDetectionGroup; label: string; tone: "purple" | "blue" | "red" | "green" }[] = [
  { id: "Person", label: "People", tone: "purple" },
  { id: "Vehicle", label: "Vehicles", tone: "blue" },
  { id: "Weapon", label: "Weapons", tone: "red" },
  { id: "License Plate", label: "Plates", tone: "green" },
]

function TypeIcon({ type }: { type: string }) {
  if (type === "Person") return <UserRound className="h-3.5 w-3.5 text-[#7552E8]" />
  if (type === "Weapon") return <AlertTriangle className="h-3.5 w-3.5 text-[#FF4D59]" />
  if (type === "License Plate") return <Car className="h-3.5 w-3.5 text-[#0875E1]" />
  return <Car className="h-3.5 w-3.5 text-[#12A66A]" />
}

export function AiSection({ cityId }: { cityId: string }) {
  const [filters, setFilters] = useState({ days: 1 as CityRangeDays, type: "", camera: "", alerts: false })
  const [page, setPage] = useState(1)
  const { days, type, camera, alerts } = filters
  const { data, error, loading } = useCitySection<CityAiSection>(cityId, "ai", { ...filters, page })

  // A new filter starts from page 1.
  const setFilter = (patch: Partial<typeof filters>) => {
    setFilters((f) => ({ ...f, ...patch }))
    setPage(1)
  }
  const setDays = (value: CityRangeDays) => setFilter({ days: value })
  const setType = (value: string) => setFilter({ type: value })
  const setCamera = (value: string) => setFilter({ camera: value })
  const setAlerts = (value: boolean) => setFilter({ alerts: value })

  if (!data) return <LoadState error={error} loading={loading} />
  const s = data.summary
  const filtered = Boolean(type || camera || alerts)

  return (
    <div className="space-y-4">
      <SectionHeader title={`AI Detections — ${data.city.name}`} subtitle={data.range_label} loading={loading}>
        <RangePicker value={days} onChange={setDays} />
      </SectionHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-6">
        <StatTile label="All detections" value={s.total.toLocaleString()} tone="gray" delta={s.delta} />
        {TYPES.map((t) => (
          <button key={t.id} type="button" onClick={() => setType(type === t.id ? "" : t.id)} className="text-left">
            <div className={cn("rounded-[10px]", type === t.id && "ring-2 ring-[#0875E1]")}>
              <StatTile label={t.label} value={(s.by_type[t.id] || 0).toLocaleString()} tone={t.tone} note={type === t.id ? "Filter on — click to clear" : "Click to filter"} />
            </div>
          </button>
        ))}
        <button type="button" onClick={() => setAlerts(!alerts)} className="text-left">
          <div className={cn("rounded-[10px]", alerts && "ring-2 ring-[#0875E1]")}>
            <StatTile label="Alerts" value={s.alerts.toLocaleString()} tone="orange" note={alerts ? "Alerts only — click to clear" : "Click to filter"} />
          </div>
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Detections by hour" className="xl:col-span-2">
          <div className="h-[220px]">
            {s.total === 0 ? (
              <p className="pt-20 text-center text-[13px] text-[#607DA5]">No detections in this period.</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.by_hour}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#607DA5" }} interval={1} />
                  <YAxis tick={{ fontSize: 11, fill: "#607DA5" }} allowDecimals={false} />
                  <Tooltip labelFormatter={(h) => `${h}:00 – ${h}:59`} />
                  <Bar dataKey="count" name="Detections" fill="#7552E8" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
        <Card title="Busiest cameras">
          {data.top_cameras.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-[#607DA5]">No camera data.</p>
          ) : (
            <ul className="space-y-2">
              {data.top_cameras.map((c) => {
                const max = data.top_cameras[0].count || 1
                const active = camera === String(c.key)
                return (
                  <li key={c.key}>
                    <button type="button" onClick={() => setCamera(active ? "" : String(c.key))} className={cn("w-full rounded-lg px-2 py-1.5 text-left hover:bg-[#F7FBFF]", active && "bg-[#EFF6FF]")}>
                      <div className="flex justify-between text-[12px]">
                        <span className="truncate font-semibold text-[#102C55]">{c.camera}</span>
                        <span className="tabular-nums text-[#607DA5]">{c.count.toLocaleString()}</span>
                      </div>
                      <div className="mt-1 h-1.5 rounded-full bg-[#EEF4FB]">
                        <div className="h-1.5 rounded-full bg-[#0875E1]" style={{ width: `${(100 * c.count) / max}%` }} />
                      </div>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </Card>
      </div>

      <Card
        title={`Detections (${data.matched.toLocaleString()})`}
        right={
          <div className="flex flex-wrap items-center gap-2">
            <select value={type} onChange={(e) => setType(e.target.value)} className={inputClass}>
              <option value="">All types</option>
              {TYPES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <select value={camera} onChange={(e) => setCamera(e.target.value)} className={inputClass}>
              <option value="">All cameras</option>
              {data.top_cameras.map((c) => (
                <option key={c.key} value={String(c.key)}>
                  {c.camera}
                </option>
              ))}
            </select>
            {filtered ? (
              <button
                type="button"
                onClick={() => setFilter({ type: "", camera: "", alerts: false })}
                className="text-[12px] font-semibold text-[#0875E1]"
              >
                Clear filters
              </button>
            ) : null}
          </div>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                <th className="py-2 pr-2 font-semibold">Image</th>
                <th className="py-2 pr-2 font-semibold">Date / Time</th>
                <th className="py-2 pr-2 font-semibold">Type</th>
                <th className="py-2 pr-2 font-semibold">Object</th>
                <th className="py-2 pr-2 font-semibold">Camera</th>
                <th className="py-2 pr-2 text-right font-semibold">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {data.items.length === 0 ? (
                <EmptyRow cols={6} text={filtered ? "Nothing matches the filter." : "No detections in this period."} />
              ) : (
                data.items.map((row) => (
                  <tr key={row.id} className={cn("border-b border-[#F2F7FC]", row.is_alert && "bg-[#FFF5F5]")}>
                    <td className="py-1.5 pr-2">
                      <DetectionThumb src={row.image} alt={row.object} />
                    </td>
                    <td className="py-2 pr-2 tabular-nums text-[#607DA5]">{row.date}</td>
                    <td className="py-2 pr-2">
                      <span className="inline-flex items-center gap-1.5 font-medium text-[#102C55]">
                        <TypeIcon type={row.type} />
                        {row.type}
                        {row.is_alert ? <span className="rounded bg-[#FF4D59] px-1 text-[10px] font-bold text-white">ALERT</span> : null}
                      </span>
                    </td>
                    <td className="py-2 pr-2 font-medium text-[#102C55]">{row.object}</td>
                    <td className="py-2 pr-2 text-[#607DA5]">{row.camera}</td>
                    <td className="py-2 pr-2 text-right tabular-nums text-[#607DA5]">{row.confidence}%</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        <Pager page={data.page} pages={data.pages} onPage={setPage} />
      </Card>
    </div>
  )
}
