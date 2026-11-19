import { useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { Search, Video } from "lucide-react"
import type { CityCamerasSection } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { Card, EmptyRow, LoadState, SectionHeader, StatTile, inputClass, useCitySection } from "./shared"

type SortKey = "name" | "activity" | "last"

const STATUS_STYLE: Record<string, string> = {
  online: "text-[#12A66A] bg-[#12A66A12] border-[#12A66A55]",
  offline: "text-[#FF4D59] bg-[#FF4D5912] border-[#FF4D5955]",
  inactive: "text-[#607DA5] bg-[#F4F8FD] border-[#D9E6F5]",
}

export function CamerasSection({ cityId, camerasPath }: { cityId: string; camerasPath: string }) {
  const { data, error, loading } = useCitySection<CityCamerasSection>(cityId, "cameras", {})
  const [query, setQuery] = useState("")
  const [zone, setZone] = useState("")
  const [status, setStatus] = useState("")
  const [sort, setSort] = useState<SortKey>("name")

  const cameras = useMemo(() => {
    if (!data) return []
    const q = query.trim().toLowerCase()
    const list = data.cameras.filter(
      (c) =>
        (!zone || c.zone === zone) &&
        (!status || c.status === status) &&
        (!q || [c.name, c.code, c.zone, c.location, c.site].some((v) => (v || "").toLowerCase().includes(q)))
    )
    if (sort === "activity") list.sort((a, b) => b.detections_24h - a.detections_24h)
    else if (sort === "last")
      list.sort((a, b) => (b.last_detection?.at || "").localeCompare(a.last_detection?.at || ""))
    else list.sort((a, b) => a.name.localeCompare(b.name))
    return list
  }, [data, query, zone, status, sort])

  if (!data) return <LoadState error={error} loading={loading} />
  const s = data.summary
  const perCamera = s.per_camera_status

  return (
    <div className="space-y-4">
      <SectionHeader
        title={`Cameras — ${data.city.name}`}
        subtitle={perCamera ? "Live status from the ML server" : "Online total reported by the city · per-camera: last detection"}
        loading={loading}
      >
        <Link
          to={camerasPath}
          className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-[#0875E1] px-3 text-[12px] font-semibold text-white hover:bg-[#086BC6]"
        >
          <Video className="h-3.5 w-3.5" />
          Open live video
        </Link>
      </SectionHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatTile label="Online" value={s.online} tone="green" note={`of ${s.total} cameras`} />
        <StatTile label="Offline" value={s.offline} tone={s.offline ? "red" : "gray"} note={s.offline ? "Need attention" : "All online"} />
        <StatTile label="Detected something (24h)" value={s.active_24h} tone="purple" note={`of ${s.listed} listed`} />
        <StatTile label="Zones" value={data.zones.length} tone="blue" note={data.zones.slice(0, 3).join(", ") || "No zones set"} />
      </div>

      <Card
        title={`Camera list (${cameras.length})`}
        right={
          <div className="flex flex-wrap items-center gap-2">
            <label className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#90A7C4]" />
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search camera..." className={cn(inputClass, "w-[180px] pl-7")} />
            </label>
            <select value={zone} onChange={(e) => setZone(e.target.value)} className={inputClass}>
              <option value="">All zones</option>
              {data.zones.map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
            {perCamera ? (
              <select value={status} onChange={(e) => setStatus(e.target.value)} className={inputClass}>
                <option value="">All status</option>
                <option value="online">Online</option>
                <option value="offline">Offline</option>
                <option value="inactive">Inactive</option>
              </select>
            ) : null}
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className={inputClass}>
              <option value="name">Sort: Name</option>
              <option value="activity">Sort: Most detections</option>
              <option value="last">Sort: Latest detection</option>
            </select>
          </div>
        }
      >
        <div className="max-h-[520px] overflow-auto">
          <table className="w-full text-left text-[12px]">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                <th className="py-2 pr-2 font-semibold">Camera</th>
                <th className="py-2 pr-2 font-semibold">Zone</th>
                <th className="py-2 pr-2 font-semibold">Location / Site</th>
                <th className="py-2 pr-2 font-semibold">Ch.</th>
                {perCamera ? <th className="py-2 pr-2 font-semibold">Status</th> : null}
                <th className="py-2 pr-2 text-right font-semibold">Detections 24h</th>
                <th className="py-2 font-semibold">Last detection</th>
              </tr>
            </thead>
            <tbody>
              {cameras.length === 0 ? (
                <EmptyRow cols={perCamera ? 7 : 6} text={data.cameras.length ? "No camera matches the filter." : "This city has not sent its camera list yet."} />
              ) : (
                cameras.map((c) => (
                  <tr key={c.key} className="border-b border-[#F2F7FC]">
                    <td className="py-2 pr-2">
                      <p className="font-semibold text-[#102C55]">{c.name}</p>
                      {c.code ? <p className="text-[11px] text-[#90A7C4]">{c.code}</p> : null}
                    </td>
                    <td className="py-2 pr-2 text-[#607DA5]">{c.zone || "—"}</td>
                    <td className="py-2 pr-2 text-[#607DA5]">{[c.location, c.site].filter(Boolean).join(" · ") || "—"}</td>
                    <td className="py-2 pr-2 tabular-nums text-[#607DA5]">{c.channel ?? "—"}</td>
                    {perCamera ? (
                      <td className="py-2 pr-2">
                        <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-semibold capitalize", STATUS_STYLE[c.status || "inactive"])}>
                          {c.status}
                        </span>
                      </td>
                    ) : null}
                    <td className="py-2 pr-2 text-right font-semibold tabular-nums text-[#102C55]">{c.detections_24h}</td>
                    <td className="py-2 text-[#607DA5]">
                      {c.last_detection ? (
                        <span title={c.last_detection.time}>{c.last_detection.ago}</span>
                      ) : (
                        <span className="text-[#A9BBD3]">Never</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
