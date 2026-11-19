import { useMemo, useState } from "react"
import type { CityInfraSection } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { Card, EmptyRow, LoadState, SectionHeader, StatTile, inputClass, useCitySection } from "./shared"

const GOOD = ["Connected", "Healthy", "Good", "online"]
const WARN = ["Intermittent", "Degraded", "degraded", "maintenance", "Unknown", "unknown"]

function tone(value: string | null | undefined) {
  if (!value) return "text-[#90A7C4]"
  if (GOOD.includes(value)) return "text-[#12A66A]"
  if (WARN.includes(value)) return "text-[#E07F0F]"
  return "text-[#FF4D59]"
}

function formatAge(sec: number) {
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.round(sec / 60)} min`
  return `${Math.round(sec / 3600)} hr`
}

export function InfraSection({ cityId }: { cityId: string }) {
  const { data, error, loading } = useCitySection<CityInfraSection>(cityId, "infra", {})
  const [status, setStatus] = useState("")
  const [type, setType] = useState("")

  const devices = useMemo(
    () => (data?.devices || []).filter((d) => (!status || d.status === status) && (!type || d.type === type)),
    [data, status, type]
  )

  if (!data) return <LoadState error={error} loading={loading} />
  const h = data.health
  const infra = h.infra
  const disk = h.disk_percent
  const types = Array.from(new Set(data.devices.map((d) => d.type)))

  return (
    <div className="space-y-4">
      <SectionHeader
        title={`Infrastructure — ${data.city.name}`}
        subtitle={
          data.city.is_hq
            ? "This server (HQ)"
            : data.heartbeat
              ? `From the city's last heartbeat · ${data.heartbeat.ago}`
              : "No heartbeat received from this city yet"
        }
        loading={loading}
      />

      {!h.vitals_reported && !data.city.is_hq ? (
        <div className="rounded-[10px] border border-[#FED7AA] bg-[#FFF7ED] px-4 py-2.5 text-[12px] text-[#9A5B13]">
          This city server has not reported uptime, disk or device numbers yet. It needs the latest update to send them.
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-5">
        <StatTile
          label="Infrastructure health"
          value={<span className={tone(h.infra_health)}>{h.infra_health || "Not reported"}</span>}
          tone="gray"
          note={infra ? `${infra.online}/${infra.total} devices online · ${infra.issues} issue${infra.issues === 1 ? "" : "s"}` : "No device numbers"}
        />
        <StatTile label="NVRs" value={infra ? `${infra.nvrs_online}/${infra.nvrs_total}` : "—"} tone="blue" note="Online / total" />
        <StatTile label="Server uptime" value={h.uptime || "—"} tone="green" note={h.uptime ? "Since last restart" : "Not reported"} />
        <StatTile
          label="Disk used"
          value={disk == null ? "—" : `${Math.round(disk)}%`}
          tone={disk == null ? "gray" : disk >= 90 ? "red" : disk >= 75 ? "orange" : "green"}
          note={disk != null && disk >= 90 ? "Almost full" : "Recording storage"}
        />
        <StatTile
          label="Last AI detection"
          value={h.last_detection || "—"}
          tone="purple"
          note={h.last_detection ? "Most recent detection" : "None reported"}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Services">
          <ul className="divide-y divide-[#E8F0FA] rounded-lg border border-[#D9E6F5] text-[12px]">
            {[
              ["Connection to HQ", h.api_connection, h.api_detail],
              ["AI (ML) service", h.ai_service, h.ai_detail],
            ].map(([label, value, detail]) => (
              <li key={label} className="px-2.5 py-2">
                <div className="flex justify-between">
                  <span className="text-[#607DA5]">{label}</span>
                  <span className={cn("font-semibold", tone(value))}>{value}</span>
                </div>
                {detail ? <p className="mt-0.5 text-[11px] text-[#90A7C4]">{detail}</p> : null}
              </li>
            ))}
          </ul>
          {h.sync_queue ? (
            <>
              <p className="mb-1.5 mt-4 text-[12px] font-semibold text-[#102C55]">Sync queue to HQ</p>
              <ul className="divide-y divide-[#E8F0FA] rounded-lg border border-[#D9E6F5] text-[12px]">
                {[
                  ["Records waiting", h.sync_queue.pending_records],
                  ["Pictures / videos waiting", h.sync_queue.pending_media],
                  ["Failed", h.sync_queue.failed],
                ].map(([label, value]) => (
                  <li key={label} className="flex justify-between px-2.5 py-1.5">
                    <span className="text-[#607DA5]">{label}</span>
                    <span className={cn("font-semibold tabular-nums", label === "Failed" && Number(value) > 0 ? "text-[#FF4D59]" : "text-[#102C55]")}>
                      {value}
                    </span>
                  </li>
                ))}
                {h.sync_queue.oldest_pending_sec ? (
                  <li className="flex justify-between px-2.5 py-1.5">
                    <span className="text-[#607DA5]">Oldest waiting</span>
                    <span className="font-semibold text-[#102C55]">{formatAge(h.sync_queue.oldest_pending_sec)}</span>
                  </li>
                ) : null}
              </ul>
            </>
          ) : null}
        </Card>

        <Card title="Devices by type" className="xl:col-span-2">
          {data.by_type.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-[#607DA5]">No device numbers reported.</p>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {data.by_type.map((t) => {
                const pct = t.total ? Math.round((100 * t.online) / t.total) : 0
                return (
                  <div key={t.type} className="rounded-lg border border-[#D9E6F5] p-3">
                    <p className="text-[12px] font-semibold text-[#102C55]">{t.type}</p>
                    <p className="mt-1 text-[18px] font-bold text-[#102C55]">
                      {t.online}
                      <span className="text-[13px] font-medium text-[#90A7C4]"> / {t.total} online</span>
                    </p>
                    <div className="mt-1.5 h-1.5 rounded-full bg-[#EEF4FB]">
                      <div
                        className="h-1.5 rounded-full"
                        style={{ width: `${pct}%`, backgroundColor: pct === 100 ? "#12A66A" : pct >= 50 ? "#FF9D2E" : "#FF4D59" }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Card>
      </div>

      {data.devices_listed ? (
        <Card
          title={`Devices (${devices.length})`}
          right={
            <div className="flex gap-2">
              <select value={type} onChange={(e) => setType(e.target.value)} className={inputClass}>
                <option value="">All types</option>
                {types.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <select value={status} onChange={(e) => setStatus(e.target.value)} className={inputClass}>
                <option value="">All status</option>
                {["online", "offline", "degraded", "fault", "unknown", "maintenance"].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          }
        >
          <div className="max-h-[480px] overflow-auto">
            <table className="w-full text-left text-[12px]">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                  <th className="py-2 pr-2 font-semibold">Device</th>
                  <th className="py-2 pr-2 font-semibold">Type</th>
                  <th className="py-2 pr-2 font-semibold">Site / Location</th>
                  <th className="py-2 pr-2 font-semibold">Status</th>
                  <th className="py-2 font-semibold">Last seen</th>
                </tr>
              </thead>
              <tbody>
                {devices.length === 0 ? (
                  <EmptyRow cols={5} text="No device matches the filter." />
                ) : (
                  devices.map((d, i) => (
                    <tr key={`${d.name}-${i}`} className="border-b border-[#F2F7FC]">
                      <td className="py-2 pr-2 font-semibold text-[#102C55]">{d.name}</td>
                      <td className="py-2 pr-2 text-[#607DA5]">{d.type}</td>
                      <td className="py-2 pr-2 text-[#607DA5]">{[d.site, d.location].filter(Boolean).join(" · ") || "—"}</td>
                      <td className={cn("py-2 pr-2 font-semibold capitalize", tone(d.status))}>{d.status}</td>
                      <td className="py-2 text-[#607DA5]" title={d.last_seen?.time}>
                        {d.last_seen?.ago || "Never"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <p className="text-[12px] text-[#90A7C4]">
          The full device list stays on the city server; HQ receives counts by type.
        </p>
      )}
    </div>
  )
}
