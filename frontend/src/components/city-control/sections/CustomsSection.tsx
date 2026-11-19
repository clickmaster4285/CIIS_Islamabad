import { useState } from "react"
import { Flame } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { CityCustomsSection, CityRangeDays } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { Card, EmptyRow, LoadState, RangePicker, SectionHeader, StatTile, useCitySection } from "./shared"

const STAGE_LABELS: Record<string, string> = {
  arrived: "Arrived",
  registered: "Registered",
  face_captured: "Face captured",
  qr_printed: "QR printed",
  zone_checked_in: "Inside (checked in)",
  "checked-in": "Checked in",
  exited: "Exited",
}

function SeverityPill({ severity }: { severity: string }) {
  const s = severity.toLowerCase()
  const style =
    s === "critical" || s === "high"
      ? "border-[#FF4D5955] bg-[#FF4D5912] text-[#FF4D59]"
      : s === "medium"
        ? "border-[#FF9D2E55] bg-[#FF9D2E12] text-[#E07F0F]"
        : "border-[#D9E6F5] bg-[#F7FBFF] text-[#607DA5]"
  return <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-semibold", style)}>{severity}</span>
}

function AckPill({ acknowledged }: { acknowledged: boolean }) {
  return acknowledged ? (
    <span className="text-[11px] font-semibold text-[#12A66A]">Acknowledged</span>
  ) : (
    <span className="text-[11px] font-semibold text-[#FF4D59]">Open</span>
  )
}

export function CustomsSection({ cityId }: { cityId: string }) {
  const [days, setDays] = useState<CityRangeDays>(7)
  const { data, error, loading } = useCitySection<CityCustomsSection>(cityId, "customs", { days })

  if (!data) return <LoadState error={error} loading={loading} />
  const { warehouse, distributions, destruction_alerts: fire, gate_today: gate, security_alerts: sec } = data
  const totalScans = gate.scans_allowed + gate.scans_denied

  return (
    <div className="space-y-4">
      <SectionHeader
        title={`Customs Operations — ${data.city.name}`}
        subtitle={`Distributions & alerts: ${data.range_label} · Gate: today · Warehouse: current stock`}
        loading={loading}
      >
        <RangePicker value={days} onChange={setDays} />
      </SectionHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-6">
        <StatTile label="Warehouse items" value={warehouse.items.toLocaleString()} tone="blue" note="In stock records" />
        <StatTile
          label="Distributions"
          value={distributions.total}
          tone="purple"
          note={`${distributions.destructed} destructed · ${distributions.inventory} to inventory`}
        />
        <StatTile label="Fire / smoke alerts" value={fire.open} tone={fire.open ? "red" : "gray"} note={`open · ${fire.in_range} this period`} />
        <StatTile label="Visitors today" value={gate.visitors} tone="green" note={`${gate.on_premises} on premises · ${gate.exited} exited`} />
        <StatTile label="Vehicles today" value={gate.vehicles} tone="blue" note="Registered at the gate" />
        <StatTile label="Security alerts" value={sec.open} tone={sec.open ? "orange" : "gray"} note={`open · ${sec.in_range} this period`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Gate today">
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="rounded-lg bg-[#ECFDF5] p-2">
              <p className="text-[18px] font-bold text-[#12A66A]">{gate.scans_allowed}</p>
              <p className="text-[11px] text-[#607DA5]">Scans allowed</p>
            </div>
            <div className="rounded-lg bg-[#FEF2F2] p-2">
              <p className="text-[18px] font-bold text-[#FF4D59]">{gate.scans_denied}</p>
              <p className="text-[11px] text-[#607DA5]">Scans denied</p>
            </div>
          </div>
          {totalScans === 0 ? <p className="mt-2 text-center text-[11px] text-[#90A7C4]">No zone scans yet today.</p> : null}
          <p className="mb-1.5 mt-4 text-[12px] font-semibold text-[#102C55]">Visitors by stage</p>
          {gate.by_stage.length === 0 ? (
            <p className="text-[12px] text-[#90A7C4]">No visitors today.</p>
          ) : (
            <ul className="divide-y divide-[#E8F0FA] rounded-lg border border-[#D9E6F5] text-[12px]">
              {gate.by_stage.map((r) => (
                <li key={r.stage} className="flex justify-between px-2.5 py-1.5">
                  <span className="text-[#607DA5]">{STAGE_LABELS[r.stage] || r.stage}</span>
                  <span className="font-semibold tabular-nums text-[#102C55]">{r.count}</span>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-[11px] text-[#90A7C4]">Visitor names and CNICs stay on the city server.</p>
        </Card>

        <Card title="Visitors & vehicles per day" className="xl:col-span-2">
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.daily_visitors}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#607DA5" }} minTickGap={8} />
                <YAxis tick={{ fontSize: 11, fill: "#607DA5" }} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="visitors" name="Visitors" fill="#12A66A" radius={[3, 3, 0, 0]} />
                <Bar dataKey="vehicles" name="Vehicles" fill="#0875E1" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Warehouse stock (latest 20)">
        {warehouse.by_status.length ? (
          <div className="mb-3 flex flex-wrap gap-2">
            {warehouse.by_status.map((r) => (
              <span key={r.status} className="rounded-full border border-[#D9E6F5] bg-[#F7FBFF] px-2.5 py-0.5 text-[11px] text-[#607DA5]">
                {r.status}: <b className="text-[#102C55]">{r.count}</b>
              </span>
            ))}
          </div>
        ) : null}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                <th className="py-2 pr-2 font-semibold">Case ref / QR</th>
                <th className="py-2 pr-2 font-semibold">Description</th>
                <th className="py-2 pr-2 text-right font-semibold">Quantity</th>
                <th className="py-2 pr-2 font-semibold">Godown</th>
                <th className="py-2 pr-2 font-semibold">Status</th>
                <th className="py-2 font-semibold">Updated</th>
              </tr>
            </thead>
            <tbody>
              {warehouse.recent.length === 0 ? (
                <EmptyRow cols={6} text="No warehouse stock synced from this city." />
              ) : (
                warehouse.recent.map((r, i) => (
                  <tr key={`${r.qr_code}-${i}`} className="border-b border-[#F2F7FC]">
                    <td className="py-2 pr-2">
                      <p className="font-semibold text-[#102C55]">{r.case_ref || "—"}</p>
                      {r.qr_code ? <p className="text-[11px] text-[#90A7C4]">{r.qr_code}</p> : null}
                    </td>
                    <td className="max-w-[280px] truncate py-2 pr-2 text-[#102C55]">{r.description || "—"}</td>
                    <td className="py-2 pr-2 text-right tabular-nums text-[#102C55]">
                      {r.quantity.toLocaleString()} {r.unit}
                    </td>
                    <td className="py-2 pr-2 text-[#607DA5]">{r.godown || "—"}</td>
                    <td className="py-2 pr-2 text-[#607DA5]">{r.status}</td>
                    <td className="py-2 whitespace-nowrap text-[#607DA5]" title={r.time}>{r.ago}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Memo distributions (latest 10)">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                <th className="py-2 pr-2 font-semibold">Case</th>
                <th className="py-2 pr-2 font-semibold">Outcome</th>
                <th className="py-2 pr-2 font-semibold">Status</th>
                <th className="py-2 pr-2 font-semibold">By</th>
                <th className="py-2 font-semibold">When</th>
              </tr>
            </thead>
            <tbody>
              {distributions.recent.length === 0 ? (
                <EmptyRow cols={5} text="No distributions in this period." />
              ) : (
                distributions.recent.map((r, i) => (
                  <tr key={`${r.at}-${i}`} className="border-b border-[#F2F7FC]">
                    <td className="py-2 pr-2 font-semibold text-[#102C55]">
                      {r.case_no || "—"}
                      {r.smoke_fire ? <Flame className="ml-1 inline h-3.5 w-3.5 text-[#FF4D59]" aria-label="Smoke / fire detected" /> : null}
                    </td>
                    <td className="py-2 pr-2 text-[#607DA5]">{r.outcome}</td>
                    <td className="py-2 pr-2 text-[#607DA5]">{r.status}</td>
                    <td className="py-2 pr-2 text-[#607DA5]">{r.performed_by || "—"}</td>
                    <td className="py-2 whitespace-nowrap text-[#607DA5]" title={r.time}>{r.ago}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </Card>

        <Card title="Fire / smoke alerts (latest 10)">
          {fire.recent.length === 0 ? (
            <p className="py-6 text-center text-[13px] text-[#607DA5]">No fire or smoke alerts in this period.</p>
          ) : (
            <ul className="space-y-2">
              {fire.recent.map((r, i) => (
                <li key={`${r.at}-${i}`} className="rounded-lg border border-[#F2F7FC] p-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12px] font-semibold text-[#102C55]">{r.case_no || "No case no."}</span>
                    <div className="flex items-center gap-2">
                      <SeverityPill severity={r.severity} />
                      <AckPill acknowledged={r.acknowledged} />
                    </div>
                  </div>
                  <p className="mt-1 text-[12px] text-[#607DA5]">{r.message}</p>
                  <p className="mt-0.5 text-[11px] text-[#90A7C4]">{r.time} · {r.ago}</p>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card title="Security alerts at gates & zones (latest 10)">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
              <th className="py-2 pr-2 font-semibold">Alert</th>
              <th className="py-2 pr-2 font-semibold">Severity</th>
              <th className="py-2 pr-2 font-semibold">Zone / Gate</th>
              <th className="py-2 pr-2 font-semibold">State</th>
              <th className="py-2 font-semibold">When</th>
            </tr>
          </thead>
          <tbody>
            {sec.recent.length === 0 ? (
              <EmptyRow cols={5} text="No security alerts in this period." />
            ) : (
              sec.recent.map((r, i) => (
                <tr key={`${r.at}-${i}`} className="border-b border-[#F2F7FC]">
                  <td className="py-2 pr-2 font-semibold text-[#102C55]">{r.type}</td>
                  <td className="py-2 pr-2">
                    <SeverityPill severity={r.severity} />
                  </td>
                  <td className="py-2 pr-2 text-[#607DA5]">{[r.zone, r.gate].filter(Boolean).join(" · ") || "—"}</td>
                  <td className="py-2 pr-2">
                    <AckPill acknowledged={r.acknowledged} />
                  </td>
                  <td className="py-2 whitespace-nowrap text-[#607DA5]" title={r.time}>{r.ago}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
