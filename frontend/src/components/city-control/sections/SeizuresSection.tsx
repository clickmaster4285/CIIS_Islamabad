import { useEffect, useState } from "react"
import { Search } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { CityRangeDays, CitySeizuresSection, CityStatusCount } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { Card, EmptyRow, LoadState, RangePicker, SectionHeader, StatTile, inputClass, useCitySection } from "./shared"

function formatPkr(value: number) {
  if (!value) return "—"
  return `PKR ${Math.round(value).toLocaleString()}`
}

function StatusList({ title, rows }: { title: string; rows: CityStatusCount[] }) {
  const total = rows.reduce((n, r) => n + r.count, 0)
  return (
    <div>
      <p className="mb-2 text-[12px] font-semibold text-[#102C55]">{title}</p>
      {rows.length === 0 ? (
        <p className="text-[12px] text-[#90A7C4]">No records in this period.</p>
      ) : (
        <ul className="space-y-1.5">
          {rows.map((r) => (
            <li key={r.status}>
              <div className="flex justify-between text-[12px]">
                <span className="text-[#607DA5]">{r.status}</span>
                <span className="font-semibold tabular-nums text-[#102C55]">{r.count}</span>
              </div>
              <div className="mt-0.5 h-1.5 rounded-full bg-[#EEF4FB]">
                <div className="h-1.5 rounded-full bg-[#0875E1]" style={{ width: `${(100 * r.count) / (total || 1)}%` }} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function SeizuresSection({ cityId }: { cityId: string }) {
  const [days, setDays] = useState<CityRangeDays>(30)
  const [search, setSearch] = useState("")
  const [q, setQ] = useState("")
  const [tab, setTab] = useState<"sheets" | "memos">("sheets")
  const { data, error, loading } = useCitySection<CitySeizuresSection>(cityId, "seizures", { days, q })

  // Search after the user stops typing.
  useEffect(() => {
    const t = window.setTimeout(() => setQ(search), 400)
    return () => window.clearTimeout(t)
  }, [search])

  if (!data) return <LoadState error={error} loading={loading} />
  const s = data.summary
  const rows = tab === "sheets" ? data.note_sheets : data.detention_memos
  const hitLimit = rows.length >= data.list_limit

  return (
    <div className="space-y-4">
      <SectionHeader
        title={`Seizures & Detentions — ${data.city.name}`}
        subtitle={`${data.range_label} · case details only (no personal data at HQ)`}
        loading={loading}
      >
        <RangePicker value={days} onChange={setDays} />
      </SectionHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatTile label="Seizures (note sheets)" value={s.seizures} tone="red" delta={s.seizure_delta} />
        <StatTile label="Detentions (memos)" value={s.detentions} tone="orange" delta={s.detention_delta} />
        <StatTile label="Seized goods value" value={formatPkr(s.seizure_value_pkr)} tone="gray" note="Assessed value, this period" />
        <StatTile label="Detained goods value" value={formatPkr(s.detention_value_pkr)} tone="gray" note="Assessable value, this period" />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Seizures & detentions per day" className="xl:col-span-2">
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#607DA5" }} minTickGap={8} />
                <YAxis tick={{ fontSize: 11, fill: "#607DA5" }} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="seizures" name="Seizures" fill="#FF4D59" radius={[3, 3, 0, 0]} />
                <Bar dataKey="detentions" name="Detentions" fill="#FF9D2E" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="By status">
          <div className="space-y-4">
            <StatusList title="Note sheets" rows={s.seizure_status} />
            <StatusList title="Detention memos" rows={s.detention_status} />
          </div>
        </Card>
      </div>

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex gap-1">
            {(
              [
                ["sheets", `Note sheets (${data.note_sheets.length})`],
                ["memos", `Detention memos (${data.detention_memos.length})`],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-[12px] font-semibold",
                  tab === id ? "bg-[#0875E1] text-white" : "text-[#607DA5] hover:bg-[#F4F8FD]"
                )}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="relative">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#90A7C4]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={tab === "sheets" ? "Note sheet / case / ref no..." : "Case / ref / FIR no..."}
              className={cn(inputClass, "w-[220px] pl-7")}
            />
          </label>
        </div>

        <div className="overflow-x-auto">
          {tab === "sheets" ? (
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                  <th className="py-2 pr-2 font-semibold">Note sheet / Case</th>
                  <th className="py-2 pr-2 font-semibold">Date</th>
                  <th className="py-2 pr-2 font-semibold">Subject / Goods</th>
                  <th className="py-2 pr-2 font-semibold">Officer</th>
                  <th className="py-2 pr-2 font-semibold">Status</th>
                  <th className="py-2 pr-2 text-right font-semibold">Items</th>
                  <th className="py-2 text-right font-semibold">Value</th>
                </tr>
              </thead>
              <tbody>
                {data.note_sheets.length === 0 ? (
                  <EmptyRow cols={7} text={q ? "Nothing matches the search." : "No note sheets in this period."} />
                ) : (
                  data.note_sheets.map((r) => (
                    <tr key={r.id} className="border-b border-[#F2F7FC] align-top">
                      <td className="py-2 pr-2">
                        <p className="font-semibold text-[#102C55]">{r.note_sheet_no || "—"}</p>
                        <p className="text-[11px] text-[#90A7C4]">{[r.case_no, r.reference_number].filter(Boolean).join(" · ") || "No case no."}</p>
                      </td>
                      <td className="py-2 pr-2 whitespace-nowrap text-[#607DA5]" title={r.ago}>{r.time}</td>
                      <td className="max-w-[280px] py-2 pr-2">
                        <p className="truncate text-[#102C55]">{r.subject || "—"}</p>
                        {r.goods ? <p className="truncate text-[11px] text-[#90A7C4]">{r.goods}</p> : null}
                      </td>
                      <td className="py-2 pr-2 text-[#607DA5]">
                        {r.officer || "—"}
                        {r.designation ? <p className="text-[11px] text-[#90A7C4]">{r.designation}</p> : null}
                      </td>
                      <td className="py-2 pr-2">
                        <span className="rounded-full border border-[#D9E6F5] bg-[#F7FBFF] px-2 py-0.5 text-[11px] font-semibold text-[#102C55]">{r.status}</span>
                        {r.priority && r.priority !== "Normal" ? <p className="mt-1 text-[11px] font-semibold text-[#FF4D59]">{r.priority}</p> : null}
                      </td>
                      <td className="py-2 pr-2 text-right tabular-nums text-[#102C55]">{r.items}</td>
                      <td className="py-2 text-right whitespace-nowrap tabular-nums font-semibold text-[#102C55]">{formatPkr(r.value_pkr)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                  <th className="py-2 pr-2 font-semibold">Case / FIR</th>
                  <th className="py-2 pr-2 font-semibold">Date</th>
                  <th className="py-2 pr-2 font-semibold">Type / Place</th>
                  <th className="py-2 pr-2 font-semibold">Goods</th>
                  <th className="py-2 pr-2 font-semibold">Officer</th>
                  <th className="py-2 pr-2 font-semibold">Status</th>
                  <th className="py-2 text-right font-semibold">Value</th>
                </tr>
              </thead>
              <tbody>
                {data.detention_memos.length === 0 ? (
                  <EmptyRow cols={7} text={q ? "Nothing matches the search." : "No detention memos in this period."} />
                ) : (
                  data.detention_memos.map((r) => (
                    <tr key={r.id} className="border-b border-[#F2F7FC] align-top">
                      <td className="py-2 pr-2">
                        <p className="font-semibold text-[#102C55]">{r.case_no || "—"}</p>
                        <p className="text-[11px] text-[#90A7C4]">{[r.fir_number && `FIR ${r.fir_number}`, r.reference_number].filter(Boolean).join(" · ") || "—"}</p>
                      </td>
                      <td className="py-2 pr-2 whitespace-nowrap text-[#607DA5]" title={r.ago}>{r.time}</td>
                      <td className="py-2 pr-2 text-[#607DA5]">
                        {r.detention_type || "—"}
                        {r.place_of_detention ? <p className="text-[11px] text-[#90A7C4]">{r.place_of_detention}</p> : null}
                      </td>
                      <td className="max-w-[240px] py-2 pr-2">
                        <p className="truncate text-[#102C55]">{r.goods || "—"}</p>
                        <p className="text-[11px] text-[#90A7C4]">{r.items} item{r.items === 1 ? "" : "s"}</p>
                      </td>
                      <td className="py-2 pr-2 text-[#607DA5]">{r.officer || "—"}</td>
                      <td className="py-2 pr-2">
                        <span className="rounded-full border border-[#D9E6F5] bg-[#F7FBFF] px-2 py-0.5 text-[11px] font-semibold capitalize text-[#102C55]">{r.status}</span>
                      </td>
                      <td className="py-2 text-right whitespace-nowrap tabular-nums font-semibold text-[#102C55]">{formatPkr(r.value_pkr)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
        {hitLimit ? (
          <p className="mt-2 text-[11px] text-[#90A7C4]">Showing the latest {data.list_limit}. Use search or a shorter period to find older records.</p>
        ) : null}
      </Card>
    </div>
  )
}
