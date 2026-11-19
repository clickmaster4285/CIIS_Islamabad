import { useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { CityChartsSection, CityRangeDays } from "@/lib/city-section-api"
import { cn } from "@/lib/utils"
import { ReportButtons } from "./ReportButtons"
import { Card, LoadState, RangePicker, SectionHeader, useCitySection } from "./shared"

const axis = { fontSize: 11, fill: "#607DA5" }

export function ChartsSection({ cityId }: { cityId: string }) {
  const [days, setDays] = useState<CityRangeDays>(30)
  const { data, error, loading } = useCitySection<CityChartsSection>(cityId, "charts", { days })

  if (!data) return <LoadState error={error} loading={loading} />
  const comparison = [...data.comparison].sort((a, b) => b.detections - a.detections)
  const chartCities = comparison.map((c) => ({ ...c, name: c.city }))

  return (
    <div className="space-y-4">
      <SectionHeader title={`Charts & Reports — ${data.city.name}`} subtitle={data.range_label} loading={loading}>
        <RangePicker value={days} onChange={setDays} />
      </SectionHeader>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="AI detections per day" className="xl:col-span-2">
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
                <XAxis dataKey="day" tick={axis} minTickGap={8} />
                <YAxis tick={axis} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="people" name="People" stroke="#7552E8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="vehicles" name="Vehicles" stroke="#0875E1" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="weapons" name="Weapons" stroke="#FF4D59" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="plates" name="Plates" stroke="#12A66A" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="History & Reports">
          <ReportButtons cityId={cityId} days={days} />
          <p className="mt-3 text-[11px] leading-relaxed text-[#90A7C4]">
            Excel and Comparative use the period selected above. Reports carry case details only — no personal data.
          </p>
        </Card>
      </div>

      <Card title="Seizures, detentions & visitors per day">
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.ops}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
              <XAxis dataKey="day" tick={axis} minTickGap={8} />
              <YAxis tick={axis} allowDecimals={false} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="seizures" name="Seizures" fill="#FF4D59" radius={[3, 3, 0, 0]} />
              <Bar dataKey="detentions" name="Detentions" fill="#FF9D2E" radius={[3, 3, 0, 0]} />
              <Bar dataKey="visitors" name="Visitors" fill="#12A66A" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title={`All cities compared (${data.range_label.toLowerCase()})`}>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartCities}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8F0FA" />
                <XAxis dataKey="name" tick={axis} interval={0} />
                <YAxis tick={axis} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="seizures" name="Seizures" fill="#FF4D59" radius={[3, 3, 0, 0]} />
                <Bar dataKey="detentions" name="Detentions" fill="#FF9D2E" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-[#E8F0FA] text-[#607DA5]">
                  <th className="py-2 pr-2 font-semibold">City</th>
                  <th className="py-2 pr-2 font-semibold">Status</th>
                  <th className="py-2 pr-2 text-right font-semibold">Cameras</th>
                  <th className="py-2 pr-2 text-right font-semibold">AI</th>
                  <th className="py-2 pr-2 text-right font-semibold">Seizures</th>
                  <th className="py-2 text-right font-semibold">Detentions</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((c) => (
                  <tr key={c.id} className={cn("border-b border-[#F2F7FC]", c.id === data.city.id && "bg-[#EFF6FF]")}>
                    <td className="py-2 pr-2 font-semibold text-[#102C55]">
                      {c.city}
                      {c.is_hq ? <span className="ml-1 text-[10px] font-bold text-[#0875E1]">HQ</span> : null}
                    </td>
                    <td
                      className={cn(
                        "py-2 pr-2 font-semibold",
                        c.status === "Online" ? "text-[#12A66A]" : c.status === "Degraded" ? "text-[#E07F0F]" : "text-[#FF4D59]"
                      )}
                    >
                      {c.status}
                    </td>
                    <td className="py-2 pr-2 text-right tabular-nums text-[#607DA5]">
                      {c.cameras_online}/{c.cameras_total}
                    </td>
                    <td className="py-2 pr-2 text-right tabular-nums text-[#102C55]">{c.detections.toLocaleString()}</td>
                    <td className="py-2 pr-2 text-right tabular-nums text-[#102C55]">{c.seizures}</td>
                    <td className="py-2 text-right tabular-nums text-[#102C55]">{c.detentions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </div>
  )
}
