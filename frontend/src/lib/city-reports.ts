/**
 * City page reports (History & Reports / Charts & Reports):
 * Daily Snapshot (PDF), Export Data (Excel), Comparative Report (PDF).
 * Built in the browser from the same read-only section API the screens use.
 */
import {
  fetchCitySection,
  type CityAiSection,
  type CityCamerasSection,
  type CityChartsSection,
  type CityComparisonRow,
  type CityCustomsSection,
  type CityInfraSection,
  type CityRangeDays,
  type CitySeizuresSection,
} from "@/lib/city-section-api"

type Cell = string | number
type JsPdf = import("jspdf").jsPDF

const BLUE: [number, number, number] = [8, 117, 225]
const DARK: [number, number, number] = [16, 44, 85]
const MUTED: [number, number, number] = [96, 125, 165]

function stamp() {
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2, "0")
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

function fileSafe(name: string) {
  return name.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "")
}

function pkr(v: number) {
  return v ? `PKR ${Math.round(v).toLocaleString()}` : "-"
}

/** Small PDF writer: title, key/value blocks and tables with page breaks. */
class PdfDoc {
  pdf: JsPdf
  y = 0
  w: number
  h: number
  margin = 12

  constructor(pdf: JsPdf) {
    this.pdf = pdf
    this.w = pdf.internal.pageSize.getWidth()
    this.h = pdf.internal.pageSize.getHeight()
  }

  ensure(space: number) {
    if (this.y + space > this.h - 14) {
      this.pdf.addPage()
      this.y = this.margin + 4
    }
  }

  header(title: string, subtitle: string) {
    const { pdf } = this
    pdf.setFillColor(...BLUE)
    pdf.rect(0, 0, this.w, 22, "F")
    pdf.setTextColor(255, 255, 255)
    pdf.setFont("helvetica", "bold")
    pdf.setFontSize(15)
    pdf.text(title, this.margin, 10)
    pdf.setFont("helvetica", "normal")
    pdf.setFontSize(9)
    pdf.text(subtitle, this.margin, 17)
    this.y = 30
  }

  heading(text: string) {
    this.ensure(14)
    const { pdf } = this
    pdf.setTextColor(...DARK)
    pdf.setFont("helvetica", "bold")
    pdf.setFontSize(11)
    pdf.text(text, this.margin, this.y)
    pdf.setDrawColor(217, 230, 245)
    pdf.line(this.margin, this.y + 1.5, this.w - this.margin, this.y + 1.5)
    this.y += 7
  }

  /** Label/value pairs in columns. */
  pairs(items: [string, Cell][], cols = 3) {
    const { pdf } = this
    const colW = (this.w - 2 * this.margin) / cols
    for (let i = 0; i < items.length; i += cols) {
      this.ensure(11)
      items.slice(i, i + cols).forEach(([label, value], j) => {
        const x = this.margin + j * colW
        pdf.setFont("helvetica", "normal")
        pdf.setFontSize(8)
        pdf.setTextColor(...MUTED)
        pdf.text(label, x, this.y)
        pdf.setFont("helvetica", "bold")
        pdf.setFontSize(11)
        pdf.setTextColor(...DARK)
        pdf.text(String(value), x, this.y + 5)
      })
      this.y += 11
    }
    this.y += 2
  }

  table(head: string[], rows: Cell[][], widths?: number[], empty = "No records.") {
    const { pdf } = this
    const total = this.w - 2 * this.margin
    const ws = widths ? widths.map((p) => (p / widths.reduce((a, b) => a + b, 0)) * total) : head.map(() => total / head.length)
    const drawHead = () => {
      pdf.setFillColor(239, 246, 255)
      pdf.rect(this.margin, this.y - 4, total, 6.5, "F")
      pdf.setFont("helvetica", "bold")
      pdf.setFontSize(8)
      pdf.setTextColor(...DARK)
      let x = this.margin + 1.5
      head.forEach((hd, i) => {
        pdf.text(hd, x, this.y)
        x += ws[i]
      })
      this.y += 6
    }
    this.ensure(14)
    drawHead()
    if (rows.length === 0) {
      pdf.setFont("helvetica", "italic")
      pdf.setFontSize(8)
      pdf.setTextColor(...MUTED)
      pdf.text(empty, this.margin + 1.5, this.y)
      this.y += 8
      return
    }
    pdf.setFont("helvetica", "normal")
    for (const row of rows) {
      if (this.y + 6 > this.h - 14) {
        pdf.addPage()
        this.y = this.margin + 4
        drawHead()
        pdf.setFont("helvetica", "normal")
      }
      pdf.setFontSize(8)
      pdf.setTextColor(...DARK)
      let x = this.margin + 1.5
      row.forEach((cell, i) => {
        const text = pdf.splitTextToSize(String(cell ?? ""), ws[i] - 2)[0] || ""
        pdf.text(text, x, this.y)
        x += ws[i]
      })
      pdf.setDrawColor(242, 247, 252)
      pdf.line(this.margin, this.y + 2, this.margin + total, this.y + 2)
      this.y += 5.5
    }
    this.y += 4
  }

  footer(note: string) {
    const { pdf } = this
    const pages = pdf.getNumberOfPages()
    for (let i = 1; i <= pages; i++) {
      pdf.setPage(i)
      pdf.setFont("helvetica", "normal")
      pdf.setFontSize(7)
      pdf.setTextColor(...MUTED)
      pdf.text(note, this.margin, this.h - 6)
      pdf.text(`Page ${i} of ${pages}`, this.w - this.margin, this.h - 6, { align: "right" })
    }
  }
}

async function newPdf(orientation: "portrait" | "landscape" = "portrait") {
  const { jsPDF } = await import("jspdf")
  return new PdfDoc(new jsPDF({ unit: "mm", format: "a4", orientation }))
}

const FOOTER = "Pakistan Customs - City Control Grid - read-only HQ copy - case details only, no personal data"

// --------------------------------------------------------------------- Daily Snapshot

export async function downloadDailySnapshotPdf(cityId: string) {
  const [cams, ai, ops, customs, infra] = await Promise.all([
    fetchCitySection<CityCamerasSection>(cityId, "cameras"),
    fetchCitySection<CityAiSection>(cityId, "ai", { days: 1 }),
    fetchCitySection<CitySeizuresSection>(cityId, "seizures", { days: 1 }),
    fetchCitySection<CityCustomsSection>(cityId, "customs", { days: 1 }),
    fetchCitySection<CityInfraSection>(cityId, "infra"),
  ])
  const city = cams.city.name
  const doc = await newPdf()
  doc.header(`${city} - Daily Snapshot`, `${new Date().toLocaleDateString(undefined, { dateStyle: "full" })} - generated ${new Date().toLocaleTimeString()}`)

  const h = infra.health
  doc.heading("System status")
  doc.pairs([
    ["Connection to HQ", h.api_connection],
    ["AI service", h.ai_service],
    ["Infrastructure", h.infra_health || "Not reported"],
    ["Cameras online", `${cams.summary.online} / ${cams.summary.total}`],
    ["Server uptime", h.uptime || "-"],
    ["Disk used", h.disk_percent == null ? "-" : `${Math.round(h.disk_percent)}%`],
  ])

  doc.heading("Today")
  const t = ai.summary.by_type
  doc.pairs([
    ["AI detections", ai.summary.delta === "—" ? String(ai.summary.total) : `${ai.summary.total} (${ai.summary.delta} vs yesterday)`],
    ["People / Vehicles", `${t.Person || 0} / ${t.Vehicle || 0}`],
    ["Weapons / Plates", `${t.Weapon || 0} / ${t["License Plate"] || 0}`],
    ["Seizures", ops.summary.seizures],
    ["Detentions", ops.summary.detentions],
    ["Seized + detained value", pkr(ops.summary.seizure_value_pkr + ops.summary.detention_value_pkr)],
    ["Visitors (on premises)", `${customs.gate_today.visitors} (${customs.gate_today.on_premises})`],
    ["Vehicles at gate", customs.gate_today.vehicles],
    ["Open alerts (security / fire)", `${customs.security_alerts.open} / ${customs.destruction_alerts.open}`],
  ])

  doc.heading("Busiest cameras today")
  doc.table(["Camera", "Detections"], ai.top_cameras.map((c) => [c.camera, c.count]), [4, 1], "No detections today.")

  doc.heading("Latest AI detections")
  doc.table(
    ["Time", "Type", "Object", "Camera", "Conf."],
    ai.items.map((r) => [r.date, r.type + (r.is_alert ? " (ALERT)" : ""), r.object, r.camera, `${r.confidence}%`]),
    [2, 2, 2, 3, 1],
    "No detections today."
  )

  doc.heading("Today's note sheets")
  doc.table(
    ["Note sheet", "Case", "Officer", "Status", "Items", "Value"],
    ops.note_sheets.map((r) => [r.note_sheet_no || "-", r.case_no || "-", r.officer || "-", r.status, r.items, pkr(r.value_pkr)]),
    [2, 2, 2, 2, 1, 2],
    "No note sheets today."
  )
  doc.heading("Today's detention memos")
  doc.table(
    ["Case", "FIR", "Type", "Status", "Items", "Value"],
    ops.detention_memos.map((r) => [r.case_no || "-", r.fir_number || "-", r.detention_type || "-", r.status, r.items, pkr(r.value_pkr)]),
    [2, 2, 2, 2, 1, 2],
    "No detention memos today."
  )

  doc.footer(FOOTER)
  doc.pdf.save(`${fileSafe(city)}-daily-snapshot-${stamp()}.pdf`)
}

// --------------------------------------------------------------------- Comparative Report

function comparisonRows(rows: CityComparisonRow[]): Cell[][] {
  const sum = (k: keyof CityComparisonRow) => rows.reduce((n, r) => n + Number(r[k] || 0), 0)
  const share = (v: number, total: number) =>
    total ? `${v.toLocaleString()} (${Math.round((100 * v) / total)}%)` : v.toLocaleString()
  const totals = { d: sum("detections"), s: sum("seizures"), m: sum("detentions") }
  return rows.map((r) => [
    r.city + (r.is_hq ? " (HQ)" : ""),
    r.status,
    `${r.cameras_online}/${r.cameras_total}`,
    share(r.detections, totals.d),
    share(r.seizures, totals.s),
    share(r.detentions, totals.m),
    r.staff_total,
  ])
}

export async function downloadComparativePdf(cityId: string, days: CityRangeDays) {
  const data = await fetchCitySection<CityChartsSection>(cityId, "charts", { days })
  const rows = [...data.comparison].sort((a, b) => b.detections - a.detections)
  const doc = await newPdf("landscape")
  doc.header("City Comparative Report", `${data.range_label} - all cities side by side - generated ${new Date().toLocaleString()}`)

  const sum = (k: keyof CityComparisonRow) => rows.reduce((n, r) => n + Number(r[k] || 0), 0)
  doc.heading("All cities")
  doc.pairs(
    [
      ["Cities", rows.length],
      ["Online", rows.filter((r) => r.status === "Online").length],
      ["Cameras online", `${sum("cameras_online")} / ${sum("cameras_total")}`],
      ["AI detections (4 types)", sum("detections").toLocaleString()],
      ["Seizures", sum("seizures")],
      ["Detentions", sum("detentions")],
    ],
    6
  )

  doc.heading("By city (share of total in brackets)")
  doc.table(["City", "Status", "Cameras", "AI detections", "Seizures", "Detentions", "Staff"], comparisonRows(rows), [3, 2, 2, 3, 2, 2, 1])

  const focus = rows.find((r) => r.id === data.city.id)
  if (focus && rows.length > 1) {
    // No rank when the city has none (a tie at zero says nothing).
    const rank = (k: keyof CityComparisonRow) =>
      Number(focus[k])
        ? `#${[...rows].sort((a, b) => Number(b[k]) - Number(a[k])).findIndex((r) => r.id === focus.id) + 1} of ${rows.length}`
        : "-"
    doc.heading(`${focus.city} ranking`)
    doc.pairs([
      ["AI detections", rank("detections")],
      ["Seizures", rank("seizures")],
      ["Detentions", rank("detentions")],
    ])
  }

  doc.footer(FOOTER)
  doc.pdf.save(`city-comparative-report-${days}d-${stamp()}.pdf`)
}

// --------------------------------------------------------------------- Excel export

export async function downloadCityExcel(cityId: string, days: CityRangeDays) {
  const [charts, ops, cams, ai, customs] = await Promise.all([
    fetchCitySection<CityChartsSection>(cityId, "charts", { days }),
    fetchCitySection<CitySeizuresSection>(cityId, "seizures", { days }),
    fetchCitySection<CityCamerasSection>(cityId, "cameras"),
    fetchCitySection<CityAiSection>(cityId, "ai", { days }),
    fetchCitySection<CityCustomsSection>(cityId, "customs", { days }),
  ])
  const XLSX = await import("xlsx")
  const wb = XLSX.utils.book_new()
  const add = (name: string, rows: Record<string, Cell | boolean>[], fallback: string) => {
    const ws = XLSX.utils.json_to_sheet(rows.length ? rows : [{ Note: fallback }])
    XLSX.utils.book_append_sheet(wb, ws, name.slice(0, 31))
  }
  const city = charts.city.name
  const t = ai.summary.by_type

  add(
    "Summary",
    [
      { Item: "City", Value: city },
      { Item: "Period", Value: charts.range_label },
      { Item: "Generated", Value: new Date().toLocaleString() },
      { Item: "Cameras online", Value: `${cams.summary.online} / ${cams.summary.total}` },
      { Item: "AI detections", Value: ai.summary.total },
      { Item: "People", Value: t.Person || 0 },
      { Item: "Vehicles", Value: t.Vehicle || 0 },
      { Item: "Weapons", Value: t.Weapon || 0 },
      { Item: "License plates", Value: t["License Plate"] || 0 },
      { Item: "AI alerts", Value: ai.summary.alerts },
      { Item: "Seizures (note sheets)", Value: ops.summary.seizures },
      { Item: "Seized goods value (PKR)", Value: ops.summary.seizure_value_pkr },
      { Item: "Detentions (memos)", Value: ops.summary.detentions },
      { Item: "Detained goods value (PKR)", Value: ops.summary.detention_value_pkr },
      { Item: "Warehouse items", Value: customs.warehouse.items },
      { Item: "Memo distributions", Value: customs.distributions.total },
      { Item: "Security alerts", Value: customs.security_alerts.in_range },
      { Item: "Fire / smoke alerts", Value: customs.destruction_alerts.in_range },
      { Item: "Note", Value: "Read-only HQ copy. Case details only, no personal data." },
    ],
    ""
  )
  add(
    "Detections per day",
    charts.trend.map((r) => ({ Day: r.day, People: r.people, Vehicles: r.vehicles, Weapons: r.weapons, Plates: r.plates })),
    "No detections."
  )
  add(
    "Operations per day",
    charts.ops.map((r) => ({ Day: r.day, Seizures: r.seizures, Detentions: r.detentions, Visitors: r.visitors })),
    "No records."
  )
  add("Detections by hour", ai.by_hour.map((r) => ({ Hour: `${r.hour}:00`, Detections: r.count })), "No detections.")
  add(
    "Note sheets",
    ops.note_sheets.map((r) => ({
      "Note sheet no": r.note_sheet_no,
      "Case no": r.case_no,
      Reference: r.reference_number,
      Date: r.time,
      Subject: r.subject,
      Office: r.office,
      Officer: r.officer,
      Designation: r.designation,
      Status: r.status,
      Priority: r.priority,
      Items: r.items,
      Goods: r.goods,
      "Value (PKR)": r.value_pkr,
    })),
    "No note sheets in this period."
  )
  add(
    "Detention memos",
    ops.detention_memos.map((r) => ({
      "Case no": r.case_no,
      "FIR no": r.fir_number,
      Reference: r.reference_number,
      Date: r.time,
      Type: r.detention_type,
      "Place of detention": r.place_of_detention,
      Officer: r.officer,
      Status: r.status,
      Items: r.items,
      Goods: r.goods,
      "Value (PKR)": r.value_pkr,
    })),
    "No detention memos in this period."
  )
  add(
    "Cameras",
    cams.cameras.map((c) => ({
      Camera: c.name,
      Code: c.code,
      Zone: c.zone,
      Location: c.location,
      Site: c.site,
      Channel: c.channel ?? "",
      Status: c.status ?? "",
      "Detections 24h": c.detections_24h,
      "Last detection": c.last_detection?.time ?? "Never",
    })),
    "No camera list from this city."
  )
  add(
    "Warehouse",
    customs.warehouse.recent.map((r) => ({
      "Case ref": r.case_ref,
      QR: r.qr_code,
      Description: r.description,
      Quantity: r.quantity,
      Unit: r.unit,
      Godown: r.godown,
      Status: r.status,
      Updated: r.time,
    })),
    "No warehouse stock."
  )
  add(
    "City comparison",
    charts.comparison.map((r) => ({
      City: r.city,
      Status: r.status,
      "Cameras online": r.cameras_online,
      "Cameras total": r.cameras_total,
      "AI detections": r.detections,
      Seizures: r.seizures,
      Detentions: r.detentions,
      Staff: r.staff_total,
    })),
    "No cities."
  )
  XLSX.writeFile(wb, `${fileSafe(city)}-city-data-${days}d-${stamp()}.xlsx`)
}
