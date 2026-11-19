import { useEffect, useMemo, useState } from "react"
import { SourceLocationSelect } from "@/components/sync/source-location-select"
import { ALL_SOURCE_LOCATIONS, useSourceLocations } from "@/lib/source-location"
import { AlertTriangle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Eye, Search } from "lucide-react"
import { ModulePageLayout } from "@/components/dashboard/module-page-layout"
import { TableActionGroup, TableActionIcon } from "@/components/seizure/table-action-icon"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ExportMenu } from "@/components/seizure/export-menu"
import DetentionMemoReportPrint from "@/components/detention/DetentionMemoReportPrint"
import { downloadDetentionMemoCsv } from "@/lib/detention-memo-csv"
import { useBatchPdfExport } from "@/hooks/use-batch-pdf-export"
import { PdfExportHost } from "@/components/seizure/pdf-export-host"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ROUTES, getDetentionMemoDetailPath } from "@/routes/config"
import { fetchDetentionMemos, type DetentionMemoApiRecord } from "@/lib/detention-memo-api"
import {
  DETENTION_WINDOW_DAYS,
  daysSinceDetention,
  fetchAssessments,
  type DetentionAssessmentRecord,
} from "@/lib/seizure-management-api"
import { cn } from "@/lib/utils"

const PAGE_SIZE_OPTIONS = [10, 20, 25, 50, 100]
const DEFAULT_PAGE_SIZE = 20

type ReportFilter = "all" | "assessed" | "pending" | "overdue"

const LIST_FILTERS: { id: ReportFilter; label: string; valueClass: string; activeClass: string }[] = [
  { id: "all", label: "Total Memos", valueClass: "text-foreground", activeClass: "border-[#0875E1] bg-[#F4F8FD] ring-1 ring-[#0875E1]" },
  { id: "assessed", label: "With Assessment", valueClass: "text-emerald-700", activeClass: "border-emerald-300 bg-emerald-50 ring-1 ring-emerald-200" },
  { id: "pending", label: "Pending Assessment", valueClass: "text-sky-700", activeClass: "border-sky-300 bg-sky-50 ring-1 ring-sky-200" },
  { id: "overdue", label: `Over ${DETENTION_WINDOW_DAYS} Days`, valueClass: "text-amber-700", activeClass: "border-amber-300 bg-amber-50 ring-1 ring-amber-200" },
]

function shortPlace(place: string | undefined): string {
  const p = (place || "").trim()
  if (!p) return "—"
  return p.replace(/^Model Customs Collectorate,?\s*/i, "MCC ").replace(/\s+/g, " ")
}

function isOverdue(dateTimeDetention: string): boolean {
  const days = daysSinceDetention(dateTimeDetention)
  return days !== null && days > DETENTION_WINDOW_DAYS
}

export default function DetentionReportingPage() {
  const [rows, setRows] = useState<DetentionMemoApiRecord[]>([])
  const [assessments, setAssessments] = useState<DetentionAssessmentRecord[]>([])
  const [search, setSearch] = useState("")
  const [loading, setLoading] = useState(true)
  const [listFilter, setListFilter] = useState<ReportFilter>("all")
  const [sourceLocation, setSourceLocation] = useState(ALL_SOURCE_LOCATIONS)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const locations = useSourceLocations()

  useEffect(() => {
    setLoading(true)
    Promise.all([
      fetchDetentionMemos().catch(() => [] as DetentionMemoApiRecord[]),
      fetchAssessments().catch(() => [] as DetentionAssessmentRecord[]),
    ])
      .then(([memos, assess]) => {
        setRows(memos)
        setAssessments(assess)
      })
      .finally(() => setLoading(false))
  }, [])

  const assessedMemoIds = useMemo(() => {
    const set = new Set<string>()
    for (const a of assessments) set.add(a.detentionMemoId)
    return set
  }, [assessments])

  const locationRows = useMemo(
    () => rows.filter((r) => locations.matches(r.locationCode, sourceLocation)),
    [rows, sourceLocation, locations]
  )

  const stats = useMemo(() => {
    const overdue = locationRows.filter((r) => isOverdue(r.dateTimeDetention)).length
    const withAssessment = locationRows.filter((r) => assessedMemoIds.has(r.id)).length
    return {
      all: locationRows.length,
      assessed: withAssessment,
      pending: locationRows.length - withAssessment,
      overdue,
    }
  }, [locationRows, assessedMemoIds])

  const filtered = useMemo(() => {
    let list = locationRows
    if (listFilter === "assessed") list = list.filter((r) => assessedMemoIds.has(r.id))
    else if (listFilter === "pending") list = list.filter((r) => !assessedMemoIds.has(r.id))
    else if (listFilter === "overdue") list = list.filter((r) => isOverdue(r.dateTimeDetention))
    const q = search.trim().toLowerCase()
    if (!q) return list
    return list.filter(
      (r) =>
        r.caseNo.toLowerCase().includes(q) ||
        (r.referenceNumber ?? "").toLowerCase().includes(q) ||
        r.placeOfDetention.toLowerCase().includes(q) ||
        (r.verificationStatus ?? "").toLowerCase().includes(q)
    )
  }, [locationRows, listFilter, search, assessedMemoIds])

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const pageRows = useMemo(
    () => filtered.slice((page - 1) * pageSize, page * pageSize),
    [filtered, page, pageSize]
  )
  const pageSerial = (index: number) => (page - 1) * pageSize + index + 1

  useEffect(() => {
    if (page > totalPages) setPage(totalPages)
  }, [page, totalPages])

  const pdf = useBatchPdfExport<DetentionMemoApiRecord>(`detention-report-${new Date().toISOString().slice(0, 10)}.pdf`)

  const exportCsv = () => {
    downloadDetentionMemoCsv(`detention-report-${new Date().toISOString().slice(0, 10)}.csv`, filtered, {
      headers: ["Days Since Detention", "Assessment", "60-Day Status"],
      values: (r) => {
        const days = daysSinceDetention(r.dateTimeDetention)
        return [
          days ?? "",
          assessedMemoIds.has(r.id) ? "Yes" : "No",
          days === null ? "" : days > DETENTION_WINDOW_DAYS ? "Overdue" : "Within window",
        ]
      },
    })
  }

  return (
    <ModulePageLayout
      title="Detention Reporting"
      description="Summary of detention memos, assessment coverage, and 60-day recovery window compliance."
      breadcrumbs={[
        { label: "Seizure Management", href: ROUTES.SEIZURE_MANAGEMENT },
        { label: "Detention" },
        { label: "Reporting" },
      ]}
      actions={
        <ExportMenu
          disabled={filtered.length === 0}
          onExportCsv={exportCsv}
          onExportPdf={() => pdf.start(filtered)}
        />
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
          {LIST_FILTERS.map((stat) => (
            <button
              key={stat.id}
              type="button"
              onClick={() => {
                setListFilter((current) => (current === stat.id ? "all" : stat.id))
                setPage(1)
              }}
              className={cn(
                "min-w-0 overflow-hidden rounded-xl border px-3 py-3 text-left shadow-sm transition",
                listFilter === stat.id
                  ? stat.activeClass
                  : "border-border/80 bg-card hover:border-slate-300 hover:bg-slate-50"
              )}
            >
              <p className="text-[11px] font-medium uppercase leading-snug tracking-wide text-muted-foreground">
                {stat.label}
              </p>
              <p className={cn("mt-1 text-2xl font-semibold tabular-nums tracking-tight", stat.valueClass)}>
                {loading ? "—" : stats[stat.id]}
              </p>
            </button>
          ))}
        </div>

        <Card className="overflow-hidden rounded-xl border-border/80 shadow-sm">
          <CardContent className="space-y-0 p-0">
            <div className="flex flex-wrap items-center gap-2 border-b bg-slate-50/60 px-4 py-3">
              <div className="relative min-w-[12rem] flex-1 basis-56">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  className="h-8 bg-white pl-8 text-sm"
                  placeholder="Search case no, memo no, place…"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value)
                    setPage(1)
                  }}
                />
              </div>
              {locations.enabled && (
                <SourceLocationSelect
                  value={sourceLocation}
                  onChange={(v) => {
                    setSourceLocation(v)
                    setPage(1)
                  }}
                  options={locations.options}
                  className="h-8 w-full shrink-0 bg-white text-sm sm:w-44"
                />
              )}
              <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums">
                {filtered.length} of {locationRows.length} shown
              </span>
            </div>

            <div className="space-y-2 p-3 lg:hidden">
              {loading ? (
                <p className="py-8 text-center text-sm text-muted-foreground">Loading detention memos…</p>
              ) : pageRows.length === 0 ? (
                <div className="rounded-lg border border-dashed px-4 py-10 text-center text-sm text-muted-foreground">
                  No detention memos found.
                </div>
              ) : (
                pageRows.map((row, index) => {
                  const days = daysSinceDetention(row.dateTimeDetention)
                  const overdue = isOverdue(row.dateTimeDetention)
                  const hasAssessment = assessedMemoIds.has(row.id)
                  return (
                    <div
                      key={row.id}
                      className={cn(
                        "rounded-xl border bg-card p-3 shadow-sm",
                        overdue && "border-amber-200 bg-amber-50/40"
                      )}
                    >
                      <div className="mb-2 flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            Sr. {pageSerial(index)}
                          </p>
                          <p className="truncate font-mono text-sm font-semibold text-black">{row.caseNo || "—"}</p>
                          <p className="truncate font-mono text-[11px] text-muted-foreground">
                            {row.referenceNumber || "—"}
                          </p>
                        </div>
                        {overdue ? (
                          <Badge className="shrink-0 gap-1 rounded-md border-0 bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800 hover:bg-amber-100">
                            <AlertTriangle className="h-3 w-3" />
                            Overdue
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                            Within window
                          </Badge>
                        )}
                      </div>
                      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                        {locations.enabled && (
                          <div className="min-w-0">
                            <dt className="text-[10px] text-muted-foreground">Location</dt>
                            <dd className="truncate">{locations.labelFor(row.locationCode)}</dd>
                          </div>
                        )}
                        <div className="min-w-0">
                          <dt className="text-[10px] text-muted-foreground">Detention date</dt>
                          <dd className="tabular-nums">{row.dateTimeDetention?.slice(0, 10) || "—"}</dd>
                        </div>
                        <div className="min-w-0">
                          <dt className="text-[10px] text-muted-foreground">Days</dt>
                          <dd className={cn("tabular-nums", overdue && "font-semibold text-amber-800")}>{days ?? "—"}</dd>
                        </div>
                        <div className="col-span-2 min-w-0">
                          <dt className="text-[10px] text-muted-foreground">Place</dt>
                          <dd className="truncate" title={row.placeOfDetention || ""}>
                            {shortPlace(row.placeOfDetention)}
                          </dd>
                        </div>
                        <div className="min-w-0">
                          <dt className="text-[10px] text-muted-foreground">Assessment</dt>
                          <dd>{hasAssessment ? "Done" : "Pending"}</dd>
                        </div>
                      </dl>
                      <div className="mt-2 flex justify-end border-t pt-2">
                        <TableActionGroup className="w-max max-w-none flex-nowrap gap-0">
                          <TableActionIcon label="View memo" to={getDetentionMemoDetailPath(row.id)}>
                            <Eye className="h-4 w-4" />
                          </TableActionIcon>
                        </TableActionGroup>
                      </div>
                    </div>
                  )
                })
              )}
            </div>

            <div className="hidden lg:block">
              <Table containerClassName="max-w-full" className="w-max min-w-full">
                <TableHeader>
                  <TableRow className="border-b bg-slate-50 hover:bg-slate-50">
                    <TableHead className="w-14 px-2 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Sr. No
                    </TableHead>
                    <TableHead className="min-w-[9rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Case No
                    </TableHead>
                    {locations.enabled && (
                      <TableHead className="min-w-[7.5rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                        Location
                      </TableHead>
                    )}
                    <TableHead className="min-w-[7.5rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Detention Date
                    </TableHead>
                    <TableHead className="min-w-[10rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Place
                    </TableHead>
                    <TableHead className="min-w-[4.5rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Days
                    </TableHead>
                    <TableHead className="min-w-[7rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Assessment
                    </TableHead>
                    <TableHead className="min-w-[8rem] px-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Window
                    </TableHead>
                    <TableHead className="sticky right-0 z-20 min-w-[5rem] bg-slate-50 px-2 text-right text-[11px] font-semibold uppercase tracking-wide text-slate-500 shadow-[-8px_0_8px_-8px_rgba(15,23,42,0.18)]">
                      Actions
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={9} className="py-14 text-center text-sm text-muted-foreground">
                        Loading detention memos…
                      </TableCell>
                    </TableRow>
                  ) : pageRows.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={9} className="py-14 text-center text-sm text-muted-foreground">
                        No detention memos found.
                      </TableCell>
                    </TableRow>
                  ) : (
                    pageRows.map((row, index) => {
                      const days = daysSinceDetention(row.dateTimeDetention)
                      const overdue = isOverdue(row.dateTimeDetention)
                      const hasAssessment = assessedMemoIds.has(row.id)
                      return (
                        <TableRow
                          key={row.id}
                          className={cn(
                            "group border-b border-slate-100 transition-colors hover:bg-sky-50/40",
                            overdue && "bg-amber-50/70 hover:bg-amber-50"
                          )}
                        >
                          <TableCell className="px-2 py-2 text-center text-xs font-medium tabular-nums text-foreground">
                            {pageSerial(index)}
                          </TableCell>
                          <TableCell className="max-w-[12rem] overflow-hidden px-2 py-2">
                            <p className="max-w-[11rem] truncate font-mono text-[12px] font-semibold text-black" title={row.caseNo}>
                              {row.caseNo || "—"}
                            </p>
                            <p className="max-w-[11rem] truncate font-mono text-[10px] text-muted-foreground" title={row.referenceNumber || ""}>
                              {row.referenceNumber || "—"}
                            </p>
                          </TableCell>
                          {locations.enabled && (
                            <TableCell className="max-w-[8rem] overflow-hidden px-2 py-2 text-xs">
                              <span className="block max-w-[7.5rem] truncate" title={locations.labelFor(row.locationCode)}>
                                {locations.labelFor(row.locationCode)}
                              </span>
                            </TableCell>
                          )}
                          <TableCell className="whitespace-nowrap px-2 py-2 text-xs tabular-nums text-slate-700">
                            {row.dateTimeDetention?.slice(0, 10) || "—"}
                          </TableCell>
                          <TableCell className="max-w-[14rem] overflow-hidden px-2 py-2">
                            <span
                              className="block max-w-[13rem] truncate text-xs text-slate-700"
                              title={row.placeOfDetention || ""}
                            >
                              {shortPlace(row.placeOfDetention)}
                            </span>
                          </TableCell>
                          <TableCell
                            className={cn(
                              "px-2 py-2 text-xs tabular-nums",
                              overdue ? "font-semibold text-amber-800" : "text-slate-700"
                            )}
                          >
                            {days ?? "—"}
                          </TableCell>
                          <TableCell className="px-2 py-2">
                            {hasAssessment ? (
                              <Badge className="rounded-md border-0 bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-800 hover:bg-emerald-100">
                                Done
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                                Pending
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell className="px-2 py-2">
                            {overdue ? (
                              <Badge className="gap-1 rounded-md border-0 bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800 hover:bg-amber-100">
                                <AlertTriangle className="h-3 w-3" />
                                Overdue
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                                Within window
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell
                            className={cn(
                              "sticky right-0 z-10 whitespace-nowrap px-1 py-1.5 text-right shadow-[-8px_0_8px_-8px_rgba(15,23,42,0.18)]",
                              overdue ? "bg-[#fff8ee] group-hover:bg-amber-50" : "bg-white group-hover:bg-[#f3f8fd]"
                            )}
                          >
                            <TableActionGroup className="w-max max-w-none flex-nowrap gap-0">
                              <TableActionIcon label="View memo" to={getDetentionMemoDetailPath(row.id)}>
                                <Eye className="h-4 w-4" />
                              </TableActionIcon>
                            </TableActionGroup>
                          </TableCell>
                        </TableRow>
                      )
                    })
                  )}
                </TableBody>
              </Table>
            </div>

            {filtered.length > 0 ? (
              <div className="flex flex-col gap-3 border-t bg-slate-50/40 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-muted-foreground">Rows per page</span>
                  <Select
                    value={pageSize.toString()}
                    onValueChange={(value) => {
                      setPageSize(Number(value))
                      setPage(1)
                    }}
                  >
                    <SelectTrigger className="h-8 w-[72px] bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PAGE_SIZE_OPTIONS.map((size) => (
                        <SelectItem key={size} value={size.toString()}>
                          {size}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filtered.length)} of {filtered.length}
                  </span>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-1 lg:justify-end">
                  <Button variant="outline" size="sm" onClick={() => setPage(1)} disabled={page === 1} className="h-8 w-8 bg-white p-0">
                    <ChevronsLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                    disabled={page === 1}
                    className="h-8 w-8 bg-white p-0"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="px-2 text-xs tabular-nums text-muted-foreground">
                    {page} / {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
                    disabled={page === totalPages}
                    className="h-8 w-8 bg-white p-0"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(totalPages)}
                    disabled={page === totalPages}
                    className="h-8 w-8 bg-white p-0"
                  >
                    <ChevronsRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
      <PdfExportHost hostRef={pdf.hostRef}>
        {pdf.items?.map((row) => (
          <DetentionMemoReportPrint key={row.id} row={row} embedded />
        ))}
      </PdfExportHost>
    </ModulePageLayout>
  )
}
