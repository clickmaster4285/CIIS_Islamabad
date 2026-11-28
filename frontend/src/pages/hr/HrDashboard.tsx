import { useCallback, useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  CalendarClock,
  Check,
  DollarSign,
  Loader2,
  RefreshCw,
  ScanFace,
  Smartphone,
  UserCheck,
  UserPlus,
  Users,
  X,
} from "lucide-react"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { ModulePageLayout } from "@/components/dashboard/module-page-layout"
import { SourceLocationSelect } from "@/components/sync/source-location-select"
import { AttendanceFaceBadge } from "@/components/hr/attendance-face-badge"
import { StaffAvatar } from "@/components/hr/staff-avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { useToast } from "@/hooks/use-toast"
import { fetchAttendance, type AttendanceRecord } from "@/lib/attendance-api"
import {
  approveLeaveRequest,
  fetchLeaveRequests,
  rejectLeaveRequest,
  type LeaveRequestRecord,
} from "@/lib/leave-api"
import { fetchPayrollRuns, type PayrollRunRecord } from "@/lib/payroll-api"
import { REALTIME_INVALIDATE_EVENT } from "@/lib/realtime-socket"
import { recognitionApi } from "@/lib/recognition-api"
import { ALL_SOURCE_LOCATIONS, useSourceLocations } from "@/lib/source-location"
import { fetchEmployeesDirectory, type StaffRecord } from "@/lib/staff-api"
import { ROUTES, getEmployeeDetailPath } from "@/routes/config"

const attendanceChartConfig = {
  onTime: { label: "On time", color: "#155DFC" },
  late: { label: "Late", color: "#F59E0B" },
} satisfies ChartConfig

function localIsoDate(d: Date): string {
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 10)
}

function formatShortDate(isoDate: string): string {
  try {
    return new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
    })
  } catch {
    return isoDate
  }
}

function formatTime(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return "—"
  }
}

function formatPkr(value: string | number | null | undefined): string {
  if (value == null) return "—"
  const n = typeof value === "string" ? parseFloat(value) : value
  if (Number.isNaN(n)) return "—"
  return `PKR ${new Intl.NumberFormat("en-PK", { maximumFractionDigits: 0 }).format(n)}`
}

function runTotalsForLocation(
  run: PayrollRunRecord | undefined,
  selected: string,
  matches: (code: string | null | undefined, selected: string) => boolean
): { count: number; gross: string | number | null; net: string | number | null } {
  if (!run) return { count: 0, gross: null, net: null }
  if (selected === ALL_SOURCE_LOCATIONS) {
    return { count: run.employee_count, gross: run.total_gross, net: run.total_net }
  }
  const entries = (run.entries ?? []).filter((e) => matches(e.location_code, selected))
  const sum = (key: "gross_salary" | "net_salary") =>
    entries.reduce((total, e) => total + (parseFloat(e[key]) || 0), 0)
  return { count: entries.length, gross: sum("gross_salary"), net: sum("net_salary") }
}

function attendancePersonId(r: AttendanceRecord): string {
  if (r.staff != null) return `staff:${r.staff}`
  if (r.user != null) return `user:${r.user}`
  if (r.staff_name) return `name:${r.staff_name}`
  return `rec:${r.id}`
}

export default function HrDashboardPage() {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const [sourceLocation, setSourceLocation] = useState(ALL_SOURCE_LOCATIONS)
  const [leaveBusyId, setLeaveBusyId] = useState<number | null>(null)
  const locations = useSourceLocations()

  const staffQuery = useQuery({
    queryKey: ["hr", "dashboard", "staff"],
    queryFn: fetchEmployeesDirectory,
  })
  const attendanceQuery = useQuery({
    queryKey: ["hr", "dashboard", "attendance"],
    queryFn: () => fetchAttendance(),
  })
  const leaveQuery = useQuery({
    queryKey: ["hr", "dashboard", "leave"],
    queryFn: () => fetchLeaveRequests(),
  })
  const payrollQuery = useQuery({
    queryKey: ["hr", "dashboard", "payroll"],
    queryFn: fetchPayrollRuns,
  })
  const samplesQuery = useQuery({
    queryKey: ["hr", "dashboard", "face-samples"],
    queryFn: () => recognitionApi.faceSamples("pending"),
    retry: false,
  })

  const refreshAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["hr", "dashboard"] })
  }, [queryClient])

  useEffect(() => {
    const onRealtime = (e: Event) => {
      const domains = (e as CustomEvent<{ domains?: string[] }>).detail?.domains || []
      if (domains.some((d) => ["attendance", "hr", "recognition"].includes(d))) {
        refreshAll()
      }
    }
    window.addEventListener(REALTIME_INVALIDATE_EVENT, onRealtime)
    return () => window.removeEventListener(REALTIME_INVALIDATE_EVENT, onRealtime)
  }, [refreshAll])

  const matchesLocation = locations.matches
  const staffList = useMemo(
    () => (staffQuery.data ?? []).filter((s) => matchesLocation(s.location_code, sourceLocation)),
    [staffQuery.data, matchesLocation, sourceLocation]
  )
  const attendanceList = useMemo(
    () => (attendanceQuery.data ?? []).filter((r) => matchesLocation(r.location_code, sourceLocation)),
    [attendanceQuery.data, matchesLocation, sourceLocation]
  )
  const leaveList = useMemo(
    () => (leaveQuery.data ?? []).filter((r) => matchesLocation(r.location_code, sourceLocation)),
    [leaveQuery.data, matchesLocation, sourceLocation]
  )
  const payrollRuns = payrollQuery.data ?? []

  const todayStr = localIsoDate(new Date())

  // Today's attendance metrics
  const todayAttendance = useMemo(
    () => attendanceList.filter((r) => r.date === todayStr),
    [attendanceList, todayStr]
  )

  const attendanceSummary = useMemo(() => {
    const presentKeys = new Set<string>()
    const lateKeys = new Set<string>()
    const checkedOutKeys = new Set<string>()
    for (const r of todayAttendance) {
      const key = attendancePersonId(r)
      if (r.check_in || r.status === "present" || r.status === "late" || r.status === "half_day") {
        presentKeys.add(key)
      }
      if (r.status === "late") lateKeys.add(key)
      if (r.check_out) checkedOutKeys.add(key)
    }
    const presentCount = presentKeys.size
    const absentCount = Math.max(staffList.length - presentCount, 0)
    const attendanceRate =
      staffList.length > 0 ? Math.round((presentCount / staffList.length) * 100) : 0
    return {
      presentCount,
      lateCount: lateKeys.size,
      checkedOutCount: checkedOutKeys.size,
      absentCount,
      attendanceRate,
    }
  }, [todayAttendance, staffList.length])

  // 7-day trend chart data
  const weeklyTrend = useMemo(() => {
    const days: { date: string; label: string; onTime: number; late: number }[] = []
    for (let offset = 6; offset >= 0; offset--) {
      const d = new Date()
      d.setDate(d.getDate() - offset)
      const iso = localIsoDate(d)
      days.push({
        date: iso,
        label: d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric" }),
        onTime: 0,
        late: 0,
      })
    }
    const byDate = new Map(days.map((d) => [d.date, d]))
    for (const r of attendanceList) {
      const bucket = byDate.get(r.date)
      if (!bucket) continue
      if (r.status === "late") {
        bucket.late += 1
      } else if (r.check_in || r.status === "present" || r.status === "half_day") {
        bucket.onTime += 1
      }
    }
    return days
  }, [attendanceList])

  // Workforce breakdown: departments, face enrollment, mobile adoption
  const workforceStats = useMemo(() => {
    const deptMap = new Map<
      string,
      { department: string; headcount: number; enrolled: number; presentToday: number }
    >()
    const presentStaffIds = new Set(
      todayAttendance.filter((r) => r.staff != null).map((r) => r.staff as number)
    )

    let enrolledFaces = 0
    let pendingFaces = 0
    let notEnrolledFaces = 0
    let mobileInstalled = 0
    let mobileLoggedIn = 0
    let permanent = 0
    let contract = 0

    for (const emp of staffList) {
      const dept = (emp.department || "Unassigned").trim() || "Unassigned"
      const row = deptMap.get(dept) ?? {
        department: dept,
        headcount: 0,
        enrolled: 0,
        presentToday: 0,
      }
      row.headcount += 1
      const faceStatus = emp.attendance_face?.status ?? "not_enrolled"
      if (faceStatus === "enrolled") {
        enrolledFaces += 1
        row.enrolled += 1
      } else if (faceStatus === "needs_photos" || faceStatus === "not_trained") {
        pendingFaces += 1
      } else {
        notEnrolledFaces += 1
      }
      if (presentStaffIds.has(emp.id)) {
        row.presentToday += 1
      }
      if (emp.mobile_app_installed) mobileInstalled += 1
      if (emp.mobile_logged_in) mobileLoggedIn += 1
      const empType = (emp.employment_type || "").toLowerCase()
      if (empType.includes("permanent")) permanent += 1
      else if (empType.includes("contract")) contract += 1
      deptMap.set(dept, row)
    }

    const departments = Array.from(deptMap.values()).sort((a, b) => b.headcount - a.headcount)
    return {
      departments,
      enrolledFaces,
      pendingFaces,
      notEnrolledFaces,
      mobileInstalled,
      mobileLoggedIn,
      permanent,
      contract,
    }
  }, [staffList, todayAttendance])

  // Leave metrics
  const pendingLeaves = useMemo(
    () => leaveList.filter((r) => r.status === "PENDING"),
    [leaveList]
  )
  const onLeaveToday = useMemo(
    () =>
      leaveList.filter(
        (r) => r.status === "APPROVED" && r.from_date <= todayStr && r.to_date >= todayStr
      ),
    [leaveList, todayStr]
  )

  // Latest payroll run
  const latestRun = payrollRuns[0]
  const latestRunTotals = useMemo(
    () => runTotalsForLocation(latestRun, sourceLocation, matchesLocation),
    [latestRun, sourceLocation, matchesLocation]
  )

  // Recent attendance activity (newest first)
  const recentAttendance = useMemo(
    () =>
      [...attendanceList]
        .sort((a, b) => {
          const ta = a.check_in || a.created_at || a.date
          const tb = b.check_in || b.created_at || b.date
          return tb.localeCompare(ta)
        })
        .slice(0, 6),
    [attendanceList]
  )

  // Recently added staff
  const recentStaff = useMemo(
    () =>
      [...staffList]
        .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "") || b.id - a.id)
        .slice(0, 5),
    [staffList]
  )

  const handleLeaveDecision = async (req: LeaveRequestRecord, action: "approve" | "reject") => {
    setLeaveBusyId(req.id)
    try {
      if (action === "approve") {
        await approveLeaveRequest(req.id)
        toast({ title: "Leave approved", description: `${req.staff_name} (${req.days}d)` })
      } else {
        await rejectLeaveRequest(req.id)
        toast({ title: "Leave rejected", description: `${req.staff_name}` })
      }
      void queryClient.invalidateQueries({ queryKey: ["hr", "dashboard", "leave"] })
    } catch (err) {
      toast({
        title: `Could not ${action} leave`,
        description: err instanceof Error ? err.message : "Request failed",
        variant: "destructive",
      })
    } finally {
      setLeaveBusyId(null)
    }
  }

  const isFetching =
    staffQuery.isFetching ||
    attendanceQuery.isFetching ||
    leaveQuery.isFetching ||
    payrollQuery.isFetching

  const pendingSamplesCount = samplesQuery.data?.pending_count ?? 0

  return (
    <ModulePageLayout
      title="HR Dashboard"
      description="Workforce overview, daily attendance pulse, leave approvals, face enrollment, and payroll status."
      breadcrumbs={[{ label: "Human Resource" }, { label: "HR Dashboard" }]}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {locations.enabled && (
            <SourceLocationSelect
              value={sourceLocation}
              onChange={setSourceLocation}
              options={locations.options}
            />
          )}
          <Button variant="outline" size="sm" onClick={refreshAll} disabled={isFetching}>
            {isFetching ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh
          </Button>
          <Button className="bg-[#3b82f6] text-white hover:bg-[#2563eb]" size="sm" asChild>
            <Link to={ROUTES.ADD_STAFF}>
              <UserPlus className="mr-2 h-4 w-4" />
              Add Staff
            </Link>
          </Button>
        </div>
      }
    >
      <div className="grid gap-6">
        {/* Top KPI Cards */}
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Staff</CardTitle>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50">
                <Users className="h-4 w-4 text-[#155DFC]" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {staffQuery.isLoading ? "—" : staffList.length}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {workforceStats.permanent} permanent · {workforceStats.contract} contract
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Present Today</CardTitle>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-50">
                <UserCheck className="h-4 w-4 text-emerald-600" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold">
                  {attendanceQuery.isLoading ? "—" : attendanceSummary.presentCount}
                </span>
                {!attendanceQuery.isLoading && staffList.length > 0 && (
                  <span className="text-xs font-medium text-emerald-700">
                    {attendanceSummary.attendanceRate}% of staff
                  </span>
                )}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {attendanceSummary.lateCount} late · {attendanceSummary.absentCount} absent ·{" "}
                {attendanceSummary.checkedOutCount} out
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Leave &amp; Approvals
              </CardTitle>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-50">
                <CalendarClock className="h-4 w-4 text-amber-600" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold">
                  {leaveQuery.isLoading ? "—" : pendingLeaves.length}
                </span>
                <span className="text-xs text-muted-foreground">pending requests</span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {onLeaveToday.length} on approved leave today
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Latest Payroll</CardTitle>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-50">
                <DollarSign className="h-4 w-4 text-violet-600" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {payrollQuery.isLoading ? "—" : formatPkr(latestRunTotals.gross)}
              </div>
              <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                <span>
                  {latestRun
                    ? `${latestRun.period_label} · ${latestRunTotals.count} staff`
                    : "No payroll run yet"}
                </span>
                {latestRun && (
                  <Badge
                    variant={latestRun.status === "PROCESSED" ? "default" : "secondary"}
                    className="h-4 px-1.5 text-[10px]"
                  >
                    {latestRun.status}
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Attendance 7-day trend + Department breakdown */}
        <div className="grid gap-6 lg:grid-cols-7">
          <Card className="lg:col-span-4">
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="text-base">7-Day Attendance Trend</CardTitle>
                <CardDescription>Daily check-ins split by on-time vs late arrivals</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to={ROUTES.ATTENDANCE}>Open Attendance</Link>
              </Button>
            </CardHeader>
            <CardContent>
              <ChartContainer config={attendanceChartConfig} className="h-[250px] w-full">
                <BarChart data={weeklyTrend} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={12} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} fontSize={12} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar
                    dataKey="onTime"
                    stackId="att"
                    fill="var(--color-onTime)"
                    radius={[0, 0, 4, 4]}
                  />
                  <Bar
                    dataKey="late"
                    stackId="att"
                    fill="var(--color-late)"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card className="lg:col-span-3">
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="text-base">Department Breakdown</CardTitle>
                <CardDescription>Headcount and today&apos;s presence by department</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to={ROUTES.EMPLOYEES}>Directory</Link>
              </Button>
            </CardHeader>
            <CardContent>
              {staffQuery.isLoading ? (
                <p className="py-6 text-sm text-muted-foreground">Loading departments…</p>
              ) : workforceStats.departments.length === 0 ? (
                <p className="py-6 text-sm text-muted-foreground">No staff records found.</p>
              ) : (
                <div className="space-y-3 max-h-[250px] overflow-y-auto pr-1">
                  {workforceStats.departments.slice(0, 7).map((dept) => {
                    const pct = staffList.length
                      ? Math.round((dept.headcount / staffList.length) * 100)
                      : 0
                    return (
                      <div key={dept.department} className="space-y-1.5">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium truncate">{dept.department}</span>
                          <span className="text-xs text-muted-foreground shrink-0">
                            {dept.presentToday}/{dept.headcount} present · {dept.enrolled} face ready
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-[#155DFC]"
                            style={{ width: `${Math.max(pct, 4)}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Pending Leave Approvals + Recent Attendance */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="text-base">
                  Pending Leave Requests{" "}
                  {pendingLeaves.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {pendingLeaves.length}
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>Requests awaiting HR approval</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to={ROUTES.LEAVE_MANAGEMENT}>Manage Leaves</Link>
              </Button>
            </CardHeader>
            <CardContent>
              {leaveQuery.isLoading ? (
                <p className="py-6 text-sm text-muted-foreground">Loading leave requests…</p>
              ) : pendingLeaves.length === 0 ? (
                <p className="py-6 text-sm text-muted-foreground">
                  No pending leave requests right now.
                </p>
              ) : (
                <div className="overflow-x-auto rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Employee</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Dates</TableHead>
                        <TableHead>Days</TableHead>
                        <TableHead className="text-right">Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pendingLeaves.slice(0, 5).map((req) => {
                        const busy = leaveBusyId === req.id
                        return (
                          <TableRow key={req.id}>
                            <TableCell className="font-medium">{req.staff_name}</TableCell>
                            <TableCell>{req.leave_type_name}</TableCell>
                            <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                              {formatShortDate(req.from_date)} – {formatShortDate(req.to_date)}
                            </TableCell>
                            <TableCell>{req.days}</TableCell>
                            <TableCell className="text-right">
                              <div className="inline-flex items-center gap-1">
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  className="h-7 w-7 text-emerald-600 hover:text-emerald-700"
                                  title="Approve"
                                  disabled={busy}
                                  onClick={() => void handleLeaveDecision(req, "approve")}
                                >
                                  <Check className="h-4 w-4" />
                                </Button>
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  className="h-7 w-7 text-red-600 hover:text-red-700"
                                  title="Reject"
                                  disabled={busy}
                                  onClick={() => void handleLeaveDecision(req, "reject")}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="text-base">Recent Attendance Activity</CardTitle>
                <CardDescription>Latest check-ins and check-outs across cameras and manual entry</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to={ROUTES.ATTENDANCE}>View All</Link>
              </Button>
            </CardHeader>
            <CardContent>
              {attendanceQuery.isLoading ? (
                <p className="py-6 text-sm text-muted-foreground">Loading attendance…</p>
              ) : recentAttendance.length === 0 ? (
                <p className="py-6 text-sm text-muted-foreground">No attendance records yet.</p>
              ) : (
                <div className="space-y-2.5">
                  {recentAttendance.map((row) => (
                    <div
                      key={row.id}
                      className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm"
                    >
                      <div className="min-w-0">
                        <div className="font-medium truncate">
                          {row.staff_name || row.username || "Unknown staff"}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {row.date} · in {formatTime(row.check_in)}
                          {row.check_out ? ` · out ${formatTime(row.check_out)}` : ""}
                          {row.source ? ` · ${row.source}` : ""}
                        </div>
                      </div>
                      <Badge
                        variant={
                          row.status === "late"
                            ? "secondary"
                            : row.status === "absent"
                              ? "outline"
                              : "default"
                        }
                        className="shrink-0 capitalize"
                      >
                        {row.status || (row.check_in ? "present" : "absent")}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Face Enrollment & Mobile Readiness + Recent Staff */}
        <div className="grid gap-6 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Biometric &amp; Mobile Readiness</CardTitle>
              <CardDescription>InsightFace attendance enrollment &amp; CIIS PWA status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded-lg border bg-emerald-50/60 p-2.5">
                  <div className="text-lg font-bold text-emerald-700">
                    {workforceStats.enrolledFaces}
                  </div>
                  <div className="text-[11px] text-emerald-800">Enrolled</div>
                </div>
                <div className="rounded-lg border bg-amber-50/60 p-2.5">
                  <div className="text-lg font-bold text-amber-700">
                    {workforceStats.pendingFaces}
                  </div>
                  <div className="text-[11px] text-amber-800">Needs sync</div>
                </div>
                <div className="rounded-lg border bg-red-50/60 p-2.5">
                  <div className="text-lg font-bold text-red-700">
                    {workforceStats.notEnrolledFaces}
                  </div>
                  <div className="text-[11px] text-red-800">Not enrolled</div>
                </div>
              </div>

              <div className="space-y-2 border-t pt-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <ScanFace className="h-4 w-4 text-[#155DFC]" />
                    CCTV face samples to review
                  </span>
                  <Link
                    to={ROUTES.ATTENDANCE}
                    className="font-semibold text-[#155DFC] hover:underline"
                  >
                    {pendingSamplesCount} pending
                  </Link>
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Smartphone className="h-4 w-4 text-muted-foreground" />
                    Mobile app installed
                  </span>
                  <span className="font-medium">
                    {workforceStats.mobileInstalled} / {staffList.length}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground pl-6">Active mobile logins</span>
                  <span className="font-medium">{workforceStats.mobileLoggedIn}</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <Button variant="outline" size="sm" asChild>
                  <Link to={ROUTES.PAYROLL}>Payroll</Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link to={ROUTES.RECRUITMENT}>Recruitment</Link>
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle className="text-base">Recent Staff Records</CardTitle>
                <CardDescription>Newest employees and their attendance face status</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to={ROUTES.EMPLOYEES}>All Employees</Link>
              </Button>
            </CardHeader>
            <CardContent>
              {staffQuery.isLoading ? (
                <p className="py-6 text-sm text-muted-foreground">Loading staff…</p>
              ) : recentStaff.length === 0 ? (
                <p className="py-6 text-sm text-muted-foreground">
                  No staff records yet. Click &quot;Add Staff&quot; to onboard an employee.
                </p>
              ) : (
                <div className="divide-y rounded-md border">
                  {recentStaff.map((emp: StaffRecord) => (
                    <Link
                      key={emp.id}
                      to={getEmployeeDetailPath(emp.id)}
                      className="flex items-center justify-between gap-3 px-3 py-2.5 hover:bg-muted/40 transition-colors"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <StaffAvatar
                          fullName={emp.full_name}
                          profileImage={emp.profile_image}
                        />
                        <div className="min-w-0">
                          <div className="font-medium text-sm truncate">{emp.full_name}</div>
                          <div className="text-xs text-muted-foreground truncate">
                            {emp.designation || "—"} · {emp.department || "Unassigned"}
                            {emp.personal_number ? ` · PN ${emp.personal_number}` : ""}
                          </div>
                        </div>
                      </div>
                      <AttendanceFaceBadge face={emp.attendance_face} />
                    </Link>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </ModulePageLayout>
  )
}
