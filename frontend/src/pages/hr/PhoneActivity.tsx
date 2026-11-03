import { useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { ModulePageLayout } from "@/components/dashboard/module-page-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  fetchAppUsage,
  fetchPhoneCalls,
  fetchRecordings,
  fetchRecordingAudioUrl,
  type CallRecordingRecord,
  type PhoneActivityFilters,
} from "@/lib/mobile-admin-api"

function formatDuration(seconds: number | null | undefined) {
  if (!seconds) return "0s"
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${s}s`
  return `${s}s`
}

function formatTime(iso: string | null) {
  if (!iso) return "—"
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString()
}

function directionTone(direction: string) {
  if (direction === "missed") return "bg-red-600"
  if (direction === "inbound") return "bg-sky-600"
  if (direction === "outbound") return "bg-emerald-600"
  return "bg-slate-500"
}

function staffLabel(row: { name: string | null; username: string | null }) {
  return row.name || row.username || "Unknown"
}

/** App launcher icon uploaded from the phone, with a colored first-letter tile as fallback. */
function AppIcon({ icon, label }: { icon?: string; label: string }) {
  if (icon) {
    return <img src={icon} alt="" className="h-7 w-7 rounded-md object-cover shrink-0" />
  }
  const letter = (label || "?").trim().charAt(0).toUpperCase()
  return (
    <div className="h-7 w-7 rounded-md bg-slate-600 text-white flex items-center justify-center text-xs font-bold shrink-0">
      {letter}
    </div>
  )
}

function recordingTone(status: string) {
  if (status === "available") return "bg-emerald-600"
  if (status === "pending") return "bg-amber-500"
  if (status === "failed") return "bg-red-600"
  return "bg-slate-500"
}

/**
 * Plays one recording. The audio is fetched as a blob through the audited, signed stream
 * endpoint only when the manager presses Play, so no recording streams until it's asked for.
 */
function RecordingPlayer({ rec }: { rec: CallRecordingRecord }) {
  const [url, setUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const urlRef = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [])

  async function play() {
    if (loading || url) return
    setLoading(true)
    setError("")
    try {
      const objectUrl = await fetchRecordingAudioUrl(rec.id)
      urlRef.current = objectUrl
      setUrl(objectUrl)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load recording")
    } finally {
      setLoading(false)
    }
  }

  if (!rec.can_play) {
    return <span className="text-xs text-muted-foreground">No audio</span>
  }
  if (url) {
    // eslint-disable-next-line jsx-a11y/media-has-caption
    return <audio src={url} controls autoPlay className="h-9 w-64 max-w-full" />
  }
  return (
    <div className="flex items-center gap-2">
      <Button type="button" size="sm" variant="outline" onClick={play} disabled={loading}>
        {loading ? "Loading…" : "▶ Listen"}
      </Button>
      {error ? <span className="text-xs text-destructive">{error}</span> : null}
    </div>
  )
}

export default function PhoneActivityPage() {
  const [date, setDate] = useState("")
  const [username, setUsername] = useState("")
  const [filters, setFilters] = useState<PhoneActivityFilters>({})

  const calls = useQuery({
    queryKey: ["phone-calls", filters],
    queryFn: () => fetchPhoneCalls(filters),
  })
  const usage = useQuery({
    queryKey: ["app-usage", filters],
    queryFn: () => fetchAppUsage(filters),
  })
  const recordings = useQuery({
    queryKey: ["recordings", filters],
    queryFn: () => fetchRecordings(filters),
  })

  const callRows = calls.data ?? []
  const usageRows = usage.data ?? []
  const recordingRows = recordings.data ?? []
  const totalCallSeconds = callRows.reduce((sum, c) => sum + (c.duration_seconds || 0), 0)
  const totalUsageSeconds = usageRows.reduce((sum, r) => sum + (r.foreground_seconds || 0), 0)

  const apply = () => setFilters({ date: date || undefined, username: username.trim() || undefined })
  const clear = () => {
    setDate("")
    setUsername("")
    setFilters({})
  }

  return (
    <ModulePageLayout
      title="Phone Activity"
      description="Call history and app usage uploaded by the CIIS mobile app from consenting staff."
      breadcrumbs={[{ label: "HR" }, { label: "Phone Activity" }]}
    >
      <Card className="mb-6">
        <CardContent className="pt-6">
          <form
            className="flex flex-wrap items-end gap-4"
            onSubmit={(e) => {
              e.preventDefault()
              apply()
            }}
          >
            <div className="space-y-1">
              <Label htmlFor="pa-date">Date</Label>
              <Input id="pa-date" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="pa-user">Username</Label>
              <Input
                id="pa-user"
                placeholder="All staff"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <Button type="submit">Apply</Button>
            <Button type="button" variant="outline" onClick={clear}>
              Clear
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                void calls.refetch()
                void usage.refetch()
                void recordings.refetch()
              }}
            >
              Refresh
            </Button>
          </form>
        </CardContent>
      </Card>

      <Tabs defaultValue="calls">
        <TabsList>
          <TabsTrigger value="calls">Call logs ({callRows.length})</TabsTrigger>
          <TabsTrigger value="usage">App usage ({usageRows.length})</TabsTrigger>
          <TabsTrigger value="recordings">Recordings ({recordingRows.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="calls">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {callRows.length} calls · {formatDuration(totalCallSeconds)} total
              </CardTitle>
            </CardHeader>
            <CardContent>
              {calls.error ? <p className="text-sm text-destructive mb-4">{String(calls.error)}</p> : null}
              {calls.isLoading ? (
                <p className="text-sm text-muted-foreground">Loading call logs…</p>
              ) : callRows.length === 0 ? (
                <p className="text-sm text-muted-foreground">No call logs for these filters.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Staff</TableHead>
                      <TableHead>Direction</TableHead>
                      <TableHead>Number</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead>Duration</TableHead>
                      <TableHead>Source</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {callRows.map((c) => (
                      <TableRow key={c.id}>
                        <TableCell>
                          <div className="font-medium">{staffLabel(c)}</div>
                          {c.name && c.username ? (
                            <div className="text-xs text-muted-foreground">{c.username}</div>
                          ) : null}
                        </TableCell>
                        <TableCell>
                          <Badge className={directionTone(c.direction)}>{c.direction}</Badge>
                        </TableCell>
                        <TableCell className="font-mono">{c.number || c.masked_number || "—"}</TableCell>
                        <TableCell>{formatTime(c.started_at)}</TableCell>
                        <TableCell>{formatDuration(c.duration_seconds)}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {c.source_label || c.source}
                          {c.sim_slot ? ` · SIM ${c.sim_slot}` : ""}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="usage">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {usageRows.length} entries · {formatDuration(totalUsageSeconds)} total screen time
              </CardTitle>
            </CardHeader>
            <CardContent>
              {usage.error ? <p className="text-sm text-destructive mb-4">{String(usage.error)}</p> : null}
              {usage.isLoading ? (
                <p className="text-sm text-muted-foreground">Loading app usage…</p>
              ) : usageRows.length === 0 ? (
                <p className="text-sm text-muted-foreground">No app usage for these filters.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Staff</TableHead>
                      <TableHead>App</TableHead>
                      <TableHead>Time in use</TableHead>
                      <TableHead>Launches</TableHead>
                      <TableHead>Period</TableHead>
                      <TableHead>Last used</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {usageRows.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell>
                          <div className="font-medium">{staffLabel(r)}</div>
                          {r.name && r.username ? (
                            <div className="text-xs text-muted-foreground">{r.username}</div>
                          ) : null}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <AppIcon icon={r.icon} label={r.app_label} />
                            <div>
                              <div className="font-medium">{r.app_label}</div>
                              <div className="text-xs text-muted-foreground font-mono">{r.package_name}</div>
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>{formatDuration(r.foreground_seconds)}</TableCell>
                        <TableCell>{r.launch_count}</TableCell>
                        <TableCell className="text-xs">
                          {formatTime(r.window_start)} – {formatTime(r.window_end)}
                        </TableCell>
                        <TableCell>{formatTime(r.last_used_at)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recordings">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{recordingRows.length} recordings</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-xs text-muted-foreground">
                Call audio is captured server-side by the company telephony route (PBX), never by the
                handset. Pressing Listen streams the audio through a short-lived signed link, and every
                playback is written to the monitoring audit log.
              </p>
              {recordings.error ? (
                <p className="text-sm text-destructive mb-4">{String(recordings.error)}</p>
              ) : null}
              {recordings.isLoading ? (
                <p className="text-sm text-muted-foreground">Loading recordings…</p>
              ) : recordingRows.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No recordings for these filters. Recordings appear here once calls are routed through
                  the recording-enabled telephony route.
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Recorded</TableHead>
                      <TableHead>Source</TableHead>
                      <TableHead>Duration</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Play</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recordingRows.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell>{formatTime(r.recorded_at)}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {r.source_label || r.source}
                        </TableCell>
                        <TableCell>{formatDuration(r.duration_seconds)}</TableCell>
                        <TableCell>
                          <Badge className={recordingTone(r.status)}>{r.status}</Badge>
                        </TableCell>
                        <TableCell>
                          <RecordingPlayer rec={r} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </ModulePageLayout>
  )
}
