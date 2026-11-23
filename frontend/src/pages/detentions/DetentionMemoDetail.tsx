import { useEffect, useState, type ReactNode } from "react"
import { useParams, Link, useSearchParams, useLocation } from "react-router-dom"
import { ArrowLeft, FileText, Package, Pencil, QrCode, Users, Paperclip, Camera } from "lucide-react"
import { ModulePageLayout } from "@/components/dashboard/module-page-layout"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { PrintMenu } from "@/components/seizure/print-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { getDetentionMemoEditPath, getDetentionMemoDetailPath, getDetentionMemoListPath, getDetentionMemoScanUrl, getDetentionMemoSectionCrumb } from "@/routes/config"
import DetentionMemoReportPrint from "@/components/detention/DetentionMemoReportPrint"
import DetentionMemoQRPrint from "@/components/detention/DetentionMemoQRPrint"
import { DestructionRecordsPanel } from "@/components/warehouse/destruction-records-panel"
import { WmsFlowOverviewPanel } from "@/components/warehouse/wms-flow-overview-panel"
import { MlCameraFeed } from "@/components/cameras/ml-camera-feed"
import {
  fetchDetentionMemoById,
  type DetentionMemoApiRecord,
  type DetentionMemoGoodsLineApi,
  type LocatedCameraApi,
} from "@/lib/detention-memo-api"
import { fetchCamera, cameraSourceLabel, type CameraRecord } from "@/lib/cameras-api"
import { GoodsLineText, goodsDetailCellClass, goodsHeadClass } from "@/components/goods/goods-line-text-field"
import { GoodsQrDisplay } from "@/components/goods/goods-qr-display"
import { cn } from "@/lib/utils"
import { canUserFullyEditSeizureDocs } from "@/lib/seizure-management-api"
import { getStoredUser } from "@/lib/auth"

type GoodsLineItem = DetentionMemoGoodsLineApi

function locatedCameraId(item: GoodsLineItem): number | null {
  const camId = item.locatedCameraId ?? item.locatedCamera?.id
  return camId == null ? null : Number(camId)
}

function cameraLabel(cam: LocatedCameraApi | null | undefined, _fallbackId?: number | null): string {
  if (cam) {
    const name =
      (cam.name || "").trim() ||
      (cam.displayLabel || "").trim() ||
      (cam.code || "").trim()
    if (name) {
      const zone = (cam.zone || "").trim()
      return zone ? `${name} · ${zone}` : name
    }
  }
  return "—"
}

type ViewCameraTarget = {
  cameraId: number
  label: string
  itemDescription?: string
  detectedAt?: string
  evidenceUrl?: string
}

function ViewLocatedCameraDialog({
  target,
  onClose,
}: {
  target: ViewCameraTarget | null
  onClose: () => void
}) {
  const [camera, setCamera] = useState<CameraRecord | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!target) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset/loading state for the fetch keyed on `target`
      setCamera(null)
      setError(null)
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    setCamera(null)
    fetchCamera(target.cameraId)
      .then((cam) => {
        if (!cancelled) setCamera(cam)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load camera")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [target])

  return (
    <Dialog open={!!target} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[92vh] w-[min(96vw,72rem)] gap-3 overflow-y-auto p-4 sm:max-w-6xl sm:p-6">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base sm:text-lg">
            <Camera className="h-4 w-4 shrink-0" />
            {target?.label || "Located camera"}
          </DialogTitle>
          <DialogDescription className="text-xs sm:text-sm">
            {target?.itemDescription
              ? `Live view for detained item: ${target.itemDescription}`
              : "Live camera feed for this detained goods line."}
            {target?.detectedAt ? ` · Detected ${target.detectedAt}` : ""}
          </DialogDescription>
        </DialogHeader>

        {target?.evidenceUrl ? (
          <div className="rounded-md border bg-muted/30 p-2">
            <p className="mb-1.5 text-xs text-muted-foreground">Evidence snapshot</p>
            <img
              src={target.evidenceUrl}
              alt="Detection evidence"
              className="max-h-48 w-full rounded object-contain bg-black"
            />
          </div>
        ) : null}

        <div className="min-h-[min(70vh,560px)] overflow-hidden rounded-lg border bg-black">
          {loading ? (
            <p className="p-8 text-center text-sm text-muted-foreground">Loading live feed…</p>
          ) : error ? (
            <p className="p-8 text-center text-sm text-destructive">{error}</p>
          ) : camera ? (
            <div className="space-y-2 p-2 sm:p-3">
              <p className="px-1 text-xs text-muted-foreground">
                {(camera.name || "").trim() || camera.code || "Camera"}
                {cameraSourceLabel(camera) ? ` · ${cameraSourceLabel(camera)}` : ""}
                {camera.zone ? ` · ${camera.zone}` : ""}
              </p>
              <MlCameraFeed
                camera={camera}
                className="min-h-[min(62vh,500px)] rounded-md"
                showBrandLogo
                showFullscreenButton
              />
            </div>
          ) : (
            <p className="p-8 text-center text-sm text-muted-foreground">No camera selected.</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

type SectionTone = "blue" | "teal" | "amber" | "violet" | "rose"

const SECTION_TONE: Record<
  SectionTone,
  { chip: string; head: string; border: string; rule: string }
> = {
  blue: {
    chip: "bg-[#D7EBFF] text-[#0B5CAD]",
    head: "bg-gradient-to-r from-[#E7F2FF] to-[#F8FBFF]",
    border: "border-[#C5DDF7]",
    rule: "border-[#D5E7FA]",
  },
  teal: {
    chip: "bg-[#D5F4EE] text-[#0C7A68]",
    head: "bg-gradient-to-r from-[#E5F8F4] to-[#F7FDFB]",
    border: "border-[#BFE8DF]",
    rule: "border-[#D3F0E9]",
  },
  amber: {
    chip: "bg-[#FFE8C7] text-[#A15C00]",
    head: "bg-gradient-to-r from-[#FFF3E0] to-[#FFFBF5]",
    border: "border-[#F5D7A8]",
    rule: "border-[#F8E4C4]",
  },
  violet: {
    chip: "bg-[#E6E0FF] text-[#5340B0]",
    head: "bg-gradient-to-r from-[#F1EDFF] to-[#FBFAFF]",
    border: "border-[#D9D0F7]",
    rule: "border-[#E6E0F8]",
  },
  rose: {
    chip: "bg-[#FFDDE4] text-[#B4234A]",
    head: "bg-gradient-to-r from-[#FFF0F3] to-[#FFF9FA]",
    border: "border-[#F6C9D3]",
    rule: "border-[#F8D7DF]",
  },
}

function FieldTile({ label, value }: { label: string; value: string | undefined }) {
  const text = (value ?? "").trim()
  return (
    <div className="min-w-0 rounded-lg border border-[#D7E6F6] bg-gradient-to-br from-white to-[#F4F9FF] px-3.5 py-2.5 shadow-[inset_3px_0_0_#3B8FE8]">
      <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#1D6BB5]">{label}</p>
      <p className="mt-1 break-words text-[13px] font-semibold leading-snug text-[#102C55] [overflow-wrap:anywhere]">
        {text || "—"}
      </p>
    </div>
  )
}

function Snapshot({
  label,
  value,
  className,
}: {
  label: string
  value: string
  className: string
}) {
  return (
    <div className={cn("min-w-0 rounded-xl border px-3.5 py-3 shadow-sm", className)}>
      <p className="text-[10px] font-semibold uppercase tracking-[0.08em] opacity-80">{label}</p>
      <p className="mt-1 truncate text-[18px] font-semibold leading-tight">{value || "—"}</p>
    </div>
  )
}

function FieldGrid({ children, columns = "sm:grid-cols-2" }: { children: ReactNode; columns?: string }) {
  return (
    <div className={cn("grid grid-cols-1 gap-2", columns)}>
      {children}
    </div>
  )
}

function Section({
  icon,
  title,
  action,
  children,
  className,
  tone = "blue",
}: {
  icon?: ReactNode
  title: string
  action?: ReactNode
  children: ReactNode
  className?: string
  tone?: SectionTone
}) {
  const paint = SECTION_TONE[tone]
  return (
    <section className={cn("min-w-0 overflow-hidden rounded-xl border bg-white shadow-sm", paint.border, className)}>
      <div className={cn("flex items-center justify-between gap-3 border-b px-4 py-3", paint.head, paint.rule)}>
        <h3 className="flex min-w-0 items-center gap-2.5 text-[13px] font-semibold text-[#102C55]">
          <span className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-lg", paint.chip)}>
            {icon ?? <span className="h-2 w-2 rounded-full bg-current" />}
          </span>
          <span className="truncate">{title}</span>
        </h3>
        {action}
      </div>
      <div className="bg-white p-4">{children}</div>
    </section>
  )
}

function StatusMark({ status }: { status?: string }) {
  const verified = status === "Verified"
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold shadow-sm",
        verified ? "bg-[#D9F5E8] text-[#0E7A45]" : "bg-[#FFE8C2] text-[#9A5B00]"
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", verified ? "bg-[#12A66A]" : "bg-[#F59E0B]")} />
      {status?.trim() || "Pending"}
    </span>
  )
}

function NoteBlock({ children, tone = "blue" }: { children: ReactNode; tone?: "blue" | "amber" }) {
  const paint =
    tone === "amber"
      ? "border-[#F5D7A8] bg-[#FFF8EE] text-[#5C3B12]"
      : "border-[#C5DDF7] bg-[#F3F8FF] text-[#102C55]"
  return (
    <p className={cn("whitespace-pre-wrap break-words rounded-lg border px-4 py-3 text-sm leading-6 [overflow-wrap:anywhere]", paint)}>
      {children}
    </p>
  )
}

function getQrCodeUrl(data: string, size = 180) {
  // Use responsive sizes based on screen
  const responsiveSize = typeof window !== 'undefined' && window.innerWidth < 640 ? 120 : size
  return `https://api.qrserver.com/v1/create-qr-code/?size=${responsiveSize}x${responsiveSize}&data=${encodeURIComponent(data)}`
}

function getGoodsQrPayload(memoId: string, item: GoodsLineItem): string {
  const ref = item.qrCodeNumber || `${memoId}-${item.id}`
  return `${window.location.origin}${getDetentionMemoDetailPath(memoId)}?goodsQr=${encodeURIComponent(ref)}&view=goods`
}

function goodsMatchesQr(item: GoodsLineItem, memoId: string, qr: string): boolean {
  const q = qr.trim().toLowerCase()
  if (!q) return true
  return (
    (item.qrCodeNumber || "").toLowerCase() === q ||
    item.id.toLowerCase() === q ||
    `${memoId}-${item.id}`.toLowerCase() === q
  )
}

function GoodsInformationBlock({
  memoId,
  items,
  highlightQr,
  onViewCamera,
}: {
  memoId: string
  items: GoodsLineItem[]
  highlightQr?: string
  onViewCamera?: (item: GoodsLineItem) => void
}) {
  const highlight = highlightQr?.trim().toLowerCase() || ""

  if (!items.length) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-muted-foreground">
          No goods line matched this QR code.
        </CardContent>
      </Card>
    )
  }

  return (
    <Section icon={<Package className="h-4 w-4" />} title="Goods information" tone="teal" className="[&>div:last-child]:p-0">
        <div className="space-y-3 p-3 xl:hidden">
          {items.map((item) => {
            const isHighlight =
              highlight &&
              ((item.qrCodeNumber || "").toLowerCase() === highlight ||
                item.id.toLowerCase() === highlight)
            const camId = locatedCameraId(item)
            return (
              <div
                key={item.id}
                className={`rounded-xl border border-[#BFE8DF] bg-gradient-to-br from-white to-[#F3FBFA] p-3 shadow-sm ${isHighlight ? "ring-2 ring-[#0E9F8A]" : ""}`}
              >
                <div className="mb-3">
                  <GoodsQrDisplay
                    code={item.qrCodeNumber}
                    imageData={getGoodsQrPayload(memoId, item)}
                    size={56}
                  />
                </div>
                <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-sm">
                  <div className="col-span-2 min-w-0">
                    <dt className="text-[10px] font-semibold uppercase tracking-wide text-[#0C7A68]">Description</dt>
                    <dd><GoodsLineText className="font-medium">{item.description || "—"}</GoodsLineText></dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">PCT code</dt>
                    <dd className="break-words">{item.pctCode?.trim() || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Qty / unit</dt>
                    <dd className="break-words">{item.quantity || "—"} / {item.unit || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Condition</dt>
                    <dd className="break-words">{item.condition || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Value (PKR)</dt>
                    <dd className="break-words">{item.assessableValuePkr?.trim() || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Perishable</dt>
                    <dd>{item.perishable ? "Yes" : "No"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Zone</dt>
                    <dd className="truncate">{item.locatedCamera?.zone?.trim() || "—"}</dd>
                  </div>
                  <div className="col-span-2 min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Located camera</dt>
                    <dd className="break-words">{cameraLabel(item.locatedCamera, item.locatedCameraId)}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Detected at</dt>
                    <dd className="break-words">{item.detectedAt?.trim() || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] text-muted-foreground">ID / chassis</dt>
                    <dd className="break-words [overflow-wrap:anywhere]">{item.identificationRef || "—"}</dd>
                  </div>
                  <div className="col-span-2 min-w-0">
                    <dt className="text-[10px] text-muted-foreground">Notes</dt>
                    <dd><GoodsLineText>{item.itemNotes || "—"}</GoodsLineText></dd>
                  </div>
                </dl>
                {camId != null && onViewCamera ? (
                  <Button size="sm" className="mt-3 gap-1.5" onClick={() => onViewCamera(item)}>
                    <Camera className="h-3.5 w-3.5" />
                    View Camera
                  </Button>
                ) : null}
                <div className="mt-3">
                  <p className="mb-1 text-xs text-muted-foreground">Images</p>
                  {item.images && item.images.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {item.images.map((imgUrl, idx) => (
                        <img
                          key={idx}
                          src={imgUrl}
                          alt={`Goods image ${idx + 1}`}
                          className="h-12 w-12 rounded border object-cover"
                        />
                      ))}
                    </div>
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        <div className="hidden xl:block">
          <Table containerClassName="max-w-full" className="w-max min-w-full text-sm">
              <TableHeader>
                <TableRow className="border-b bg-[#E7F4F1] hover:bg-[#E7F4F1]">
                  <TableHead className={goodsHeadClass}>QR Code</TableHead>
                  <TableHead className={goodsHeadClass}>Description</TableHead>
                  <TableHead className={goodsHeadClass}>Located Camera</TableHead>
                  <TableHead className={goodsHeadClass}>PCT Code</TableHead>
                  <TableHead className={goodsHeadClass}>Qty</TableHead>
                  <TableHead className={goodsHeadClass}>Unit</TableHead>
                  <TableHead className={goodsHeadClass}>Condition</TableHead>
                  <TableHead className={goodsHeadClass}>Assessable Value</TableHead>
                  <TableHead className={goodsHeadClass}>Perishable</TableHead>
                  <TableHead className={goodsHeadClass}>ID / Chassis</TableHead>
                  <TableHead className={goodsHeadClass}>Item Notes</TableHead>
                  <TableHead className={goodsHeadClass}>Images</TableHead>
                  <TableHead className={goodsHeadClass} />
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((item) => {
                  const isHighlight =
                    highlight &&
                    ((item.qrCodeNumber || "").toLowerCase() === highlight ||
                      item.id.toLowerCase() === highlight)
                  const camId = locatedCameraId(item)
                  return (
                    <TableRow
                      key={item.id}
                      className={isHighlight ? "bg-primary/5 ring-1 ring-inset ring-primary/30" : undefined}
                    >
                      <TableCell className={cn(goodsDetailCellClass, "w-[7rem]")}>
                        <GoodsQrDisplay
                          code={item.qrCodeNumber}
                          imageData={getGoodsQrPayload(memoId, item)}
                          size={56}
                        />
                      </TableCell>
                      <TableCell className={cn(goodsDetailCellClass, "min-w-[10rem] max-w-[16rem] font-medium")}>
                        <GoodsLineText>{item.description || "—"}</GoodsLineText>
                      </TableCell>
                      <TableCell className={cn(goodsDetailCellClass, "max-w-[12rem] overflow-hidden text-xs")}>
                        <div className="max-w-[11rem] space-y-0.5">
                          <p className="truncate text-sm font-medium" title={cameraLabel(item.locatedCamera, item.locatedCameraId)}>
                            {cameraLabel(item.locatedCamera, item.locatedCameraId)}
                          </p>
                          <p className="text-muted-foreground">
                            Zone: {item.locatedCamera?.zone?.trim() || "—"}
                          </p>
                          {item.detectedAt ? (
                            <p className="text-muted-foreground">{item.detectedAt}</p>
                          ) : null}
                        </div>
                      </TableCell>
                      <TableCell className={cn(goodsDetailCellClass, "font-mono")}>{item.pctCode?.trim() || "—"}</TableCell>
                      <TableCell className={goodsDetailCellClass}>{item.quantity || "—"}</TableCell>
                      <TableCell className={goodsDetailCellClass}>{item.unit || "—"}</TableCell>
                      <TableCell className={goodsDetailCellClass}>{item.condition || "—"}</TableCell>
                      <TableCell className={goodsDetailCellClass}>{item.assessableValuePkr?.trim() || "—"}</TableCell>
                      <TableCell className={goodsDetailCellClass}>{item.perishable ? "Yes" : "No"}</TableCell>
                      <TableCell className={cn(goodsDetailCellClass, "max-w-[10rem] overflow-hidden")}>
                        <span className="block max-w-[9rem] truncate" title={item.identificationRef || ""}>
                          {item.identificationRef || "—"}
                        </span>
                      </TableCell>
                      <TableCell className={cn(goodsDetailCellClass, "min-w-[8rem] max-w-[14rem] text-muted-foreground")}>
                        <GoodsLineText>{item.itemNotes || "—"}</GoodsLineText>
                      </TableCell>
                      <TableCell className={goodsDetailCellClass}>
                        {item.images && item.images.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {item.images.map((imgUrl, idx) => (
                              <img
                                key={idx}
                                src={imgUrl}
                                alt={`Goods image ${idx + 1}`}
                                className="h-8 w-8 sm:h-10 sm:w-10 object-cover rounded border"
                              />
                            ))}
                          </div>
                        ) : item.evidenceUrl ? (
                          <img
                            src={item.evidenceUrl}
                            alt="Evidence"
                            className="h-8 w-8 sm:h-10 sm:w-10 object-cover rounded border"
                          />
                        ) : (
                          <span className="text-muted-foreground text-xs">—</span>
                        )}
                      </TableCell>
                      <TableCell className={goodsDetailCellClass}>
                        {camId != null && onViewCamera ? (
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-1 whitespace-nowrap"
                            onClick={() => onViewCamera(item)}
                          >
                            <Camera className="h-3.5 w-3.5" />
                            View Camera
                          </Button>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
        </div>
    </Section>
  )
}

export default function DetentionMemoDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { pathname } = useLocation()
  const listPath = getDetentionMemoListPath(pathname)
  const listSection = getDetentionMemoSectionCrumb(pathname)
  const [searchParams] = useSearchParams()
  const [row, setRow] = useState<DetentionMemoApiRecord | null | undefined>(undefined)
  const [viewCamera, setViewCamera] = useState<ViewCameraTarget | null>(null)
  const printMode = searchParams.get("print")
  const autoPrint = searchParams.get("autoprint") === "1"
  const autoSavePdf = searchParams.get("savepdf") === "1"
  const isPrintMode = printMode === "qr" || printMode === "full"
  const goodsQrFilter = searchParams.get("goodsQr")?.trim() || ""
  const goodsOnlyView = searchParams.get("view") === "goods" && !!goodsQrFilter

  const openViewCamera = (item: GoodsLineItem) => {
    const camId = locatedCameraId(item)
    if (camId == null) return
    setViewCamera({
      cameraId: camId,
      label: cameraLabel(item.locatedCamera, camId),
      itemDescription: item.description?.trim() || undefined,
      detectedAt: item.detectedAt?.trim() || undefined,
      evidenceUrl: item.evidenceUrl?.trim() || undefined,
    })
  }

  useEffect(() => {
    if (!id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset/loading state for the fetch keyed on the route id
      setRow(null)
      return
    }
    let cancelled = false
    setRow(undefined)
    fetchDetentionMemoById(id)
      .then((data) => {
        if (!cancelled) setRow(data)
      })
      .catch(() => {
        if (!cancelled) setRow(null)
      })
    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    if (isPrintMode && autoPrint && row) {
      const timer = window.setTimeout(() => window.print(), 350)
      return () => window.clearTimeout(timer)
    }
  }, [isPrintMode, autoPrint, row])

  if (!id) {
    return (
      <ModulePageLayout
        title="Detention Memo Not Found"
        description="The requested record could not be found."
        breadcrumbs={[
          listSection,
          { label: "Detention Memo", href: listPath },
          { label: "Detail" },
        ]}
      >
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground mb-4">Record not found or has been removed.</p>
            <Button asChild variant="outline">
              <Link to={listPath}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Detention Memo
              </Link>
            </Button>
          </CardContent>
        </Card>
      </ModulePageLayout>
    )
  }

  if (row === undefined) {
    return (
      <ModulePageLayout
        title="Detention Memo"
        description="Loading record…"
        breadcrumbs={[
          listSection,
          { label: "Detention Memo", href: listPath },
          { label: "Detail" },
        ]}
      >
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">Loading detention memo…</p>
          </CardContent>
        </Card>
      </ModulePageLayout>
    )
  }

  if (!row) {
    return (
      <ModulePageLayout
        title="Detention Memo Not Found"
        description="The requested record could not be found."
        breadcrumbs={[
          listSection,
          { label: "Detention Memo", href: listPath },
          { label: "Detail" },
        ]}
      >
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground mb-4">Record not found or has been removed.</p>
            <Button asChild variant="outline">
              <Link to={listPath}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Detention Memo
              </Link>
            </Button>
          </CardContent>
        </Card>
      </ModulePageLayout>
    )
  }

  // Always encode the live detail route — never stale DB payloads with /detention-memo or ?print=full
  const qrPayload = getDetentionMemoScanUrl(row.id)
  const qrNumber = row.memoQrCodeNumber || `DM-${row.caseNo}`

  if (printMode === "qr") {
    return (
      <DetentionMemoQRPrint
        caseNo={row.caseNo}
        referenceNumber={row.referenceNumber}
        createdBy={row.createdBy || "ASO Portal"}
        createdAt={row.createdAt}
        qrPayload={qrPayload}
        qrNumber={qrNumber}
      />
    )
  }

  if (printMode === "full") {
    return (
      <DetentionMemoReportPrint
        row={row}
        qrPayload={qrPayload}
        qrNumber={qrNumber}
        autoSavePdf={autoSavePdf}
      />
    )
  }

  if (goodsOnlyView) {
    const visibleGoods = (row.goodsItems || []).filter((item) =>
      goodsMatchesQr(item, row.id, goodsQrFilter)
    )
    const goodsLabel =
      visibleGoods[0]?.description ||
      visibleGoods[0]?.qrCodeNumber ||
      goodsQrFilter

    return (
      <ModulePageLayout
        title={`Goods: ${goodsLabel}`}
        description={`Goods line from detention memo ${row.caseNo}.`}
        breadcrumbs={[
          listSection,
          { label: "Detention Memo", href: listPath },
          { label: row.caseNo, href: `${listPath}/${encodeURIComponent(row.id)}` },
          { label: "Goods" },
        ]}
      >
        <div className="grid gap-4 sm:gap-6">
          <div className="flex flex-wrap items-center gap-2">
            <Button asChild variant="outline" size="sm">
              <Link to={listPath}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to list
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to={`${listPath}/${encodeURIComponent(row.id)}`}>
                <FileText className="h-4 w-4 mr-2" />
                View full memo
              </Link>
            </Button>
          </div>
          <GoodsInformationBlock
            memoId={row.id}
            items={visibleGoods}
            highlightQr={goodsQrFilter}
            onViewCamera={openViewCamera}
          />
          <ViewLocatedCameraDialog target={viewCamera} onClose={() => setViewCamera(null)} />
        </div>
      </ModulePageLayout>
    )
  }

  // Normal view (screen) – includes sidebar via ModulePageLayout
  return (
    <ModulePageLayout
      title="Detention Memo"
      description={
        row.referenceNumber
          ? `Memo No. ${row.referenceNumber} · Pakistan Customs detention memo details.`
          : "Pakistan Customs detention memo details. Data from the database."
      }
      breadcrumbs={[
        listSection,
        { label: "Detention Memo", href: listPath },
        { label: row.caseNo },
      ]}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {canUserFullyEditSeizureDocs(getStoredUser()?.role) ? (
            <Button size="sm" asChild>
              <Link to={getDetentionMemoEditPath(row.id, pathname)}>
                <Pencil className="mr-2 h-4 w-4" />
                Edit record
              </Link>
            </Button>
          ) : null}
          <PrintMenu
            printHref={`${listPath}/${encodeURIComponent(row.id)}?print=full`}
            pdfHref={`${listPath}/${encodeURIComponent(row.id)}?print=full&savepdf=1`}
          />
        </div>
      }
    >
      {/* Global print style to hide sidebar if user accidentally prints this view */}
      <style>{`
        @media print {
          aside, nav, header, .sidebar, .main-nav, .breadcrumbs {
            display: none !important;
          }
          main, .main-content, .print-keep {
            margin-left: 0 !important;
            padding: 0 !important;
            width: 100% !important;
          }
        }
      `}</style>
      <div className="grid gap-4 print-keep">
        <div className="overflow-hidden rounded-xl bg-[#102C55] text-white shadow-md">
          <div className="flex flex-wrap items-start justify-between gap-4 px-5 py-5">
            <div className="min-w-0">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#D6EBFF]">
                Pakistan Customs · Detention memo
              </p>
              <h2 className="mt-1 break-all font-mono text-[24px] font-semibold leading-tight text-white">
                {row.caseNo}
              </h2>
              <div className="mt-3 flex flex-wrap gap-2 text-[12px]">
                <span className="rounded-full bg-white/15 px-2.5 py-1">
                  Memo <span className="font-semibold">{row.referenceNumber || "—"}</span>
                </span>
                <span className="rounded-full bg-[#FFE3B0] px-2.5 py-1 font-semibold text-[#7A4A00]">
                  {row.detentionType || "Type —"}
                </span>
                <span className="rounded-full bg-[#D7F6EA] px-2.5 py-1 font-semibold text-[#0C6B45]">
                  {row.dateTimeDetention || "Date —"}
                </span>
                <span className="max-w-full rounded-full bg-white/15 px-2.5 py-1">
                  {row.placeOfDetention || "Place —"}
                </span>
              </div>
            </div>
            <StatusMark status={row.verificationStatus} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <Snapshot
            label="Goods lines"
            value={String(row.goodsItems?.length ?? 0)}
            className="border-[#B7E6DC] bg-[#E7F8F4] text-[#0C6B56]"
          />
          <Snapshot
            label="Verification"
            value={row.verificationStatus || "Pending"}
            className={
              row.verificationStatus === "Verified"
                ? "border-[#B7E7CC] bg-[#E8F8EF] text-[#0E7A45]"
                : "border-[#F5D7A8] bg-[#FFF4E4] text-[#9A5B00]"
            }
          />
          <Snapshot
            label="Settlement"
            value={row.settlementStatus || "—"}
            className="border-[#D4CBF6] bg-[#F3F0FF] text-[#5340B0]"
          />
          <Snapshot
            label="Disposition"
            value={row.dispositionStatus || "—"}
            className="border-[#C5DDF7] bg-[#EAF3FF] text-[#0B5CAD]"
          />
        </div>

        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <Section icon={<FileText className="h-4 w-4" />} title="Case particulars" tone="blue">
            <FieldGrid>
              <FieldTile label="Case no." value={row.caseNo} />
              <FieldTile label="Detention memo no." value={row.referenceNumber} />
              <FieldTile label="FIR number" value={row.firNumber} />
              <FieldTile label="Directorate" value={row.directorate} />
              <FieldTile label="Date/time of occurrence" value={row.dateTimeOccurrence} />
              <FieldTile label="Date/time of detention" value={row.dateTimeDetention} />
              <FieldTile label="Place of occurrence" value={row.placeOfOccurrence} />
              <FieldTile label="Place of detention" value={row.placeOfDetention} />
              <FieldTile label="Detention type" value={row.detentionType} />
              <FieldTile label="Reason for detention" value={row.reasonForDetention} />
              <FieldTile label="Location of detention" value={row.locationOfDetention} />
              <FieldTile label="Disposition status" value={row.dispositionStatus} />
              <FieldTile label="Search / chassis number" value={row.searchChassisNumber} />
              <FieldTile label="Receipt officer" value={row.receiptOfficer} />
              <FieldTile label="GD number" value={row.gdNumber} />
              <FieldTile label="GD number 2" value={row.gdNumber2} />
            </FieldGrid>
          </Section>
          <Section icon={<QrCode className="h-4 w-4" />} title="Memo QR" tone="violet">
            <div className="flex flex-col items-center rounded-xl bg-gradient-to-b from-[#F4F0FF] to-white px-3 py-4 text-center">
              <img
                src={getQrCodeUrl(qrPayload, 168)}
                alt="Memo QR code"
                className="h-auto max-w-full rounded-lg border border-[#DDD4F8] bg-white p-2 shadow-sm"
              />
              <p className="mt-3 break-all font-mono text-[11px] text-[#607DA5]">{qrNumber}</p>
              <Button variant="outline" size="sm" className="mt-3 w-full" asChild>
                <Link to={`${listPath}/${encodeURIComponent(row.id)}?print=qr`}>Print QR label</Link>
              </Button>
            </div>
          </Section>
        </div>

        <Section title="Custody and verification" tone="teal">
          <FieldGrid columns="sm:grid-cols-2 xl:grid-cols-4">
            <FieldTile label="Goods detained at" value={row.whereDeposited} />
            <FieldTile label="Settlement status" value={row.settlementStatus} />
            <FieldTile label="Verification status" value={row.verificationStatus} />
            <FieldTile label="Memo QR number" value={row.memoQrCodeNumber || qrNumber} />
          </FieldGrid>
        </Section>

        <Section title="Record history" tone="amber">
          <FieldGrid>
            <FieldTile label="Created by" value={row.createdBy} />
            <FieldTile label="Created on" value={row.createdAt} />
            <FieldTile label="Updated by" value={row.updatedBy} />
            <FieldTile label="Updated on" value={row.updatedAt ?? row.createdAt} />
          </FieldGrid>
          {(row.auditLog?.length ?? 0) > 0 ? (
            <ol className="mt-4 space-y-3 border-t border-[#E8F0FA] pt-4">
              {row.auditLog!.map((entry) => (
                <li key={entry.id} className="grid grid-cols-[10px_minmax(0,1fr)] gap-3">
                  <span className="mt-1.5 h-2.5 w-2.5 rounded-full bg-[#0875E1]" />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-[#102C55]">
                      {entry.actionLabel || entry.action}
                      {entry.performedBy ? ` · ${entry.performedBy}` : ""}
                    </p>
                    {entry.message ? (
                      <p className="mt-0.5 whitespace-pre-wrap break-words text-xs text-[#607DA5]">{entry.message}</p>
                    ) : null}
                    <p className="mt-0.5 text-[11px] text-[#90A7C4]">
                      {entry.createdAt ? new Date(entry.createdAt).toLocaleString() : "—"}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <p className="mt-3 text-sm text-[#607DA5]">No audit events recorded yet.</p>
          )}
        </Section>

        <div className="grid items-start gap-4 xl:grid-cols-2">
          <Section icon={<Users className="h-4 w-4" />} title="Owner" tone="blue">
            <div className="flex items-start gap-4">
              {row.owner?.picture ? (
                <img
                  src={row.owner.picture}
                  alt="Owner"
                  className="h-20 w-20 shrink-0 rounded-lg border border-[#E4EEF8] bg-[#F7FAFE] object-cover"
                />
              ) : null}
              <div className="min-w-0 flex-1">
                <FieldGrid>
                  <FieldTile label="Name" value={row.owner?.name} />
                  <FieldTile label="CNIC" value={row.owner?.cnic} />
                  <FieldTile label="Contact" value={row.owner?.contact} />
                </FieldGrid>
              </div>
            </div>
          </Section>
          <Section icon={<Users className="h-4 w-4" />} title="Driver" tone="violet">
            <div className="flex items-start gap-4">
              {row.driver?.picture ? (
                <img
                  src={row.driver.picture}
                  alt="Driver"
                  className="h-20 w-20 shrink-0 rounded-lg border border-[#E4EEF8] bg-[#F7FAFE] object-cover"
                />
              ) : null}
              <div className="min-w-0 flex-1">
                <FieldGrid>
                  <FieldTile label="Name" value={row.driver?.name} />
                  <FieldTile label="CNIC" value={row.driver?.cnic} />
                  <FieldTile label="Contact" value={row.driver?.contact} />
                </FieldGrid>
              </div>
            </div>
          </Section>
        </div>

        <Section title="Purpose of detention" tone="amber">
          <NoteBlock tone="amber">{row.purposeOfDetention || "—"}</NoteBlock>
        </Section>

            {row.mediaAttachments && row.mediaAttachments.length > 0 && (
              <Section icon={<Paperclip className="h-4 w-4" />} title="Attached documents and videos" tone="rose">
                <div className="space-y-4">
                  {row.mediaAttachments.map((att) =>
                    att.kind === "video" ? (
                      <div key={att.id} className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground break-words">{att.originalFilename || "Video"}</p>
                        <video src={att.url} controls className="w-full max-w-2xl rounded-lg border bg-black">
                          Your browser does not support video playback.
                        </video>
                      </div>
                    ) : (
                      <div key={att.id} className="flex flex-wrap items-center gap-3">
                        <a
                          href={att.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium text-primary underline-offset-4 hover:underline break-words"
                        >
                          {att.originalFilename || "Download document"}
                        </a>
                      </div>
                    )
                  )}
                </div>
              </Section>
            )}

            <Section title="Memo description" tone="blue">
              <NoteBlock>{row.briefFacts || "—"}</NoteBlock>
            </Section>

            {row.goodsItems && row.goodsItems.length > 0 && (
              <GoodsInformationBlock
                memoId={row.id}
                items={row.goodsItems}
                onViewCamera={openViewCamera}
              />
            )}

            <WmsFlowOverviewPanel detentionMemoId={row.id} caseNo={row.caseNo} />

            <DestructionRecordsPanel
              detentionMemoId={row.id}
              caseNo={row.caseNo}
              title="Destruction history"
              description="All warehouse destruction sessions for this detention — view detailed reports with camera evidence and inventory deductions."
            />

            <ViewLocatedCameraDialog target={viewCamera} onClose={() => setViewCamera(null)} />

            <Section title="Officer notes" tone="teal">
              <FieldGrid>
                <FieldTile label="Seizing officer notes" value={row.seizingOfficerNotes} />
                <FieldTile label="Examining officer notes" value={row.examiningOfficerNotes} />
                <FieldTile label="Detention / customs clarification" value={row.detentionNotes} />
                <FieldTile label="Forwarding officer remarks" value={row.forwardingOfficerRemarks} />
              </FieldGrid>
            </Section>

            <Button asChild variant="outline" className="w-full sm:w-auto">
              <Link to={listPath}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to list
              </Link>
            </Button>
      </div>
    </ModulePageLayout>
  )
}