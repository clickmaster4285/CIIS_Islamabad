import { useEffect, useState, type ReactNode } from "react"
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom"
import {
  ArrowLeft,
  CheckCircle,
  ChevronDown,
  ClipboardList,
  Clock,
  FileOutput,
  FileText,
  History,
  MapPin,
  Package,
  Paperclip,
  Pencil,
  Printer,
  Scale,
  Send,
  Stamp,
  Trash2,
  User,
  Users,
  XCircle,
} from "lucide-react"
import { ModulePageLayout } from "@/components/dashboard/module-page-layout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  ROUTES,
  getDetentionMemoDetailPath,
  getSeizureMgmtNoteSheetDetailPath,
  getSeizureMgmtNoteSheetEditPath,
} from "@/routes/config"
import {
  canUserApproveNoteSheet,
  canUserDeleteNoteSheet,
  canUserEditNoteSheet,
  deleteNoteSheet,
  fetchNoteSheetById,
  noteSheetApproval,
  type NoteSheetRecord,
  type NoteSheetStatus,
} from "@/lib/seizure-management-api"
import { getStoredUser, setAuthenticatedWithToken, type AuthUser } from "@/lib/auth"
import { getStoredToken } from "@/lib/api"
import { fetchCurrentUser } from "@/lib/users-api"
import { toast } from "@/hooks/use-toast"
import { reportMissingField } from "@/lib/form-missing-field"
import { GoodsLineText, goodsDetailCellClass, goodsHeadClass } from "@/components/goods/goods-line-text-field"
import { GoodsQrDisplay } from "@/components/goods/goods-qr-display"
import { cn } from "@/lib/utils"
import {
  ViewLocatedCameraDialog,
  ViewCameraButton,
  locatedCameraLabel,
  type ViewCameraTarget,
} from "@/components/detention/view-located-camera-dialog"
import NoteSheetReportPrint from "@/components/seizure/NoteSheetReportPrint"

function getQrCodeUrl(data: string, size = 120) {
  const responsiveSize = typeof window !== "undefined" && window.innerWidth < 640 ? Math.min(size, 100) : size
  return `https://api.qrserver.com/v1/create-qr-code/?size=${responsiveSize}x${responsiveSize}&data=${encodeURIComponent(data)}`
}

function statusChipClass(status: NoteSheetStatus) {
  if (status === "Approved") return "bg-[#D9F5E8] text-[#0E7A45]"
  if (status === "Rejected") return "bg-[#FFDDE4] text-[#B4234A]"
  if (status === "Submitted") return "bg-[#FFE8C2] text-[#9A5B00]"
  return "bg-[#D7EBFF] text-[#0B5CAD]"
}

type SectionTone = "blue" | "teal" | "amber" | "violet" | "rose"

const SECTION_TONE: Record<SectionTone, { chip: string; head: string; border: string; rule: string }> = {
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

function FieldTile({ label, value }: { label: string; value?: string | null }) {
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

function FieldGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">{children}</div>
}

function Section({
  icon,
  title,
  tone = "blue",
  children,
  flush,
}: {
  icon: ReactNode
  title: string
  tone?: SectionTone
  children: ReactNode
  flush?: boolean
}) {
  const paint = SECTION_TONE[tone]
  return (
    <section className={cn("min-w-0 overflow-hidden rounded-xl border bg-white shadow-sm", paint.border)}>
      <div className={cn("flex items-center gap-2.5 border-b px-4 py-3", paint.head, paint.rule)}>
        <span className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-lg", paint.chip)}>{icon}</span>
        <h3 className="min-w-0 truncate text-[13px] font-semibold text-[#102C55]">{title}</h3>
      </div>
      <div className={flush ? "" : "bg-white p-4"}>{children}</div>
    </section>
  )
}

function NoteBlock({ children, tone = "blue" }: { children: ReactNode; tone?: "blue" | "rose" | "amber" }) {
  const paint =
    tone === "rose"
      ? "border-[#F6C9D3] bg-[#FFF5F7] text-[#8E2340]"
      : tone === "amber"
        ? "border-[#F5D7A8] bg-[#FFF8EE] text-[#5C3B12]"
        : "border-[#C5DDF7] bg-[#F3F8FF] text-[#102C55]"
  return (
    <div className={cn("whitespace-pre-wrap break-words rounded-lg border px-4 py-3 text-sm leading-6 [overflow-wrap:anywhere]", paint)}>
      {children}
    </div>
  )
}

export default function NoteSheetDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const printMode = searchParams.get("print")
  const autoPrint = searchParams.get("autoprint") === "1"
  const autoSavePdf = searchParams.get("savepdf") === "1"
  const [row, setRow] = useState<NoteSheetRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [acting, setActing] = useState(false)
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(() => getStoredUser())
  const [approver, setApprover] = useState("")
  const [approvalRemarks, setApprovalRemarks] = useState("")
  const [rejectionReason, setRejectionReason] = useState("")
  const [invalidField, setInvalidField] = useState("")
  const [viewCamera, setViewCamera] = useState<ViewCameraTarget | null>(null)

  const load = () => {
    if (!id) return
    setLoading(true)
    fetchNoteSheetById(id)
      .then((data) => {
        setRow(data)
        void noteSheetApproval(data.id, "view").catch(() => undefined)
      })
      .catch(() => setRow(null))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once per id
  }, [id])

  useEffect(() => {
    let cancelled = false
    const stored = getStoredUser()
    if (stored) {
      setCurrentUser(stored)
      setApprover((stored.full_name || "").trim() || stored.username || "")
    }
    fetchCurrentUser()
      .then((profile) => {
        if (cancelled || !profile) return
        const base = getStoredUser()
        const merged: AuthUser = {
          id: profile.id,
          username: profile.username,
          email: profile.email || base?.email || "",
          role: profile.role || base?.role || "",
          phone: profile.phone || base?.phone || "",
          location: profile.location || base?.location,
          full_name: profile.full_name || "",
          designation: profile.designation || base?.designation,
          employee_id: profile.employee_id || base?.employee_id,
          cell_no: profile.cell_no || base?.cell_no,
          office_phone_1: profile.office_phone_1 || base?.office_phone_1,
          office_phone_2: profile.office_phone_2 || base?.office_phone_2,
          collectorate: profile.collectorate || base?.collectorate,
          is_active: profile.is_active,
        }
        setCurrentUser(merged)
        setApprover((merged.full_name || "").trim() || merged.username || "")
        const token = getStoredToken()
        if (token) setAuthenticatedWithToken(token, merged)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    if (printMode === "full" && autoPrint && row) {
      const timer = window.setTimeout(() => window.print(), 350)
      return () => window.clearTimeout(timer)
    }
  }, [printMode, autoPrint, row])

  if (loading) {
    return (
      <ModulePageLayout title="Note Sheet" description="Loading..." breadcrumbs={[]}>
        <p className="text-muted-foreground">Loading...</p>
      </ModulePageLayout>
    )
  }

  if (!row) {
    return (
      <ModulePageLayout
        title="Note Sheet"
        description="Not found"
        breadcrumbs={[
          { label: "Seizure Management", href: ROUTES.SEIZURE_MANAGEMENT },
          { label: "Note Sheet", href: ROUTES.SEIZURE_MGMT_NOTE_SHEET },
        ]}
      >
        <p className="text-muted-foreground">Note sheet not found.</p>
        <Button className="mt-4" variant="outline" onClick={() => navigate(ROUTES.SEIZURE_MGMT_NOTE_SHEET)}>
          Back
        </Button>
      </ModulePageLayout>
    )
  }

  if (printMode === "full") {
    return <NoteSheetReportPrint row={row} autoSavePdf={autoSavePdf} />
  }

  const runApproval = async (action: "submit" | "approve" | "reject") => {
    if (action === "approve" && !approver.trim()) {
      setInvalidField("ns-approver")
      reportMissingField({ id: "ns-approver", label: "Approving Officer" })
      return
    }
    if (action === "reject" && !rejectionReason.trim()) {
      setInvalidField("ns-rejection")
      reportMissingField({ id: "ns-rejection", label: "Rejection Reason" })
      return
    }
    setInvalidField("")
    setActing(true)
    try {
      const updated = await noteSheetApproval(row.id, action, {
        approvedBy: approver,
        approvalRemarks,
        rejectionReason,
      })
      setRow(updated)
      toast({
        title:
          action === "submit"
            ? "Note sheet sent for approval"
            : action === "approve"
              ? "Note sheet approved"
              : "Note sheet rejected",
      })
    } catch (e) {
      toast({
        title: e instanceof Error ? e.message : "Action failed",
        variant: "destructive",
      })
    } finally {
      setActing(false)
    }
  }

  const title = row.noteSheetNo || row.referenceNumber || row.subject || "Note Sheet"
  const canEdit = canUserEditNoteSheet(row, currentUser?.role)
  const canApprove = canUserApproveNoteSheet(row, currentUser)
  const canDelete = canUserDeleteNoteSheet(row, currentUser?.role)

  const handleDelete = async () => {
    const label = row.noteSheetNo || row.referenceNumber || "this note sheet"
    const approvedHint =
      row.status === "Approved"
        ? " This sheet is already approved. Only a higher official can delete it."
        : ""
    if (!window.confirm(`Delete ${label}? This cannot be undone.${approvedHint}`)) return
    setActing(true)
    try {
      await deleteNoteSheet(row.id)
      toast({ title: `${label} deleted` })
      navigate(ROUTES.SEIZURE_MGMT_NOTE_SHEET)
    } catch (e) {
      toast({
        title: e instanceof Error ? e.message : "Could not delete note sheet",
        variant: "destructive",
      })
    } finally {
      setActing(false)
    }
  }
  const approvalStatusLabel =
    row.status === "Approved"
      ? "Approved"
      : row.status === "Rejected"
        ? "Rejected"
        : row.status === "Submitted"
          ? "Pending"
          : "Draft"

  return (
    <ModulePageLayout
      title={title}
      description={`${row.subject || "Note sheet"} · ${row.status}`}
      breadcrumbs={[
        { label: "Seizure Management", href: ROUTES.SEIZURE_MANAGEMENT },
        { label: "Note Sheet", href: ROUTES.SEIZURE_MGMT_NOTE_SHEET },
        { label: row.noteSheetNo || row.referenceNumber || "Detail" },
      ]}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" className="text-[#102C55]" asChild>
          <Link to={ROUTES.SEIZURE_MGMT_NOTE_SHEET}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Link>
        </Button>
        {canEdit && (
          <Button variant="outline" size="sm" className="border-[#C5DDF7] text-[#102C55]" asChild>
            <Link to={getSeizureMgmtNoteSheetEditPath(row.id)}>
              <Pencil className="mr-2 h-4 w-4" />
              Edit
            </Link>
          </Button>
        )}
        {canDelete && (
          <Button
            variant="outline"
            size="sm"
            className="border-[#F6C9D3] text-[#B4234A] hover:bg-[#FFF5F7] hover:text-[#B4234A]"
            disabled={acting}
            onClick={() => void handleDelete()}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </Button>
        )}
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" className="bg-[#102C55] text-white hover:bg-[#0C2748]">
              <Printer className="mr-2 h-4 w-4" />
              Print
              <ChevronDown className="ml-1 h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-0 w-max">
            <DropdownMenuItem asChild>
              <Link to={`${getSeizureMgmtNoteSheetDetailPath(row.id)}?print=full`}>
                <Printer className="h-4 w-4" />
                Print
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link to={`${getSeizureMgmtNoteSheetDetailPath(row.id)}?print=full&savepdf=1`}>
                <FileOutput className="h-4 w-4" />
                Save as PDF
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="mb-4 overflow-hidden rounded-xl bg-[#102C55] px-5 py-4 text-white shadow-md">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#D6EBFF]">
          Seizure management · Note sheet
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold", statusChipClass(row.status))}>
            {row.status}
          </span>
          {row.priority === "Urgent" ? (
            <span className="inline-flex items-center rounded-full bg-[#FFDDE4] px-2.5 py-0.5 text-[11px] font-semibold text-[#B4234A]">
              Urgent
            </span>
          ) : (
            <span className="inline-flex items-center rounded-full bg-white/15 px-2.5 py-0.5 text-[11px] font-semibold text-white">
              {row.priority || "Normal"}
            </span>
          )}
        </div>
        <p className="mt-1 max-w-3xl text-sm text-[#D6EBFF]">{row.subject?.trim() || "Note sheet record"}</p>
      </div>

      {row.status === "Submitted" && canApprove && (
        <div className="mb-4 rounded-xl border border-[#C5DDF7] bg-[#EAF3FF] px-4 py-3 text-sm text-[#102C55]">
          Prepared by <span className="font-semibold">{row.preparedBy || row.createdBy || "—"}</span>
          {" · "}
          Pending approval by Assistant Collector, Deputy Collector, Location Admin, or Super Admin.
          You can approve or reject as{" "}
          <span className="font-semibold">{currentUser?.full_name || currentUser?.username || "you"}</span>.
        </div>
      )}

      {row.status === "Submitted" && !canApprove && (
        <div className="mb-4 rounded-xl border border-[#F5D7A8] bg-[#FFF8EE] px-4 py-3 text-sm text-[#5C3B12]">
          Prepared by <span className="font-semibold">{row.preparedBy || row.createdBy || "—"}</span>
          {" · "}
          Waiting for approval from Assistant Collector, Deputy Collector, Location Admin, or Super
          Admin.
        </div>
      )}

      <div className="space-y-4 max-w-[1600px]">
        {row.timeline?.length > 0 && (
          <Section icon={<Clock className="h-4 w-4" />} title="Timeline" tone="amber">
            <ol className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {row.timeline.map((step) => (
                <li
                  key={step.key}
                  className={cn(
                    "min-w-0 rounded-lg border px-3 py-2.5",
                    step.done ? "border-[#BFE8DF] bg-[#F3FBFA]" : "border-[#E4EEF8] bg-[#F8FBFF]"
                  )}
                >
                  <p className={cn("text-sm font-semibold", step.done ? "text-[#0C7A68]" : "text-[#607DA5]")}>
                    {step.label}
                  </p>
                  <p className="mt-0.5 text-xs text-[#607DA5]">{step.at || "—"}</p>
                </li>
              ))}
            </ol>
          </Section>
        )}

        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_240px]">
          <Section icon={<FileText className="h-4 w-4" />} title="1. Basic information" tone="blue">
            <FieldGrid>
              <FieldTile label="Note sheet number" value={row.noteSheetNo || row.referenceNumber} />
              <FieldTile label="Date & time" value={row.dateTime} />
              <FieldTile label="Office / region" value={row.office} />
              <FieldTile label="Investigation / case number" value={row.caseNo} />
              <FieldTile label="Priority" value={row.priority} />
              <FieldTile label="Status" value={row.status} />
              <div className="sm:col-span-2">
                <FieldTile label="Subject" value={row.subject} />
              </div>
            </FieldGrid>
          </Section>
          <Section icon={<Stamp className="h-4 w-4" />} title="QR code" tone="violet">
            <div className="flex flex-col items-center gap-2">
              <img
                src={getQrCodeUrl(
                  `${typeof window !== "undefined" ? window.location.origin : ""}${getSeizureMgmtNoteSheetDetailPath(row.id)}`,
                  160
                )}
                alt="Note sheet QR code"
                className="rounded-lg border border-[#D9D0F7] bg-white p-2"
              />
              <p className="break-all text-center font-mono text-[10px] text-[#607DA5]">
                {row.noteSheetNo || row.referenceNumber || row.id}
              </p>
            </div>
          </Section>
        </div>

        <Section icon={<User className="h-4 w-4" />} title="2. Preparing officer" tone="teal">
          <p className="mb-3 text-xs text-[#607DA5]">Officer who created this note sheet, not the current login.</p>
          <FieldGrid>
            <FieldTile label="Preparing officer" value={row.preparedBy} />
            <FieldTile label="Badge / employee ID" value={row.badgeId} />
            <FieldTile label="Designation" value={row.designation} />
            <FieldTile label="Department" value={row.department} />
            <div className="sm:col-span-2">
              <FieldTile label="Contact number" value={row.officerContact} />
            </div>
          </FieldGrid>
        </Section>

        <Section icon={<Users className="h-4 w-4" />} title="3. Suspect / accused" tone="violet">
          <FieldGrid>
            <FieldTile label="Name" value={row.accusedName} />
            <FieldTile label="Father name" value={row.accusedFatherName} />
            <FieldTile label="CNIC / passport" value={row.accusedCnic} />
            <FieldTile label="Mobile number" value={row.accusedMobile} />
            <div className="sm:col-span-2">
              <FieldTile label="Address" value={row.accusedAddress} />
            </div>
            <FieldTile label="Business name" value={row.businessName} />
            <FieldTile label="NTN / STRN" value={row.ntnStrn} />
          </FieldGrid>
        </Section>

        <Section icon={<Package className="h-4 w-4" />} title="4. Goods information" tone="teal" flush>
            {row.items?.length ? (
              <div className="max-w-full overflow-x-auto">
                <Table className="w-max min-w-full text-sm" containerClassName="overflow-visible">
                  <TableHeader>
                    <TableRow className="border-b border-[#D3F0E9] bg-[#E7F4F1] hover:bg-[#E7F4F1]">
                      <TableHead className={goodsHeadClass}>QR Code</TableHead>
                      <TableHead className={goodsHeadClass}>Description</TableHead>
                      <TableHead className={goodsHeadClass}>Located Camera</TableHead>
                      <TableHead className={goodsHeadClass}>Qty</TableHead>
                      <TableHead className={goodsHeadClass}>Unit</TableHead>
                      <TableHead className={goodsHeadClass}>Condition</TableHead>
                      <TableHead className={goodsHeadClass}>Perishable</TableHead>
                      <TableHead className={goodsHeadClass}>ID / Chassis</TableHead>
                      <TableHead className={goodsHeadClass}>Item Notes</TableHead>
                      <TableHead className={goodsHeadClass}>Images</TableHead>
                      <TableHead className={goodsHeadClass} />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {row.items.map((item, i) => (
                      <TableRow key={item.id ?? i}>
                        <TableCell className={cn(goodsDetailCellClass, "w-[7.5rem]")}>
                          <GoodsQrDisplay code={item.qrCodeNumber} size={64} />
                        </TableCell>
                        <TableCell className={cn(goodsDetailCellClass, "min-w-[10rem] max-w-[16rem] font-medium")}>
                          <GoodsLineText>{item.product || item.description || "—"}</GoodsLineText>
                        </TableCell>
                        <TableCell className={cn(goodsDetailCellClass, "min-w-[9rem] text-xs")}>
                          <div className="space-y-0.5">
                            <p className="font-medium text-sm">
                              {locatedCameraLabel(item.locatedCamera, item.locatedCameraId)}
                            </p>
                            <p className="text-muted-foreground">
                              Zone: {item.locatedCamera?.zone?.trim() || "—"}
                            </p>
                            {item.detectedAt ? (
                              <p className="text-muted-foreground">{item.detectedAt}</p>
                            ) : null}
                          </div>
                        </TableCell>
                        <TableCell className={goodsDetailCellClass}>{item.quantity || "—"}</TableCell>
                        <TableCell className={goodsDetailCellClass}>{item.unit || "—"}</TableCell>
                        <TableCell className={goodsDetailCellClass}>{item.condition || "—"}</TableCell>
                        <TableCell className={goodsDetailCellClass}>{item.perishable ? "Yes" : "No"}</TableCell>
                        <TableCell className={goodsDetailCellClass}>{item.identificationRef || "—"}</TableCell>
                        <TableCell className={cn(goodsDetailCellClass, "min-w-[8rem] max-w-[14rem]")}>
                          <GoodsLineText>{item.remarks || item.itemNotes || "—"}</GoodsLineText>
                        </TableCell>
                        <TableCell className={goodsDetailCellClass}>
                          {item.images?.length ? (
                            <div className="flex flex-wrap gap-1">
                              {item.images.map((url, idx) => (
                                <a key={idx} href={url} target="_blank" rel="noopener noreferrer">
                                  <img src={url} alt="" className="h-8 w-8 object-cover rounded border" />
                                </a>
                              ))}
                            </div>
                          ) : item.evidenceUrl ? (
                            <img
                              src={item.evidenceUrl}
                              alt="Evidence"
                              className="h-8 w-8 object-cover rounded border"
                            />
                          ) : (
                            "—"
                          )}
                        </TableCell>
                        <TableCell className={goodsDetailCellClass}>
                          <ViewCameraButton item={item} onView={setViewCamera} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <ViewLocatedCameraDialog target={viewCamera} onClose={() => setViewCamera(null)} />
              </div>
            ) : (
              <p className="p-4 text-sm text-[#607DA5]">No goods recorded.</p>
            )}
        </Section>

        <Section icon={<MapPin className="h-4 w-4" />} title="5. Location" tone="blue">
          <FieldGrid>
            <FieldTile label="Place of inspection" value={row.placeOfInspection} />
            <FieldTile label="Warehouse / shop" value={row.warehouseShop} />
            <FieldTile label="GPS location" value={row.gpsLocation} />
            <FieldTile label="Inspection date & time" value={row.inspectionDate} />
          </FieldGrid>
        </Section>

        <Section icon={<ClipboardList className="h-4 w-4" />} title="6. Grounds of suspicion" tone="amber">
          <NoteBlock tone="amber">{row.groundsOfSuspicion || "—"}</NoteBlock>
        </Section>

        <Section icon={<Paperclip className="h-4 w-4" />} title="7. Evidence collected" tone="teal">
            {row.evidenceCollected?.length ? (
              <ul className="space-y-1.5 text-sm text-[#102C55]">
                {row.evidenceCollected.map((e) => (
                  <li key={e} className="rounded-lg border border-[#D3F0E9] bg-[#F7FDFB] px-3 py-2">{e}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-[#607DA5]">None recorded.</p>
            )}
        </Section>

        <Section icon={<Paperclip className="h-4 w-4" />} title="8. Documents attached" tone="rose">
            {row.attachments?.length ? (
              <ul className="space-y-2 text-sm">
                {row.attachments.map((att) => (
                  <li key={att.id} className="rounded-lg border border-[#F8D7DF] bg-[#FFF9FA] px-3 py-2">
                    <a
                      href={att.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-[#0B5CAD] hover:underline"
                    >
                      {att.originalFilename || att.fileType}
                    </a>
                    <span className="ml-2 text-xs text-[#607DA5]">({att.fileType})</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-[#607DA5]">No attachments.</p>
            )}
        </Section>

        <Section icon={<Scale className="h-4 w-4" />} title="9. Preliminary findings" tone="blue">
          <NoteBlock>{row.preliminaryFindings || row.content || "—"}</NoteBlock>
        </Section>

        <Section icon={<Stamp className="h-4 w-4" />} title="10. Recommendation" tone="violet">
          <NoteBlock>{row.recommendation || "—"}</NoteBlock>
        </Section>

        <Section icon={<CheckCircle className="h-4 w-4" />} title="11. Approval workflow" tone="blue">
          <div className="space-y-4">
            <div>
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#1D6BB5]">Prepared by</p>
              <FieldGrid>
                <FieldTile label="Officer name" value={row.preparedBy} />
                <FieldTile label="Signature" value={row.preparedSignature} />
                <FieldTile label="Date" value={row.preparedDate} />
              </FieldGrid>
            </div>
            <div>
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#1D6BB5]">Forward to</p>
              <FieldGrid>
                <FieldTile
                  label="Approving officials"
                  value={
                    row.forwardTo?.trim() ||
                    "Assistant Collector, Deputy Collector, Location Admin, Super Admin"
                  }
                />
                <FieldTile label="Status" value={approvalStatusLabel} />
                <FieldTile label="Approval date" value={row.approvedAt} />
                <FieldTile label="Approving officer" value={row.approvedBy} />
              </FieldGrid>
            </div>
            {row.detentionMemoId && (
              <Link
                to={getDetentionMemoDetailPath(row.detentionMemoId)}
                className="inline-flex rounded-lg border border-[#C5DDF7] bg-[#EAF3FF] px-3 py-2 text-sm font-semibold text-[#0B5CAD] hover:bg-[#D7EBFF]"
              >
                Open linked detention memo
              </Link>
            )}
          </div>
        </Section>

        <Section icon={<ClipboardList className="h-4 w-4" />} title="12. Approval remarks" tone={row.status === "Rejected" ? "rose" : "amber"}>
            {row.status === "Rejected" && row.rejectionReason ? (
              <NoteBlock tone="rose">{row.rejectionReason}</NoteBlock>
            ) : (
              <NoteBlock tone="amber">{row.approvalRemarks || "—"}</NoteBlock>
            )}
        </Section>

        <Section icon={<History className="h-4 w-4" />} title="14. Audit log" tone="amber">
          <FieldGrid>
            <FieldTile label="Created by" value={row.createdBy} />
            <FieldTile label="Created on" value={row.createdAt} />
            <FieldTile label="Updated by" value={row.updatedBy} />
            <FieldTile label="Updated on" value={row.updatedAt} />
            <FieldTile label="Approved by" value={row.approvedBy} />
            <FieldTile label="Approval time" value={row.approvedAt} />
          </FieldGrid>
        </Section>

        {(row.status === "Draft" || row.status === "Rejected") && (
          <div className="sticky bottom-3 z-20 flex flex-wrap gap-2 rounded-xl border border-[#C5DDF7] bg-white/95 px-4 py-3 shadow-md backdrop-blur">
              {canEdit && (
                <Button variant="outline" className="border-[#C5DDF7] text-[#102C55]" asChild>
                  <Link to={getSeizureMgmtNoteSheetEditPath(row.id)}>
                    <Pencil className="mr-2 h-4 w-4" />
                    Edit
                  </Link>
                </Button>
              )}
              <Button className="bg-[#102C55] text-white hover:bg-[#0C2748]" disabled={acting} onClick={() => void runApproval("submit")}>
                <Send className="mr-2 h-4 w-4" />
                Send for Approval
              </Button>
          </div>
        )}

        {row.status === "Submitted" && canApprove && (
          <Section icon={<CheckCircle className="h-4 w-4" />} title="Senior officer decision" tone="blue">
            <div className="space-y-4">
              <div className="space-y-2 max-w-sm">
                <Label htmlFor="ns-approver">Approving Officer (you) <span className="text-red-600">*</span></Label>
                <Input
                  id="ns-approver"
                  value={approver}
                  onChange={(e) => {
                    setApprover(e.target.value)
                    if (invalidField === "ns-approver") setInvalidField("")
                  }}
                  aria-invalid={invalidField === "ns-approver"}
                  className="bg-muted"
                  readOnly
                />
                <p className="text-xs text-muted-foreground">
                  Filled from your login — not the preparing officer above.
                </p>
              </div>
              <div className="space-y-2">
                <Label>Approval Remarks</Label>
                <Textarea
                  rows={3}
                  value={approvalRemarks}
                  onChange={(e) => setApprovalRemarks(e.target.value)}
                  placeholder="e.g. Approved for issuance of Detention Memo under Section XX after preliminary examination."
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ns-rejection">Rejection Reason (if rejecting) <span className="text-red-600">*</span></Label>
                <Textarea
                  id="ns-rejection"
                  rows={3}
                  value={rejectionReason}
                  aria-invalid={invalidField === "ns-rejection"}
                  onChange={(e) => {
                    setRejectionReason(e.target.value)
                    if (invalidField === "ns-rejection") setInvalidField("")
                  }}
                  placeholder="e.g. Rejected because sufficient evidence has not been provided."
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button className="bg-[#0E9F8A] text-white hover:bg-[#0C7A68]" disabled={acting} onClick={() => void runApproval("approve")}>
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Approve
                </Button>
                <Button
                  variant="destructive"
                  disabled={acting}
                  onClick={() => void runApproval("reject")}
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  Reject
                </Button>
              </div>
            </div>
          </Section>
        )}

        {row.status === "Submitted" && !canApprove && (
          <div className="rounded-xl border border-[#F5D7A8] bg-[#FFF8EE] px-4 py-3 text-sm text-[#5C3B12]">
            Waiting for approval from{" "}
            <span className="font-semibold">
              Assistant Collector, Deputy Collector, Location Admin, or Super Admin
            </span>
            . Only those officials can approve or reject this note sheet.
          </div>
        )}

        {row.status === "Approved" && !row.detentionMemoId && (
          <div className="flex flex-col gap-3 rounded-xl border border-[#BFE8DF] bg-[#F3FBFA] px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-[#0C7A68]">
              Note sheet approved. You can now create the detention memo. Owner, goods, location, and grounds will be prefilled.
            </p>
            <Button className="bg-[#102C55] text-white hover:bg-[#0C2748]" asChild>
              <Link to={`${ROUTES.DETENTION_MEMO_CREATE}?noteSheetId=${encodeURIComponent(row.id)}`}>
                Create Detention Memo
              </Link>
            </Button>
          </div>
        )}
      </div>
    </ModulePageLayout>
  )
}
