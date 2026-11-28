import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react"
import type { AttendanceFaceStatus } from "@/lib/staff-api"
import { cn } from "@/lib/utils"

/** Attendance face enrollment status (InsightFace embeddings built from the staff photos). */
export function AttendanceFaceBadge({
  face,
  className,
}: {
  face?: AttendanceFaceStatus | null
  className?: string
}) {
  const status = face?.status ?? "not_enrolled"
  const label = face?.label ?? "Not enrolled"
  const title = face
    ? `${label} — ${face.photos}/${face.required} usable photos${face.model ? ` · ${face.model}` : ""}`
    : "No attendance face enrollment"
  const style =
    status === "enrolled"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : status === "not_enrolled"
        ? "border-red-200 bg-red-50 text-red-700"
        : "border-amber-200 bg-amber-50 text-amber-800"
  const Icon = status === "enrolled" ? CheckCircle2 : status === "not_enrolled" ? XCircle : AlertTriangle
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium",
        style,
        className
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </span>
  )
}
