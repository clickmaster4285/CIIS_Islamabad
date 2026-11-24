import { useCallback, useEffect, useState } from "react"
import { fetchCameras, type CameraRecord } from "@/lib/cameras-api"

export function useCameras(options?: {
  activeOnly?: boolean
  onlineOnly?: boolean
  /** When true, only cameras assigned via Camera Distribution. */
  allocatedOnly?: boolean
}) {
  const [cameras, setCameras] = useState<CameraRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback((silent = false) => {
    if (!silent) {
      setLoading(true)
      setError(null)
    }
    fetchCameras({ allocatedOnly: options?.allocatedOnly === true })
      .then((rows) => {
        let list = rows
        if (options?.activeOnly) list = list.filter((c) => c.is_active)
        if (options?.onlineOnly) list = list.filter((c) => c.status === "Online")
        setCameras(list)
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load cameras"))
      .finally(() => setLoading(false))
  }, [options?.allocatedOnly, options?.activeOnly, options?.onlineOnly])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch; reload() flips `loading` before the async request
    reload()
    const onUpdate = () => reload()
    window.addEventListener("camera-integration-updated", onUpdate)
    // Quiet refresh so camera status (e.g. "Offline" from the offline monitor) stays current.
    const poll = window.setInterval(() => {
      if (document.visibilityState === "visible") reload(true)
    }, 30000)
    return () => {
      window.removeEventListener("camera-integration-updated", onUpdate)
      window.clearInterval(poll)
    }
  }, [reload])

  return { cameras, loading, error, reload }
}
