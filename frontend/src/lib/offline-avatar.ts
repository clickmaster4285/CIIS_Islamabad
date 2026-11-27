/** Initials avatar as an SVG data URL — replaces the internet-only i.pravatar.cc placeholder photos. */
const COLORS = ["#2563eb", "#7c3aed", "#db2777", "#dc2626", "#ea580c", "#16a34a", "#0891b2", "#4f46e5", "#0d9488", "#9333ea"]

export function initialsAvatarUrl(name: string, seed?: string): string {
  const words = (name || "").trim().split(/\s+/).filter(Boolean)
  const initials = ((words[0]?.[0] || "?") + (words.length > 1 ? words[words.length - 1][0] : "")).toUpperCase()
  const key = seed || name || "person"
  let hash = 0
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0
  const bg = COLORS[hash % COLORS.length]
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="150" height="150" viewBox="0 0 150 150">` +
    `<rect width="150" height="150" fill="${bg}"/>` +
    `<text x="50%" y="50%" dy=".35em" text-anchor="middle" fill="#fff" font-family="Inter, Segoe UI, sans-serif" font-size="60" font-weight="600">${initials.replace(/[<&>"]/g, "")}</text></svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}
