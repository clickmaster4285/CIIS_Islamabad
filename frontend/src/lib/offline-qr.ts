import QRCode from "qrcode"

/**
 * QR code as an <img>-ready SVG data URL, generated in the browser (no internet / api.qrserver.com).
 * Synchronous so it can replace the old remote URL builders one-for-one.
 */
export function qrImageUrl(data: string, size = 180): string {
  try {
    const qr = QRCode.create(data || " ", { errorCorrectionLevel: "M" })
    const count = qr.modules.size
    const margin = 1
    const total = count + margin * 2
    let path = ""
    for (let y = 0; y < count; y++) {
      for (let x = 0; x < count; x++) {
        if (qr.modules.get(x, y)) path += `M${x + margin} ${y + margin}h1v1h-1z`
      }
    }
    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${total} ${total}" shape-rendering="crispEdges">` +
      `<rect width="${total}" height="${total}" fill="#fff"/><path d="${path}" fill="#000"/></svg>`
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
  } catch {
    return ""
  }
}
