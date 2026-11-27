declare module "qrcode" {
  export interface QRCodeToDataURLOptions {
    width?: number
    margin?: number
    color?: { dark?: string; light?: string }
    errorCorrectionLevel?: "L" | "M" | "Q" | "H"
  }
  export function toDataURL(
    text: string,
    options?: QRCodeToDataURLOptions
  ): Promise<string>
  export interface QRCodeModel {
    modules: { size: number; get(col: number, row: number): boolean | number }
  }
  export function create(
    text: string,
    options?: { errorCorrectionLevel?: "L" | "M" | "Q" | "H" }
  ): QRCodeModel
  const QRCode: {
    toDataURL: typeof toDataURL
    create: typeof create
  }
  export default QRCode
}
