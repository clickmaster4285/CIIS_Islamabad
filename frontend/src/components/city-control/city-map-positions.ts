import { GPS_DEFAULT_CENTER } from "@/components/gps/officer-gps-map"

/**
 * Where each location sits on the City Status Map when no office pin is set on
 * Locations & Sync: the city centre as Google Maps shows it. Islamabad (head
 * office) uses the GPS default centre. Set the exact office pin per location on Locations & Sync.
 */
export const CITY_CENTRES: Record<string, [number, number]> = {
  ISLAMABAD: GPS_DEFAULT_CENTER,
  // PESHAWAR: [34.008, 71.5789],  // Islamabad deployment
  // HAYATABAD: [33.9948, 71.4363],  // Islamabad deployment
  // MARDAN: [34.1986, 72.0404],  // Islamabad deployment
  // NOWSHERA: [34.0159, 71.9747],  // Islamabad deployment
  // KOHAT: [33.5869, 71.4429],  // Islamabad deployment
  // DI_KHAN: [31.8313, 70.9019],  // Islamabad deployment
  // ABBOTTABAD: [34.1688, 73.2215],  // Islamabad deployment
  // SWAT: [34.7717, 72.36],  // Islamabad deployment
  // THAKOT: [34.865, 72.9165],  // Islamabad deployment
}

/** Map position for a location: its office pin, else the city centre, else null (not plotted). */
export function mapPosition(code: string, lat?: number | null, lng?: number | null): [number, number] | null {
  if (lat != null && lng != null) return [lat, lng]
  return CITY_CENTRES[code.toUpperCase().replace(/[\s-]+/g, "_")] ?? null
}
