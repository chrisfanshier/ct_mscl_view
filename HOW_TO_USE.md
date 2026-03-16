# Core Summary Viewer — How to Use

A standalone Windows application for browsing, previewing, and exporting CT core scan images, with optional overlay of Geotek MSCL data.

No Python installation is required — just download and run the `.exe`.

---

## Download

Download **CoreSummaryViewer.exe** from this repository (`dist/CoreSummaryViewer.exe`).  
No installation needed. Place it anywhere and double-click to launch.

---

## Quick Start

1. **Launch** `CoreSummaryViewer.exe`.
2. Click **"1. Select Core Directory"** and choose the folder containing your `.tif` and `.xml` files.
3. The app scans the directory and populates the **Available Sections** list on the left. Check or uncheck sections to include/exclude them from the preview.
4. Click **"2. Export Selected to PDF"** to save a PDF summary of the checked sections.

---

## Interface Overview

### Left Panel

| Control | Description |
|---|---|
| **1. Select Core Directory** | Opens a folder browser. The app finds all paired `.tif`/`.xml` files and groups them by core ID and section number. |
| **Available Sections list** | Check/uncheck individual sections. The preview updates automatically. |
| **Add Full Depth Scale** | Toggles a depth scale bar on every section image. |
| **Scale mode dropdown** | Choose between a scale bar on each section, or a single continuous scale on the left margin. |
| **Load MSCL File (.out)** | Loads a Geotek MSCL `.out` data file for the core. |
| **Show MSCL Plot** | Overlays MSCL data as a plot column to the right of the core images (enabled after loading a file). |
| **Plot width slider** | Adjusts the horizontal width of the MSCL plot column. |
| **MSCL Columns list** | Check any MSCL data column to include it in the overlay. Optionally set custom min/max range per column. |
| **Text Size** | Font size (pt) used for section labels in the exported PDF. |
| **PDF Scale** | Scale factor applied to images in the exported PDF (50%–200%). |
| **Quality** | PDF image quality — Full Quality (PNG), Compact (JPEG 85%), or Draft (JPEG 50%, 2× downsample). |
| **2. Export Selected to PDF** | Exports checked sections to a PDF file. You choose the save location. |

### Preview Area

- **Scroll wheel** — zoom in/out.
- **Click and drag** — pan around the preview.
- The zoom level is shown at the bottom of the left panel.

---

## Data Requirements

- **Images:** 16-bit grayscale TIFF files (`.tif`). The app applies automatic contrast stretching (1st–99th percentile) and border cropping.
- **Metadata:** Each TIFF needs a paired XML file in the same directory. The XML must contain at minimum a `Section-ID` and `Core-ID` field. An optional `horizontal-resolution` (pixels/cm) field enables accurate physical depth rendering.
- **MSCL data:** A tab-delimited Geotek `.out` file with a `SB DEPTH`, `SECT NUM`, and `SECT DEPTH` column header row.

---

## Tips

- Sections are automatically sorted by section number; multiple image types per section (e.g., optical, X-ray) are each shown as a separate column.
- If XML metadata is missing for a section, the app will note it but still display the image where possible.
- For large cores, use **Draft** quality and a reduced **PDF Scale** to keep file sizes manageable.
- To check which XML files are missing, see the `missing xml/` folder for a log generated during scanning.
