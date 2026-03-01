#!/usr/bin/env python3
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# ── tuneable layout constants ────────────────────────────────────────────────
MARGIN           = 1.5 * cm
INTER_IMAGE_GAP  = 0.6 * cm
HEADER_HEIGHT    = 2.0 * cm   # space at top for page title
LABEL_HEIGHT     = 0.6 * cm  # space below each image for "Section N"

FONT_TITLE  = ("Helvetica-Bold", 14)
FONT_LABEL  = ("Helvetica", 8)

BG_COLOR    = (0.12, 0.12, 0.12)
COLOR_TITLE = (1.0, 1.0, 1.0)
COLOR_LABEL = (0.80, 0.80, 0.80)

CROP_COL_THRESH = 0.15
CROP_ROW_THRESH = 0.25
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_TYPE_NAMES = {
    "gixrayimage":        "X-Ray",
    "gictorthogonalview": "CT Orthogonal",
}

# ── image processing ─────────────────────────────────────────────────────────

def auto_crop(arr):
    max_val = arr.max()
    if max_val == 0: return arr
    col_means = arr.mean(axis=0)
    row_means = arr.mean(axis=1)
    left   = int(np.argmax(col_means > max_val * CROP_COL_THRESH))
    right  = int(len(col_means) - np.argmax(col_means[::-1] > max_val * CROP_COL_THRESH) - 1)
    top    = int(np.argmax(row_means > max_val * CROP_ROW_THRESH))
    bottom = int(len(row_means) - np.argmax(row_means[::-1] > max_val * CROP_ROW_THRESH) - 1)
    if right <= left or bottom <= top: return arr
    return arr[top:bottom + 1, left:right + 1]

def load_tif(tif_path):
    img = Image.open(tif_path)
    dpi_info = img.info.get("dpi")
    dpi = float(dpi_info[1]) if dpi_info else 300.0
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32)
    arr = auto_crop(arr)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr[:] = 0
    pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return pil_img, dpi

def img_to_buf(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ── Robust XML Parsing ────────────────────────────────────────────────────────

def parse_xml(xml_path):
    """Integrated robust XML parsing logic."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root_tag = root.tag

        def get_text(*tags):
            for tag in tags:
                # Use .// to find the tag anywhere in the tree
                el = root.find(f".//{tag}")
                if el is not None and el.text:
                    return el.text.strip()
            return None

        # Resolve Section ID/Number
        section_id = get_text("Section-ID", "section-id", "sectionID", "Section-Number", "section-number")
        
        # Resolve and Normalize Core ID
        core_name = get_text("Core-ID", "coreID")
        if core_name:
            core_name = core_name.strip().replace("_", "-").upper()
        else:
            core_name = "UNKNOWN-CORE"

        data = {
            "core_id":        core_name,
            "section_number": int(section_id) if (section_id and section_id.isdigit()) else 0,
            "image_type":     (get_text("Image-Type", "image-type") or "").lower(),
            "scan_id":        get_text("Scan-ID", "scanID"),
            "xml_root_tag":   root_tag
        }
        
        del tree
        return data
    except Exception as e:
        print(f"  [warn] Error parsing {xml_path}: {e}")
        return None

# ── file discovery ────────────────────────────────────────────────────────────

def find_tif_xml_pairs(root_dir):
    pairs = []
    for tif_path in Path(root_dir).rglob("*"):
        if not tif_path.is_file() or "corrected" not in tif_path.name.lower():
            continue
        if tif_path.suffix.lower() not in (".tif", ".tiff"):
            continue
        xml_path = None
        for candidate in tif_path.parent.iterdir():
            if (candidate.suffix.lower() == ".xml" and candidate.stem.lower() == tif_path.stem.lower()):
                xml_path = candidate
                break
        if xml_path:
            pairs.append((tif_path, xml_path))
    return pairs

def group_images(pairs):
    groups = defaultdict(list)
    for tif_path, xml_path in pairs:
        meta = parse_xml(xml_path)
        if not meta: continue
        
        key = (meta["core_id"], meta["image_type"])
        groups[key].append({
            "tif":     tif_path,
            "section": meta["section_number"],
            "meta":    meta,
        })
    for key in groups:
        groups[key].sort(key=lambda x: x["section"])
    return groups

# ── PDF page drawing ──────────────────────────────────────────────────────────

def draw_page(c_canvas, group_key, entries):
    core_id, image_type = group_key
    type_label = IMAGE_TYPE_NAMES.get(image_type, image_type.replace("-", " ").title())

    print(f"  Processing: {core_id} / {type_label} ({len(entries)} sections)")

    images = []
    for entry in entries:
        pil_img, dpi = load_tif(entry["tif"])
        px_w, px_h = pil_img.size
        pt_w = px_w * 72.0 / dpi
        pt_h = px_h * 72.0 / dpi
        images.append({
            "buf":     img_to_buf(pil_img),
            "pt_w":    pt_w,
            "pt_h":    pt_h,
            "section": entry["section"],
        })

    total_img_w = sum(im["pt_w"] for im in images)
    max_img_h   = max(im["pt_h"] for im in images)
    n_gaps      = max(len(images) - 1, 0)

    page_w = 2 * MARGIN + total_img_w + n_gaps * INTER_IMAGE_GAP
    page_h = 2 * MARGIN + HEADER_HEIGHT + LABEL_HEIGHT + max_img_h

    c_canvas.setPageSize((page_w, page_h))
    c_canvas.setFillColorRGB(*BG_COLOR)
    c_canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c_canvas.setFillColorRGB(*COLOR_TITLE)
    c_canvas.setFont(*FONT_TITLE)
    c_canvas.drawString(MARGIN, page_h - MARGIN - FONT_TITLE[1], f"{core_id}  |  {type_label}")

    img_top_y = page_h - MARGIN - HEADER_HEIGHT
    cur_x = MARGIN

    for im in images:
        img_y = img_top_y - im["pt_h"]
        im["buf"].seek(0)
        c_canvas.drawImage(ImageReader(im["buf"]), cur_x, img_y, width=im["pt_w"], height=im["pt_h"])
        c_canvas.setFillColorRGB(*COLOR_LABEL)
        c_canvas.setFont(*FONT_LABEL)
        c_canvas.drawCentredString(cur_x + im["pt_w"] / 2, MARGIN, f"Section {im['section']}")
        cur_x += im["pt_w"] + INTER_IMAGE_GAP

def build_pdf(root_dir, output_pdf):
    print(f"\nScanning: {root_dir}")
    pairs = find_tif_xml_pairs(root_dir)
    if not pairs:
        print("No matching pairs found.")
        return
    groups = group_images(pairs)
    c = canvas.Canvas(output_pdf)
    for group_key in sorted(groups.keys()):
        draw_page(c, group_key, groups[group_key])
        c.showPage()
    c.save()
    print(f"\nPDF written to: {output_pdf}")

if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_name = sys.argv[2] if len(sys.argv) > 2 else "core_layout.pdf"
    build_pdf(root_dir, out_name)