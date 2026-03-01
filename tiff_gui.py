import sys
import io
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from PIL import Image

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                             QListWidgetItem, QLabel, QScrollArea, QCheckBox, QFrame,
                             QComboBox, QSlider, QSizePolicy, QSpinBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QWheelEvent

from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ── Layout Constants ─────────────────────────────────────────────────────────
MARGIN           = 1.5 * cm
INTER_IMAGE_GAP  = 0.6 * cm
HEADER_HEIGHT    = 2.0 * cm
LABEL_HEIGHT     = 0.6 * cm
BG_COLOR         = (0.12, 0.12, 0.12)
CROP_COL_THRESH  = 0.15
CROP_ROW_THRESH  = 0.25

FALLBACK_PX_PER_CM = 35.0   # used only if XML has no pixels-per-CM
CM_TO_PT           = 28.3465  # 1 cm in PDF points

MSCL_COLORS_PIL = [
    (80, 190, 255), (255, 100, 100), (100, 255, 100), (255, 200, 50),
    (200, 100, 255), (255, 150, 50), (50, 255, 200), (255, 100, 200),
]
MSCL_COLORS_PDF = [
    (0.31, 0.75, 1.0), (1.0, 0.39, 0.39), (0.39, 1.0, 0.39), (1.0, 0.78, 0.2),
    (0.78, 0.39, 1.0), (1.0, 0.59, 0.2), (0.2, 1.0, 0.78), (1.0, 0.39, 0.78),
]

# ── Scientific Image Logic ──────────────────────────────────────────────────

def auto_crop(arr):
    max_val = arr.max()
    if max_val == 0:
        return arr, 0
    col_means = arr.mean(axis=0)
    row_means = arr.mean(axis=1)
    left   = int(np.argmax(col_means > max_val * CROP_COL_THRESH))
    right  = int(len(col_means) - np.argmax(col_means[::-1] > max_val * CROP_COL_THRESH) - 1)
    top    = int(np.argmax(row_means > max_val * CROP_ROW_THRESH))
    bottom = int(len(row_means) - np.argmax(row_means[::-1] > max_val * CROP_ROW_THRESH) - 1)
    if right > left and bottom > top:
        return arr[top:bottom + 1, left:right + 1], top
    return arr, 0


def load_tif_scientific(tif_path, px_per_cm=FALLBACK_PX_PER_CM):
    """
    Load a TIF, auto-crop, contrast-stretch.
    Returns (PIL image, px_per_cm, crop_top_px).

    Scale comes from px_per_cm (parsed from XML), NOT the DPI tag.
    The DPI tag in these Geotek scanner files is 300 — a hardware value
    that corresponds to ~45 cm for a 151 cm core, not the true physical scale.
    """
    img = Image.open(tif_path)
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32)
    arr, crop_top_px = auto_crop(arr)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr[:] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L"), px_per_cm, crop_top_px

# ── Metadata Logic ──────────────────────────────────────────────────────────

def parse_xml_robust(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        def get_text(*tags):
            for tag in tags:
                el = root.find(f".//{tag}")
                if el is not None and el.text:
                    return el.text.strip()
            return None

        sec_id    = get_text("Section-ID", "section-id", "sectionID", "Section-Number", "section-number")
        core_name = get_text("Core-ID", "coreID")
        core_name = core_name.strip().replace("_", "-").upper() if core_name else "UNKNOWN"

        # pixels-per-CM is the authoritative spatial scale for these files
        px_per_cm_raw = get_text("pixels-per-CM", "pixels-per-cm", "PixelsPerCM")
        try:
            px_per_cm = float(px_per_cm_raw) if px_per_cm_raw else FALLBACK_PX_PER_CM
        except ValueError:
            px_per_cm = FALLBACK_PX_PER_CM

        return {
            "core_id":        core_name,
            "section_number": int(sec_id) if (sec_id and sec_id.isdigit()) else 0,
            "image_type":     (get_text("Image-Type", "image-type") or "Unknown").lower(),
            "px_per_cm":      px_per_cm,
        }
    except:
        return None

# ── MSCL Data Parsing ────────────────────────────────────────────────────────

def parse_mscl_file(filepath):
    """Parse a Geotek MSCL .out file.
    Returns (section_data, data_columns) where:
      section_data: {section_num: {'SECT DEPTH': np.array, col: np.array, ...}}
      data_columns: list of plottable column names
    """
    lines = Path(filepath).read_text(encoding='utf-8', errors='replace').splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("SB DEPTH"):
            header_idx = i
            break
    if header_idx is None:
        return None, []

    headers = [h.strip() for h in lines[header_idx].split('\t')]
    data_start = header_idx + 2  # skip units row

    skip_cols = {"SB DEPTH", "SECT NUM", "SECT DEPTH"}
    data_columns = [h for h in headers
                    if h and h.upper() not in {s.upper() for s in skip_cols}]

    sect_num_idx = next((i for i, h in enumerate(headers)
                         if h.upper() == "SECT NUM"), None)
    sect_depth_idx = next((i for i, h in enumerate(headers)
                           if h.upper() == "SECT DEPTH"), None)
    if sect_num_idx is None or sect_depth_idx is None:
        return None, []

    section_data = defaultdict(lambda: {c: [] for c in ["SECT DEPTH"] + data_columns})

    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < len(headers):
            continue
        raw_sec = parts[sect_num_idx].strip()
        if "cal" in raw_sec.lower():
            continue
        try:
            sec_num = int(raw_sec)
        except ValueError:
            continue
        try:
            sect_depth = float(parts[sect_depth_idx].strip())
        except ValueError:
            continue

        section_data[sec_num]["SECT DEPTH"].append(sect_depth)
        for col in data_columns:
            idx = headers.index(col)
            try:
                section_data[sec_num][col].append(float(parts[idx].strip()))
            except (ValueError, IndexError):
                section_data[sec_num][col].append(float('nan'))

    for sec in section_data:
        for col in section_data[sec]:
            section_data[sec][col] = np.array(section_data[sec][col])

    return dict(section_data), data_columns

# ── Application Components ──────────────────────────────────────────────────

class ScanWorker(QThread):
    finished = Signal(list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def run(self):
        pairs = []
        for tif_path in Path(self.directory).rglob("*"):
            if "corrected" in tif_path.name.lower() and tif_path.suffix.lower() in (".tif", ".tiff"):
                xml_path = tif_path.with_suffix('.xml')
                if xml_path.exists():
                    meta = parse_xml_robust(xml_path)
                    if meta:
                        pairs.append({"tif": tif_path, "meta": meta})
        self.finished.emit(pairs)


class ZoomableScrollArea(QScrollArea):
    """QScrollArea with mouse-wheel zoom and click-drag panning."""
    zoom_changed = Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._drag_start = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        self.viewport().setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self.zoom_changed.emit(delta)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._scroll_start_h = self.horizontalScrollBar().value()
            self._scroll_start_v = self.verticalScrollBar().value()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None:
            delta = event.position().toPoint() - self._drag_start
            self.horizontalScrollBar().setValue(self._scroll_start_h - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_v - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CoreCollectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Core Summary Viewer")
        self.resize(1300, 850)
        self.found_items = []
        self.zoom_factor = 1.0

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        self.btn_browse = QPushButton("1. Select Core Directory")
        self.btn_browse.clicked.connect(self.browse_directory)
        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self.update_previews)

        # ── Scale bar controls ───────────────────────────────────────────
        self.chk_scale_bar = QCheckBox("Add Full Depth Scale")
        self.chk_scale_bar.setChecked(True)
        self.chk_scale_bar.stateChanged.connect(self.update_previews)

        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["One scale per section", "Single scale on left"])
        self.scale_mode_combo.setCurrentIndex(0)
        self.scale_mode_combo.currentIndexChanged.connect(self.update_previews)

        # ── MSCL controls ────────────────────────────────────────────────
        self.mscl_data = None
        self.mscl_columns = []
        self.btn_mscl = QPushButton("Load MSCL File (.out)")
        self.btn_mscl.clicked.connect(self.load_mscl_file)
        self.chk_mscl = QCheckBox("Show MSCL Plot")
        self.chk_mscl.setChecked(False)
        self.chk_mscl.setEnabled(False)
        self.chk_mscl.stateChanged.connect(self.update_previews)

        self.mscl_width_label = QLabel("Plot width: 80")
        self.mscl_width_slider = QSlider(Qt.Horizontal)
        self.mscl_width_slider.setRange(30, 300)
        self.mscl_width_slider.setValue(80)
        self.mscl_width_slider.setEnabled(False)
        self.mscl_width_slider.valueChanged.connect(self._on_mscl_width_changed)

        self.mscl_col_list = QListWidget()
        self.mscl_col_list.setMaximumHeight(130)
        self.mscl_col_list.setEnabled(False)
        self.mscl_col_list.itemChanged.connect(self.update_previews)

        # ── Export options ────────────────────────────────────────────────
        self.pdf_text_size_spin = QSpinBox()
        self.pdf_text_size_spin.setRange(6, 36)
        self.pdf_text_size_spin.setValue(14)
        self.pdf_text_size_spin.setSuffix(" pt")

        self.pdf_scale_combo = QComboBox()
        self.pdf_scale_combo.addItems(["50%", "75%", "100%", "150%", "200%"])
        self.pdf_scale_combo.setCurrentText("100%")

        self.pdf_quality_combo = QComboBox()
        self.pdf_quality_combo.addItems([
            "Full Quality (PNG)",
            "Compact (JPEG 85%)",
            "Draft (JPEG 50%, downsample 2\u00d7)",
        ])
        self.pdf_quality_combo.setCurrentIndex(0)

        # ── Zoom label (wheel to zoom, click-drag to pan) ────────────────
        self.zoom_label = QLabel("Zoom: 100%  (scroll to zoom, drag to pan)")

        self.btn_export = QPushButton("2. Export Selected to PDF")
        self.btn_export.clicked.connect(self.export_pdf)
        self.btn_export.setEnabled(False)

        left_panel.addWidget(self.btn_browse)
        left_panel.addWidget(QLabel("Available Sections:"))
        left_panel.addWidget(self.list_widget)
        left_panel.addWidget(self.chk_scale_bar)
        left_panel.addWidget(self.scale_mode_combo)
        left_panel.addWidget(QLabel(""))
        left_panel.addWidget(self.btn_mscl)
        left_panel.addWidget(self.chk_mscl)
        mscl_w_row = QHBoxLayout()
        mscl_w_row.addWidget(self.mscl_width_label)
        mscl_w_row.addWidget(self.mscl_width_slider)
        left_panel.addLayout(mscl_w_row)
        left_panel.addWidget(QLabel("MSCL Columns:"))
        left_panel.addWidget(self.mscl_col_list)
        left_panel.addWidget(QLabel(""))
        exp_lbl = QLabel("\u2500\u2500 Export Options \u2500\u2500")
        exp_lbl.setStyleSheet("font-weight: bold;")
        left_panel.addWidget(exp_lbl)
        ts_row = QHBoxLayout()
        ts_row.addWidget(QLabel("Text Size:"))
        ts_row.addWidget(self.pdf_text_size_spin)
        left_panel.addLayout(ts_row)
        sc_row = QHBoxLayout()
        sc_row.addWidget(QLabel("PDF Scale:"))
        sc_row.addWidget(self.pdf_scale_combo)
        left_panel.addLayout(sc_row)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Quality:"))
        q_row.addWidget(self.pdf_quality_combo)
        left_panel.addLayout(q_row)
        left_panel.addWidget(self.btn_export)
        left_panel.addWidget(self.zoom_label)

        self.preview_area = ZoomableScrollArea()
        self.preview_area.zoom_changed.connect(self._on_wheel_zoom)
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_area.setWidget(self.preview_container)
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setStyleSheet("background-color: #1a1a1a;")

        layout.addLayout(left_panel, 1)
        layout.addWidget(self.preview_area, 4)

    # ── Zoom helpers ──────────────────────────────────────────────────────
    def _set_zoom(self, factor):
        self.zoom_factor = max(0.25, min(factor, 5.0))
        self.zoom_label.setText(f"Zoom: {int(self.zoom_factor * 100)}%")
        self.update_previews()

    def _on_wheel_zoom(self, delta):
        step = 0.15 if delta > 0 else -0.15
        self._set_zoom(self.zoom_factor + step)

    def _on_mscl_width_changed(self, val):
        self.mscl_width_label.setText(f"Plot width: {val}")
        self.update_previews()

    def get_selected_mscl_columns(self):
        """Return list of checked column names from the MSCL column list."""
        cols = []
        for i in range(self.mscl_col_list.count()):
            item = self.mscl_col_list.item(i)
            if item.checkState() == Qt.Checked:
                cols.append(item.text())
        return cols

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if dir_path:
            self.list_widget.clear()
            self.worker = ScanWorker(dir_path)
            self.worker.finished.connect(self.on_scan_finished)
            self.worker.start()

    def on_scan_finished(self, results):
        self.found_items = results
        for item in results:
            label = f"{item['meta']['core_id']} | Sec {item['meta']['section_number']} | {item['meta']['image_type']}"
            list_item = QListWidgetItem(label)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(list_item)
        self.btn_export.setEnabled(True)

    def load_mscl_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select MSCL .out File", "", "MSCL Files (*.out);;All Files (*)")
        if not filepath:
            return
        data, columns = parse_mscl_file(filepath)
        if data is None:
            self.mscl_data = None
            self.mscl_columns = []
            self.chk_mscl.setEnabled(False)
            self.mscl_col_list.setEnabled(False)
            self.mscl_width_slider.setEnabled(False)
            return
        self.mscl_data = data
        self.mscl_columns = columns
        self.mscl_col_list.blockSignals(True)
        self.mscl_col_list.clear()
        for col in columns:
            li = QListWidgetItem(col)
            li.setFlags(li.flags() | Qt.ItemIsUserCheckable)
            li.setCheckState(Qt.Checked if col == "Den1" else Qt.Unchecked)
            self.mscl_col_list.addItem(li)
        self.mscl_col_list.blockSignals(False)
        self.chk_mscl.setEnabled(True)
        self.mscl_col_list.setEnabled(True)
        self.mscl_width_slider.setEnabled(True)
        self.chk_mscl.setChecked(True)
        self.btn_mscl.setText(f"MSCL: {Path(filepath).name}")
        self.update_previews()

    def update_previews(self):
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        checked = [self.found_items[i] for i in range(self.list_widget.count())
                   if self.list_widget.item(i).checkState() == Qt.Checked]
        checked.sort(key=lambda x: x['meta']['section_number'])
        if not checked:
            return

        show_scale = self.chk_scale_bar.isChecked()
        per_section = self.scale_mode_combo.currentIndex() == 0  # 0 = per section

        # "Single scale on left" — build a standalone scale bar from the
        # tallest selected section and insert it as the first widget
        if show_scale and not per_section:
            self._add_standalone_scale(checked)

        for item in checked:
            draw_scale = show_scale and per_section
            self.add_thumbnail(item, draw_scale=draw_scale)

    # ── Standalone scale bar ("single scale on left" mode) ────────────────
    def _add_standalone_scale(self, checked_items):
        """Create a standalone depth-scale widget to the left of all thumbnails."""
        from PIL import ImageDraw, ImageFont, Image as PILImage

        # Find the tallest section to size the scale bar
        tallest, tallest_h_px = None, 0
        for item in checked_items:
            pil_img, _, crop_top_px = load_tif_scientific(
                item['tif'], item['meta']['px_per_cm'])
            BASE_IMG_W = 150
            PREVIEW_IMG_W = int(BASE_IMG_W * self.zoom_factor)
            scale = PREVIEW_IMG_W / pil_img.size[0]
            preview_h = int(pil_img.size[1] * scale)
            if preview_h > tallest_h_px:
                tallest_h_px = preview_h
                tallest = {
                    'phys_h_cm':   pil_img.size[1] / item['meta']['px_per_cm'],
                    'crop_top_cm': crop_top_px / item['meta']['px_per_cm'],
                }
        if tallest is None:
            return

        phys_h_cm   = tallest['phys_h_cm']
        crop_top_cm = tallest['crop_top_cm']
        preview_h   = tallest_h_px
        px_per_cm_preview = preview_h / phys_h_cm if phys_h_cm else 1

        SCALEBAR_W = int(45 * self.zoom_factor)
        LABEL_H_PX = max(20, int(28 * self.zoom_factor))
        total_h = preview_h + LABEL_H_PX

        bar_img = PILImage.new("RGB", (SCALEBAR_W, total_h), color=(31, 31, 31))
        draw = ImageDraw.Draw(bar_img)

        bar_x = SCALEBAR_W - 2
        draw.line([(bar_x, 0), (bar_x, preview_h)], fill=(200, 200, 200), width=1)

        font_size = 11
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        first_tick_cm = (int(crop_top_cm / 10) + 1) * 10
        tick_cm = first_tick_cm
        while True:
            offset_cm = tick_cm - crop_top_cm
            if offset_cm > phys_h_cm:
                break
            tick_y = int(offset_cm * px_per_cm_preview)
            if tick_cm % 10 == 0:
                draw.line([(bar_x - 5, tick_y), (bar_x, tick_y)], fill=(220, 220, 220), width=1)
                draw.text((2, tick_y - 6), str(tick_cm), fill=(200, 200, 200), font=font)
            elif tick_cm % 5 == 0:
                draw.line([(bar_x - 4, tick_y), (bar_x, tick_y)], fill=(170, 170, 170), width=1)
            tick_cm += 1

        img_data = bar_img.tobytes()
        qimg = QImage(img_data, SCALEBAR_W, total_h, SCALEBAR_W * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        lbl = QLabel()
        lbl.setPixmap(pix)
        self.preview_layout.addWidget(lbl, 0, Qt.AlignTop)

    def _render_mscl_plot_pil(self, series_list, plot_w, plot_h,
                               crop_top_cm, phys_h_cm):
        """Render one or more MSCL series on a vertical line plot (PIL image).
        series_list: [(col_name, depths, values, color_rgb), ...]
        """
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", (plot_w, plot_h), color=(25, 25, 35))
        draw = ImageDraw.Draw(img)

        PAD_X = 3
        usable_w = plot_w - 2 * PAD_X

        # Axis line
        draw.line([(PAD_X, 0), (PAD_X, plot_h - 1)], fill=(60, 60, 60), width=1)

        font_size = 8
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        names = []
        for col_name, depths, values, color in series_list:
            v_lo, v_hi = np.percentile(values, (2, 98))
            if v_hi <= v_lo:
                v_lo, v_hi = values.min(), values.max()
            if v_hi <= v_lo:
                continue

            points = []
            for i in range(len(depths)):
                py = (depths[i] - crop_top_cm) / phys_h_cm * plot_h
                if py < -10 or py > plot_h + 10:
                    points.append(None)
                    continue
                py = max(0, min(plot_h - 1, py))
                v_clamped = max(v_lo, min(v_hi, values[i]))
                px = PAD_X + int((v_clamped - v_lo) / (v_hi - v_lo)
                                 * max(1, usable_w - 1))
                points.append((int(px), int(py)))

            prev = None
            for pt in points:
                if pt is not None and prev is not None:
                    draw.line([prev, pt], fill=color, width=1)
                prev = pt if pt is not None else None

            names.append((col_name, color))

        # Legend at bottom
        label_y = plot_h - font_size - 2
        lx = 2
        for name, color in names:
            draw.text((lx, label_y), name, fill=color, font=font)
            bbox = draw.textbbox((0, 0), name + "  ", font=font)
            lx += bbox[2] - bbox[0]

        return img

    def add_thumbnail(self, data, draw_scale=True):
        try:
            from PIL import ImageDraw, ImageFont

            px_per_cm    = data['meta']['px_per_cm']
            section_num  = data['meta']['section_number']
            pil_img, _, crop_top_px = load_tif_scientific(data['tif'], px_per_cm)

            # Scale image to preview width (affected by zoom)
            BASE_IMG_W   = 150
            PREVIEW_IMG_W = int(BASE_IMG_W * self.zoom_factor)
            scale = PREVIEW_IMG_W / pil_img.size[0]
            preview_h = int(pil_img.size[1] * scale)
            resized = pil_img.resize((PREVIEW_IMG_W, preview_h), Image.Resampling.LANCZOS)

            # Physical dimensions at preview scale
            phys_h_cm   = pil_img.size[1] / px_per_cm
            crop_top_cm = crop_top_px / px_per_cm
            px_per_cm_preview = preview_h / phys_h_cm   # preview pixels per cm

            # ── MSCL plot (if enabled) ───────────────────────────────────────
            mscl_plot = None
            MSCL_W = 0
            selected_cols = self.get_selected_mscl_columns()
            show_mscl = (self.mscl_data is not None
                         and self.chk_mscl.isChecked()
                         and len(selected_cols) > 0)
            if show_mscl:
                sec_data = self.mscl_data.get(section_num)
                if sec_data and "SECT DEPTH" in sec_data:
                    series_list = []
                    for ci, col_name in enumerate(selected_cols):
                        if col_name not in sec_data:
                            continue
                        depths = sec_data["SECT DEPTH"]
                        values = sec_data[col_name]
                        mask = ~np.isnan(values)
                        if mask.sum() >= 2:
                            color = MSCL_COLORS_PIL[ci % len(MSCL_COLORS_PIL)]
                            series_list.append((col_name, depths[mask],
                                                values[mask], color))
                    if series_list:
                        MSCL_W = int(self.mscl_width_slider.value()
                                     * self.zoom_factor)
                        mscl_plot = self._render_mscl_plot_pil(
                            series_list, MSCL_W, preview_h,
                            crop_top_cm, phys_h_cm)

            # Label strip below image
            LABEL_H_PX  = max(20, int(28 * self.zoom_factor))
            SCALEBAR_W  = int(45 * self.zoom_factor) if draw_scale else 0

            # Compose full thumbnail: mscl plot, scale bar, image, label
            total_w = MSCL_W + SCALEBAR_W + PREVIEW_IMG_W
            total_h = preview_h + LABEL_H_PX
            thumb = Image.new("RGB", (total_w, total_h), color=(31, 31, 31))

            if mscl_plot:
                thumb.paste(mscl_plot, (0, 0))

            # Paste greyscale image as RGB
            thumb.paste(resized.convert("RGB"), (MSCL_W + SCALEBAR_W, 0))

            draw = ImageDraw.Draw(thumb)

            # ── scale bar (only if requested) ────────────────────────────────
            if draw_scale:
                bar_x = MSCL_W + SCALEBAR_W - 2   # right edge of bar column
                draw.line([(bar_x, 0), (bar_x, preview_h)], fill=(200, 200, 200), width=1)

                font_size = 11
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()

                first_tick_cm = (int(crop_top_cm / 10) + 1) * 10
                tick_cm = first_tick_cm
                while True:
                    offset_cm = tick_cm - crop_top_cm
                    if offset_cm > phys_h_cm:
                        break
                    tick_y = int(offset_cm * px_per_cm_preview)
                    if tick_cm % 10 == 0:
                        draw.line([(bar_x - 5, tick_y), (bar_x, tick_y)], fill=(220, 220, 220), width=1)
                        draw.text((MSCL_W + 2, tick_y - 6), str(tick_cm), fill=(200, 200, 200), font=font)
                    elif tick_cm % 5 == 0:
                        draw.line([(bar_x - 4, tick_y), (bar_x, tick_y)], fill=(170, 170, 170), width=1)
                    tick_cm += 1

            # ── section label ────────────────────────────────────────────────
            label_font_size = 12
            try:
                font_label = ImageFont.truetype("arialbd.ttf", label_font_size)
            except Exception:
                try:
                    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", label_font_size)
                except Exception:
                    font_label = ImageFont.load_default()
            label_text = f"Section {section_num}"
            bbox = draw.textbbox((0, 0), label_text, font=font_label)
            text_w = bbox[2] - bbox[0]
            label_x = MSCL_W + SCALEBAR_W + (PREVIEW_IMG_W - text_w) // 2
            draw.text((label_x, preview_h + 4),
                      label_text, fill=(200, 200, 200), font=font_label)

            # ── convert to QPixmap ───────────────────────────────────────────
            img_data = thumb.tobytes()
            qimg = QImage(img_data, total_w, total_h, total_w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFrameShape(QFrame.Panel)
            self.preview_layout.addWidget(lbl, 0, Qt.AlignTop)
        except Exception as e:
            print(f"Thumbnail error: {e}")

    def export_pdf(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF (*.pdf)")
        if not save_path:
            return
        selected = [self.found_items[i] for i in range(self.list_widget.count())
                    if self.list_widget.item(i).checkState() == Qt.Checked]
        if not selected:
            return
        groups = defaultdict(list)
        for item in selected:
            groups[(item['meta']['core_id'], item['meta']['image_type'])].append(item)
        c = canvas.Canvas(save_path)
        for key in sorted(groups.keys()):
            self.generate_page(c, key, sorted(groups[key], key=lambda x: x['meta']['section_number']))
            c.showPage()
        c.save()

    def draw_scale_bar(self, c, x, top_y, total_pt_height, phys_h_cm, crop_top_cm, font_size):
        """
        Draw a vertical scale bar.

        x             : x position of the bar line
        top_y         : PDF y-coordinate of the top of the images
        total_pt_height: height of the tallest image in points
        phys_h_cm     : physical height of the tallest image in cm
        crop_top_cm   : cm from the physical section top to the crop edge,
                        so tick labels reflect true depth within the section
        font_size     : font size for tick labels
        """
        pt_per_cm = total_pt_height / phys_h_cm

        c.setStrokeColorRGB(1, 1, 1)
        c.setLineWidth(1.2)
        c.line(x, top_y, x, top_y - total_pt_height)

        # First tick at the nearest 10 cm boundary at or below crop_top_cm
        first_tick_cm = (int(crop_top_cm / 10) + 1) * 10
        tick_cm = first_tick_cm

        while True:
            offset_cm = tick_cm - crop_top_cm
            if offset_cm > phys_h_cm:
                break
            tick_y = top_y - (offset_cm * pt_per_cm)

            if tick_cm % 10 == 0:
                c.setLineWidth(1.5)
                c.line(x, tick_y, x + 8, tick_y)
                c.setFillColorRGB(1, 1, 1)
                c.setFont("Helvetica-Bold", font_size)
                c.drawString(x + 12, tick_y - 3, f"{tick_cm} cm")
            elif tick_cm % 5 == 0:
                c.setLineWidth(1.0)
                c.line(x, tick_y, x + 5, tick_y)
            else:
                c.setLineWidth(0.5)
                c.line(x, tick_y, x + 3, tick_y)

            tick_cm += 1

    def _draw_mscl_plot_pdf(self, c, x, top_y, plot_w_pt, plot_h_pt,
                             section_num, phys_h_cm, crop_top_cm):
        """Draw one or more MSCL series on the PDF canvas."""
        selected_cols = self.get_selected_mscl_columns()
        if not selected_cols or self.mscl_data is None:
            return
        sec_data = self.mscl_data.get(section_num)
        if not sec_data or "SECT DEPTH" not in sec_data:
            return

        pt_per_cm = plot_h_pt / phys_h_cm if phys_h_cm else 1
        PAD = 3
        usable_w = plot_w_pt - 2 * PAD

        c.setStrokeColorRGB(0.25, 0.25, 0.25)
        c.setLineWidth(0.5)
        c.line(x + PAD, top_y, x + PAD, top_y - plot_h_pt)

        names_colors = []
        for ci, col_name in enumerate(selected_cols):
            if col_name not in sec_data:
                continue
            depths = sec_data["SECT DEPTH"]
            values = sec_data[col_name]
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                continue
            d, v = depths[mask], values[mask]
            v_lo, v_hi = np.percentile(v, (2, 98))
            if v_hi <= v_lo:
                v_lo, v_hi = v.min(), v.max()
            if v_hi <= v_lo:
                continue

            r, g, b = MSCL_COLORS_PDF[ci % len(MSCL_COLORS_PDF)]
            c.setStrokeColorRGB(r, g, b)
            c.setLineWidth(0.8)
            prev = None
            for i in range(len(d)):
                py = top_y - ((d[i] - crop_top_cm) * pt_per_cm)
                if py > top_y + 5 or py < top_y - plot_h_pt - 5:
                    prev = None
                    continue
                py = max(top_y - plot_h_pt, min(top_y, py))
                v_clamped = max(v_lo, min(v_hi, v[i]))
                px = x + PAD + ((v_clamped - v_lo) / (v_hi - v_lo)) * usable_w
                if prev is not None:
                    c.line(prev[0], prev[1], px, py)
                prev = (px, py)

            names_colors.append((col_name, (r, g, b)))

        # Legend at bottom
        c.setFont("Helvetica", 8)
        lx = x + 1
        for name, (r, g, b) in names_colors:
            c.setFillColorRGB(r, g, b)
            c.drawString(lx, top_y - plot_h_pt - 10, name)
            lx += c.stringWidth(name, "Helvetica", 8) + 4

    def generate_page(self, c, key, items):
        core_id, img_type = key
        processed_images = []

        # Export settings
        scale_pct = int(self.pdf_scale_combo.currentText().replace('%', ''))
        pdf_scale = scale_pct / 100.0
        quality_idx = self.pdf_quality_combo.currentIndex()
        title_font_size = self.pdf_text_size_spin.value()

        for item in items:
            px_per_cm = item['meta']['px_per_cm']
            pil_img, _, crop_top_px = load_tif_scientific(item['tif'], px_per_cm)
            px_w, px_h = pil_img.size

            # True physical dimensions from scanner pixel density, scaled
            pt_w = (px_w / px_per_cm) * CM_TO_PT * pdf_scale
            pt_h = (px_h / px_per_cm) * CM_TO_PT * pdf_scale
            phys_h_cm   = px_h / px_per_cm
            crop_top_cm = crop_top_px / px_per_cm

            buf = io.BytesIO()
            if quality_idx == 0:
                pil_img.save(buf, format="PNG")
            elif quality_idx == 1:
                pil_img.save(buf, format="JPEG", quality=85)
            else:
                ds = pil_img.resize((max(1, px_w // 2), max(1, px_h // 2)),
                                    Image.Resampling.LANCZOS)
                ds.save(buf, format="JPEG", quality=50)
            processed_images.append({
                'buf':          buf,
                'pt_w':         pt_w,
                'pt_h':         pt_h,
                'phys_h_cm':    phys_h_cm,
                'crop_top_cm':  crop_top_cm,
                'sec':          item['meta']['section_number'],
            })

        max_h        = max(im['pt_h'] for im in processed_images)
        # Use the tallest image's physical height and crop offset for the scale bar
        tallest      = max(processed_images, key=lambda im: im['pt_h'])
        phys_h_cm    = tallest['phys_h_cm']
        crop_top_cm  = tallest['crop_top_cm']

        show_mscl_pdf = (self.mscl_data is not None
                         and self.chk_mscl.isChecked()
                         and len(self.get_selected_mscl_columns()) > 0)
        # MSCL width: slider value / 80 gives ratio relative to 2cm baseline
        # This makes slider value of 80 → 2cm, 160 → 4cm, etc.
        mscl_base_w  = self.mscl_width_slider.value()
        MSCL_PLOT_W  = ((mscl_base_w / 80.0) * 2.0 * cm * pdf_scale if show_mscl_pdf else 0)

        total_w      = (sum(im['pt_w'] + MSCL_PLOT_W for im in processed_images)
                        + (len(processed_images) - 1) * INTER_IMAGE_GAP)

        SCALE_WIDTH  = 1.5 * cm
        page_w       = SCALE_WIDTH + MARGIN + total_w
        page_h       = 2 * MARGIN + HEADER_HEIGHT + LABEL_HEIGHT + max_h

        c.setPageSize((page_w, page_h))
        c.setFillColorRGB(*BG_COLOR)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        top_y = page_h - MARGIN - HEADER_HEIGHT

        scale_bar_font_size = max(8, title_font_size - 3)
        if self.chk_scale_bar.isChecked():
            self.draw_scale_bar(c, MARGIN / 2, top_y, max_h, phys_h_cm,
                               crop_top_cm, scale_bar_font_size)

        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", title_font_size)
        c.drawString(MARGIN, page_h - MARGIN - title_font_size,
                     f"{core_id} | {img_type.upper()}")

        label_font_size = max(8, title_font_size - 4)
        cur_x = SCALE_WIDTH
        for im in processed_images:
            if show_mscl_pdf:
                self._draw_mscl_plot_pdf(c, cur_x, top_y, MSCL_PLOT_W,
                                         im['pt_h'], im['sec'],
                                         im['phys_h_cm'], im['crop_top_cm'])
                cur_x += MSCL_PLOT_W
            im['buf'].seek(0)
            c.drawImage(ImageReader(im['buf']), cur_x, top_y - im['pt_h'],
                        width=im['pt_w'], height=im['pt_h'])
            c.setFont("Helvetica", label_font_size)
            c.setFillColorRGB(1, 1, 1)
            c.drawCentredString(cur_x + im['pt_w'] / 2, MARGIN, f"Section {im['sec']}")
            cur_x += im['pt_w'] + INTER_IMAGE_GAP


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoreCollectorApp()
    window.show()
    sys.exit(app.exec())