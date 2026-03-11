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
                               QComboBox, QSlider, QSizePolicy, QSpinBox, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QWheelEvent

from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ── Layout Constants ──────────────────────────────────────────────────────────
MARGIN            = 1.5 * cm
INTER_IMAGE_GAP   = 0.6 * cm
HEADER_HEIGHT     = 2.0 * cm
LABEL_HEIGHT      = 0.6 * cm
BG_COLOR          = (0.12, 0.12, 0.12)
CROP_COL_THRESH   = 0.15
CROP_ROW_THRESH   = 0.25

FALLBACK_PX_PER_CM = 35.0
CM_TO_PT           = 28.3465   # 1 cm → PDF points
UI_PX_PER_CM       = 22.0      # Screen rendering constant — the "source of truth"
                                # for physical alignment in the preview.
                                # All preview heights = phys_cm * UI_PX_PER_CM * zoom

MSCL_COLORS_PIL = [
    (80, 190, 255), (255, 100, 100), (100, 255, 100), (255, 200, 50),
    (200, 100, 255), (255, 150, 50), (50, 255, 200), (255, 100, 200),
]
MSCL_COLORS_PDF = [
    (0.31, 0.75, 1.0), (1.0, 0.39, 0.39), (0.39, 1.0, 0.39), (1.0, 0.78, 0.2),
    (0.78, 0.39, 1.0), (1.0, 0.59, 0.2), (0.2, 1.0, 0.78), (1.0, 0.39, 0.78),
]

# ── Scientific Image Logic ────────────────────────────────────────────────────

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
    Returns (PIL image, px_per_cm, crop_top_px, original_px_h).

    original_px_h is the pre-crop pixel height — used to compute the true
    physical section length regardless of how much black border was trimmed.
    """
    img = Image.open(tif_path)
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32)
    original_px_h = arr.shape[0]
    arr, crop_top_px = auto_crop(arr)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr[:] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L"), px_per_cm, crop_top_px, original_px_h


# ── Metadata Logic ────────────────────────────────────────────────────────────

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

        # Prioritise the high-precision horizontal-resolution tag when present
        px_per_cm_raw = get_text("horizontal-resolution", "pixels-per-CM",
                                 "pixels-per-cm", "PixelsPerCM")
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
    except Exception:
        return None


# ── MSCL Data Parsing ─────────────────────────────────────────────────────────

def parse_mscl_file(filepath):
    """Parse a Geotek MSCL .out file.
    Returns (section_data, data_columns).
    section_data: {section_num: {'SECT DEPTH': np.array, col: np.array, ...}}
    """
    lines = Path(filepath).read_text(encoding='utf-8', errors='replace').splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("SB DEPTH"):
            header_idx = i
            break
    if header_idx is None:
        return None, []

    headers    = [h.strip() for h in lines[header_idx].split('\t')]
    data_start = header_idx + 2   # skip units row

    skip_cols  = {"SB DEPTH", "SECT NUM", "SECT DEPTH"}
    data_columns = [h for h in headers
                    if h and h.upper() not in {s.upper() for s in skip_cols}]

    sect_num_idx   = next((i for i, h in enumerate(headers) if h.upper() == "SECT NUM"), None)
    sect_depth_idx = next((i for i, h in enumerate(headers) if h.upper() == "SECT DEPTH"), None)
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
            sec_num    = int(raw_sec)
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


# ── Font helper ───────────────────────────────────────────────────────────────

def _pil_font(size):
    from PIL import ImageFont
    for path in ("arialbd.ttf", "arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Application Components ────────────────────────────────────────────────────

class ScanWorker(QThread):
    finished = Signal(list)

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        pairs = []
        for tif_path in Path(self.directory).rglob("*"):
            if ("corrected" in tif_path.name.lower()
                    and tif_path.suffix.lower() in (".tif", ".tiff")):
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
        self._dragging    = False
        self._drag_start  = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        self.viewport().setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event: QWheelEvent):
        self.zoom_changed.emit(event.angleDelta().y())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging       = True
            self._drag_start     = event.position().toPoint()
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
        self.resize(1350, 900)
        self.found_items  = []
        self.zoom_factor  = 1.0
        self.mscl_data    = None
        self.mscl_columns = []
        self.mscl_col_rows = {}    # col_name -> {'check', 'min_edit', 'max_edit'}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # ── Left panel ────────────────────────────────────────────────────
        left_panel = QVBoxLayout()

        self.btn_browse = QPushButton("1. Select Core Directory")
        self.btn_browse.clicked.connect(self.browse_directory)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self.update_previews)

        # Scale bar controls
        self.chk_scale_bar = QCheckBox("Add Full Depth Scale")
        self.chk_scale_bar.setChecked(True)
        self.chk_scale_bar.stateChanged.connect(self.update_previews)

        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(["One scale per section", "Single scale on left"])
        self.scale_mode_combo.setCurrentIndex(0)
        self.scale_mode_combo.currentIndexChanged.connect(self.update_previews)

        # MSCL controls
        self.btn_mscl = QPushButton("Load MSCL File (.out)")
        self.btn_mscl.clicked.connect(self.load_mscl_file)

        self.chk_mscl = QCheckBox("Show MSCL Plot")
        self.chk_mscl.setChecked(False)
        self.chk_mscl.setEnabled(False)
        self.chk_mscl.stateChanged.connect(self.update_previews)

        self.mscl_width_label  = QLabel("Plot width: 80")
        self.mscl_width_slider = QSlider(Qt.Horizontal)
        self.mscl_width_slider.setRange(30, 300)
        self.mscl_width_slider.setValue(80)
        self.mscl_width_slider.setEnabled(False)
        self.mscl_width_slider.valueChanged.connect(self._on_mscl_width_changed)

        self._mscl_col_inner   = QWidget()
        self._mscl_col_vlayout = QVBoxLayout(self._mscl_col_inner)
        self._mscl_col_vlayout.setContentsMargins(2, 2, 2, 2)
        self._mscl_col_vlayout.setSpacing(2)
        self.mscl_col_scroll = QScrollArea()
        self.mscl_col_scroll.setWidget(self._mscl_col_inner)
        self.mscl_col_scroll.setWidgetResizable(True)
        self.mscl_col_scroll.setMaximumHeight(160)
        self.mscl_col_scroll.setEnabled(False)

        # Export options (restored from v1)
        exp_lbl = QLabel("── Export Options ──")
        exp_lbl.setStyleSheet("font-weight: bold;")

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
            "Draft (JPEG 50%, downsample 2×)",
        ])
        self.pdf_quality_combo.setCurrentIndex(0)

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
        left_panel.addWidget(QLabel("MSCL Columns (checkbox  min  max):"))
        left_panel.addWidget(self.mscl_col_scroll)
        left_panel.addWidget(QLabel(""))
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

        # ── Preview area ──────────────────────────────────────────────────
        self.preview_area = ZoomableScrollArea()
        self.preview_area.zoom_changed.connect(self._on_wheel_zoom)
        self.preview_container = QWidget()
        self.preview_layout    = QHBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_area.setWidget(self.preview_container)
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setStyleSheet("background-color: #1a1a1a;")

        layout.addLayout(left_panel, 1)
        layout.addWidget(self.preview_area, 4)

    # ── Zoom ──────────────────────────────────────────────────────────────

    def _set_zoom(self, factor):
        self.zoom_factor = max(0.25, min(factor, 5.0))
        self.zoom_label.setText(f"Zoom: {int(self.zoom_factor * 100)}%  "
                                "(scroll to zoom, drag to pan)")
        self.update_previews()

    def _on_wheel_zoom(self, delta):
        self._set_zoom(self.zoom_factor + (0.15 if delta > 0 else -0.15))

    def _on_mscl_width_changed(self, val):
        self.mscl_width_label.setText(f"Plot width: {val}")
        self.update_previews()

    def get_selected_mscl_columns(self):
        return [col for col, row in self.mscl_col_rows.items()
                if row['check'].isChecked()]

    def _get_mscl_ranges(self, checked_items, selected_cols):
        """Compute global x-axis (value) ranges for each MSCL column across
        all displayed sections.  User-supplied min/max fields override the
        auto range.  Returns {col_name: (v_lo, v_hi)}.
        """
        ranges = {}
        if self.mscl_data is None:
            return ranges
        for col_name in selected_cols:
            all_vals = []
            for item in checked_items:
                sec_num  = item['meta']['section_number']
                sec_data = self.mscl_data.get(sec_num)
                if sec_data and col_name in sec_data:
                    vals = sec_data[col_name]
                    all_vals.append(vals[~np.isnan(vals)])
            if not all_vals:
                continue
            combined = np.concatenate(all_vals)
            if len(combined) < 2:
                continue
            v_lo, v_hi = np.percentile(combined, (2, 98))
            if v_hi <= v_lo:
                v_lo, v_hi = combined.min(), combined.max()
            if v_hi <= v_lo:
                continue
            # Apply user overrides from the min/max input fields
            row      = self.mscl_col_rows.get(col_name, {})
            min_edit = row.get('min_edit')
            max_edit = row.get('max_edit')
            if min_edit:
                try:
                    v_lo = float(min_edit.text().strip())
                except ValueError:
                    pass
            if max_edit:
                try:
                    v_hi = float(max_edit.text().strip())
                except ValueError:
                    pass
            ranges[col_name] = (v_lo, v_hi)
        return ranges

    # ── Directory / file loading ──────────────────────────────────────────

    def browse_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if d:
            self.list_widget.clear()
            self.worker = ScanWorker(d)
            self.worker.finished.connect(self.on_scan_finished)
            self.worker.start()

    def on_scan_finished(self, results):
        self.found_items = results
        for item in results:
            label = (f"{item['meta']['core_id']} | "
                     f"Sec {item['meta']['section_number']} | "
                     f"{item['meta']['image_type']}")
            li = QListWidgetItem(label)
            li.setFlags(li.flags() | Qt.ItemIsUserCheckable)
            li.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(li)
        self.btn_export.setEnabled(True)

    def load_mscl_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select MSCL .out File", "", "MSCL Files (*.out);;All Files (*)")
        if not fp:
            return
        data, columns = parse_mscl_file(fp)
        if data is None:
            self.mscl_data    = None
            self.mscl_columns = []
            self.chk_mscl.setEnabled(False)
            self.mscl_col_scroll.setEnabled(False)
            self.mscl_width_slider.setEnabled(False)
            return
        self.mscl_data    = data
        self.mscl_columns = columns
        # Rebuild custom column rows
        self.mscl_col_rows = {}
        while self._mscl_col_vlayout.count():
            child = self._mscl_col_vlayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for col in columns:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            chk = QCheckBox(col)
            chk.setChecked(col == "Den1")
            chk.stateChanged.connect(self.update_previews)
            min_edit = QLineEdit()
            min_edit.setPlaceholderText("auto")
            min_edit.setFixedWidth(52)
            min_edit.editingFinished.connect(self.update_previews)
            max_edit = QLineEdit()
            max_edit.setPlaceholderText("auto")
            max_edit.setFixedWidth(52)
            max_edit.editingFinished.connect(self.update_previews)
            row_layout.addWidget(chk)
            row_layout.addStretch()
            row_layout.addWidget(QLabel("min"))
            row_layout.addWidget(min_edit)
            row_layout.addWidget(QLabel("max"))
            row_layout.addWidget(max_edit)
            self._mscl_col_vlayout.addWidget(row_widget)
            self.mscl_col_rows[col] = {'check': chk, 'min_edit': min_edit, 'max_edit': max_edit}
        self._mscl_col_vlayout.addStretch()
        self.chk_mscl.setEnabled(True)
        self.mscl_col_scroll.setEnabled(True)
        self.mscl_width_slider.setEnabled(True)
        self.chk_mscl.setChecked(True)
        self.btn_mscl.setText(f"MSCL: {Path(fp).name}")
        self.update_previews()

    # ── Preview rendering ─────────────────────────────────────────────────

    def update_previews(self):
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        checked = [self.found_items[i]
                   for i in range(self.list_widget.count())
                   if self.list_widget.item(i).checkState() == Qt.Checked]
        checked.sort(key=lambda x: x['meta']['section_number'])
        if not checked:
            return

        show_scale = self.chk_scale_bar.isChecked()
        per_section = (self.scale_mode_combo.currentIndex() == 0)

        if show_scale and not per_section:
            self._add_standalone_scale(checked)

        # Compute global MSCL ranges once across all displayed sections
        mscl_ranges = {}
        selected_cols = self.get_selected_mscl_columns()
        if (self.mscl_data is not None and self.chk_mscl.isChecked()
                and selected_cols):
            mscl_ranges = self._get_mscl_ranges(checked, selected_cols)

        for item in checked:
            self.add_thumbnail(item, draw_scale=(show_scale and per_section),
                               mscl_ranges=mscl_ranges)

    def _add_standalone_scale(self, checked_items):
        """Standalone depth scale shown once to the left of all thumbnails."""
        from PIL import ImageDraw

        preview_px_per_cm = UI_PX_PER_CM * self.zoom_factor

        max_phys_h = 0
        for item in checked_items:
            _, _, _, orig_h = load_tif_scientific(item['tif'], item['meta']['px_per_cm'])
            max_phys_h = max(max_phys_h, orig_h / item['meta']['px_per_cm'])

        LABEL_H = max(28, int(40 * self.zoom_factor))  # Increased label height
        bar_w   = int(50 * self.zoom_factor)
        bar_h   = int(max_phys_h * preview_px_per_cm)
        total_h = bar_h + LABEL_H

        bar_img = Image.new("RGB", (bar_w, total_h), (31, 31, 31))
        draw    = ImageDraw.Draw(bar_img)
        font    = _pil_font(int(18 * self.zoom_factor))  # Increased font size for scale numbers
        bx      = bar_w - 2

        draw.line([(bx, 0), (bx, bar_h)], fill=(200, 200, 200), width=1)

        tick = 0
        while tick <= int(max_phys_h):
            ty = int(tick * preview_px_per_cm)
            if ty >= bar_h:
                break
            if tick % 10 == 0:
                draw.line([(bx - 5, ty), (bx, ty)], fill=(220, 220, 220), width=1)
                draw.text((2, ty - 10), str(tick), fill=(200, 200, 200), font=font)  # Adjusted y for larger font
            elif tick % 5 == 0:
                draw.line([(bx - 4, ty), (bx, ty)], fill=(170, 170, 170), width=1)
            tick += 1

        qimg = QImage(bar_img.tobytes(), bar_w, total_h,
                      bar_w * 3, QImage.Format_RGB888)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.preview_layout.addWidget(lbl, 0, Qt.AlignTop)

    def _render_mscl_plot_pil(self, series_list, plot_w, plot_h, preview_px_per_cm, ranges=None):
        """
        Render MSCL series as a vertical PIL image.
        Depths are SECT DEPTH (0 = physical section top).
        plot_h corresponds to the full physical section canvas height.
        preview_px_per_cm converts depth (cm) → pixel row in this canvas.
        """
        from PIL import ImageDraw

        img  = Image.new("RGB", (plot_w, plot_h), (25, 25, 35))
        draw = ImageDraw.Draw(img)
        font = _pil_font(8)
        PAD  = 3
        usable_w = plot_w - 2 * PAD

        draw.line([(PAD, 0), (PAD, plot_h - 1)], fill=(60, 60, 60), width=1)

        names = []
        for col_name, depths, values, color in series_list:
            if ranges and col_name in ranges:
                v_lo, v_hi = ranges[col_name]
            else:
                v_lo, v_hi = np.percentile(values, (2, 98))
                if v_hi <= v_lo:
                    v_lo, v_hi = values.min(), values.max()
            if v_hi <= v_lo:
                continue

            points = []
            for i in range(len(depths)):
                py = int(depths[i] * preview_px_per_cm)
                if py < 0 or py >= plot_h:
                    continue
                v_c = max(v_lo, min(v_hi, values[i]))
                px  = PAD + int((v_c - v_lo) / (v_hi - v_lo) * max(1, usable_w - 1))
                points.append((px, py))

            if len(points) > 1:
                draw.line(points, fill=color, width=1)
            names.append((col_name, color))

        # Legend
        font_sm = _pil_font(8)
        label_y = plot_h - 10
        lx = 2
        for name, color in names:
            draw.text((lx, label_y), name, fill=color, font=font_sm)
            bb  = draw.textbbox((0, 0), name + "  ", font=font_sm)
            lx += bb[2] - bb[0]

        return img

    def add_thumbnail(self, data, draw_scale=True, mscl_ranges=None):
        from PIL import ImageDraw

        px_per_cm = data['meta']['px_per_cm']
        pil_img, _, crop_top_px, original_px_h = load_tif_scientific(
            data['tif'], px_per_cm)

        preview_px_per_cm = UI_PX_PER_CM * self.zoom_factor
        scale             = preview_px_per_cm / px_per_cm

        full_phys_h_cm = original_px_h / px_per_cm
        canvas_h       = int(original_px_h * scale)
        img_w          = int(pil_img.size[0] * scale)
        img_h          = int(pil_img.size[1] * scale)
        crop_y         = int(crop_top_px * scale)

        selected_cols = self.get_selected_mscl_columns()
        show_mscl = (self.mscl_data is not None
                     and self.chk_mscl.isChecked()
                     and len(selected_cols) > 0)
        mscl_w = int(self.mscl_width_slider.value() * self.zoom_factor) if show_mscl else 0

        mscl_plot = None
        if show_mscl:
            sec_data = self.mscl_data.get(data['meta']['section_number'])
            if sec_data and "SECT DEPTH" in sec_data:
                series_list = []
                for ci, col_name in enumerate(selected_cols):
                    if col_name not in sec_data:
                        continue
                    depths = sec_data["SECT DEPTH"]
                    values = sec_data[col_name]
                    mask   = ~np.isnan(values)
                    if mask.sum() >= 2:
                        series_list.append((col_name, depths[mask],
                                            values[mask],
                                            MSCL_COLORS_PIL[ci % len(MSCL_COLORS_PIL)]))
                if series_list:
                    mscl_plot = self._render_mscl_plot_pil(
                        series_list, mscl_w, canvas_h, preview_px_per_cm,
                        ranges=mscl_ranges)

        LABEL_H   = max(28, int(40 * self.zoom_factor))  # Increased label height
        SCALEBAR_W = int(45 * self.zoom_factor) if draw_scale else 0
        total_w   = mscl_w + SCALEBAR_W + img_w
        total_h   = canvas_h + LABEL_H

        thumb = Image.new("RGB", (total_w, total_h), (31, 31, 31))

        if mscl_plot:
            thumb.paste(mscl_plot, (0, 0))

        thumb.paste(pil_img.resize((img_w, img_h), Image.Resampling.LANCZOS).convert("RGB"),
                    (mscl_w + SCALEBAR_W, crop_y))

        draw = ImageDraw.Draw(thumb)

        # ── Scale bar ────────────────────────────────────────────────────
        if draw_scale:
            font  = _pil_font(int(18 * self.zoom_factor))  # Increased font size for scale numbers
            bx    = mscl_w + SCALEBAR_W - 2
            draw.line([(bx, 0), (bx, canvas_h)], fill=(200, 200, 200), width=1)

            tick = 0
            while tick <= int(full_phys_h_cm):
                ty = int(tick * preview_px_per_cm)
                if ty >= canvas_h:
                    break
                if tick % 10 == 0:
                    draw.line([(bx - 5, ty), (bx, ty)], fill=(220, 220, 220), width=1)
                    draw.text((mscl_w + 2, ty - 10), str(tick),
                              fill=(200, 200, 200), font=font)  # Adjusted y for larger font
                elif tick % 5 == 0:
                    draw.line([(bx - 4, ty), (bx, ty)], fill=(170, 170, 170), width=1)
                else:
                    draw.line([(bx - 2, ty), (bx, ty)], fill=(120, 120, 120), width=1)
                tick += 1

        # ── Section label ────────────────────────────────────────────────
        font_lbl   = _pil_font(int(24 * self.zoom_factor))  # Increased font size for section label
        label_text = f"Section {data['meta']['section_number']}"
        bb         = draw.textbbox((0, 0), label_text, font=font_lbl)
        label_x    = mscl_w + SCALEBAR_W + (img_w - (bb[2] - bb[0])) // 2
        draw.text((label_x, canvas_h + 4), label_text,
                  fill=(200, 200, 200), font=font_lbl)

        qimg = QImage(thumb.tobytes(), total_w, total_h,
                      total_w * 3, QImage.Format_RGB888)
        lbl  = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg.copy()))
        lbl.setFrameShape(QFrame.Panel)
        self.preview_layout.addWidget(lbl, 0, Qt.AlignTop)

    # ── PDF Export ────────────────────────────────────────────────────────

    def export_pdf(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "", "PDF (*.pdf)")
        if not save_path:
            return
        selected = [self.found_items[i]
                    for i in range(self.list_widget.count())
                    if self.list_widget.item(i).checkState() == Qt.Checked]
        if not selected:
            return
        groups = defaultdict(list)
        for item in selected:
            groups[(item['meta']['core_id'], item['meta']['image_type'])].append(item)
        c = canvas.Canvas(save_path)
        for key in sorted(groups.keys()):
            self.generate_page(
                c, key,
                sorted(groups[key], key=lambda x: x['meta']['section_number']))
            c.showPage()
        c.save()

    def draw_scale_bar(self, c, x, top_y, total_pt_height, phys_h_cm, font_size):
        """
        Draw a vertical depth scale bar on the PDF canvas.
        x            : x-position of the bar line
        top_y        : PDF y of the physical section top (depth = 0 cm)
        total_pt_height : height of the full section in points
        phys_h_cm    : physical height of the section in cm
        font_size    : label font size
        """
        pt_per_cm = total_pt_height / phys_h_cm

        c.setStrokeColorRGB(1, 1, 1)
        c.setLineWidth(1.2)
        c.line(x, top_y, x, top_y - total_pt_height)

        tick = 0
        while tick <= int(phys_h_cm):
            tick_y = top_y - tick * pt_per_cm
            if tick_y < top_y - total_pt_height - 1:
                break
            if tick % 10 == 0:
                c.setLineWidth(1.5)
                c.line(x, tick_y, x + 8, tick_y)
                c.setFillColorRGB(1, 1, 1)
                c.setFont("Helvetica-Bold", font_size)
                c.drawString(x + 12, tick_y - 3, f"{tick} cm")
            elif tick % 5 == 0:
                c.setLineWidth(1.0)
                c.line(x, tick_y, x + 5, tick_y)
            else:
                c.setLineWidth(0.5)
                c.line(x, tick_y, x + 3, tick_y)
            tick += 1

    def _draw_mscl_plot_pdf(self, c, x, top_y, plot_w_pt, plot_h_pt,
                             section_num, full_phys_h_cm, ranges=None):
        """
        Draw MSCL series on the PDF canvas.
        top_y is the PDF y of the physical section top (depth = 0 cm).
        Depths are SECT DEPTH (0 = section top).
        plot_h_pt covers the full physical section height.
        """
        selected_cols = self.get_selected_mscl_columns()
        if not selected_cols or self.mscl_data is None:
            return
        sec_data = self.mscl_data.get(section_num)
        if not sec_data or "SECT DEPTH" not in sec_data:
            return

        pt_per_cm = plot_h_pt / full_phys_h_cm if full_phys_h_cm else 1
        PAD       = 3
        usable_w  = plot_w_pt - 2 * PAD

        c.setStrokeColorRGB(0.25, 0.25, 0.25)
        c.setLineWidth(0.5)
        c.line(x + PAD, top_y, x + PAD, top_y - plot_h_pt)

        names_colors = []
        for ci, col_name in enumerate(selected_cols):
            if col_name not in sec_data:
                continue
            depths = sec_data["SECT DEPTH"]
            values = sec_data[col_name]
            mask   = ~np.isnan(values)
            if mask.sum() < 2:
                continue
            d, v = depths[mask], values[mask]
            if ranges and col_name in ranges:
                v_lo, v_hi = ranges[col_name]
            else:
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
                py  = top_y - (d[i] * pt_per_cm)
                py  = max(top_y - plot_h_pt, min(top_y, py))
                v_c = max(v_lo, min(v_hi, v[i]))
                px  = x + PAD + ((v_c - v_lo) / (v_hi - v_lo)) * usable_w
                if prev is not None:
                    c.line(prev[0], prev[1], px, py)
                prev = (px, py)

            names_colors.append((col_name, (r, g, b)))

        # Legend
        c.setFont("Helvetica", 8)
        lx = x + 1
        for name, (r, g, b) in names_colors:
            c.setFillColorRGB(r, g, b)
            c.drawString(lx, top_y - plot_h_pt - 10, name)
            lx += c.stringWidth(name, "Helvetica", 8) + 4

    def generate_page(self, c, key, items):
        """Render one PDF page for a (core_id, image_type) group."""
        core_id, img_type = key

        # ── Export settings ───────────────────────────────────────────────
        scale_pct       = int(self.pdf_scale_combo.currentText().replace('%', ''))
        pdf_scale       = scale_pct / 100.0
        quality_idx     = self.pdf_quality_combo.currentIndex()
        title_font_size = self.pdf_text_size_spin.value()

        # pt_per_cm is the single physical constant for this PDF render.
        # All heights/widths derive from phys_cm * pt_per_cm, so sections
        # that share the same physical length will always be identical height.
        pt_per_cm = CM_TO_PT * pdf_scale

        processed = []
        for item in items:
            px_per_cm = item['meta']['px_per_cm']
            pil_img, _, crop_top_px, original_px_h = load_tif_scientific(
                item['tif'], px_per_cm)
            px_w, px_h = pil_img.size

            full_phys_h_cm = original_px_h / px_per_cm
            pt_h_full      = full_phys_h_cm * pt_per_cm   # canvas height in pts
            crop_top_cm    = crop_top_px / px_per_cm
            crop_top_pt    = crop_top_cm * pt_per_cm
            pt_h_img       = (px_h / px_per_cm) * pt_per_cm
            pt_w           = (px_w / px_per_cm) * pt_per_cm

            # Image buffer with selected quality
            buf = io.BytesIO()
            if quality_idx == 0:
                pil_img.save(buf, format="PNG")
            elif quality_idx == 1:
                pil_img.save(buf, format="JPEG", quality=85)
            else:
                ds = pil_img.resize((max(1, px_w // 2), max(1, px_h // 2)),
                                    Image.Resampling.LANCZOS)
                ds.save(buf, format="JPEG", quality=50)

            processed.append({
                'buf':             buf,
                'pt_w':            pt_w,
                'pt_h_full':       pt_h_full,
                'pt_h_img':        pt_h_img,
                'crop_top_pt':     crop_top_pt,
                'full_phys_h_cm':  full_phys_h_cm,
                'sec':             item['meta']['section_number'],
            })

        max_h          = max(im['pt_h_full'] for im in processed)
        tallest        = max(processed, key=lambda im: im['pt_h_full'])
        full_phys_h_cm = tallest['full_phys_h_cm']

        show_mscl_pdf = (self.mscl_data is not None
                         and self.chk_mscl.isChecked()
                         and len(self.get_selected_mscl_columns()) > 0)

        # MSCL width in PDF points — same fraction of image width as the
        # preview slider represents relative to the slider's own scale.
        # slider range 30–300 maps against image pixel width; we replicate
        # that as a fraction of the PDF image width for each section.
        mscl_frac = self.mscl_width_slider.value() / 150.0  # same ratio as preview

        for im in processed:
            im['mscl_pt_w'] = (im['pt_w'] * mscl_frac) if show_mscl_pdf else 0

        SCALE_WIDTH = 1.5 * cm
        total_w = (SCALE_WIDTH + MARGIN
                   + sum(im['pt_w'] + im['mscl_pt_w'] for im in processed)
                   + (len(processed) - 1) * INTER_IMAGE_GAP)
        page_h  = 2 * MARGIN + HEADER_HEIGHT + LABEL_HEIGHT + max_h

        c.setPageSize((total_w, page_h))
        c.setFillColorRGB(*BG_COLOR)
        c.rect(0, 0, total_w, page_h, fill=1, stroke=0)

        top_y = page_h - MARGIN - HEADER_HEIGHT

        # Master scale bar
        scale_bar_font = max(8, title_font_size - 3)
        if self.chk_scale_bar.isChecked():
            self.draw_scale_bar(c, MARGIN / 2, top_y, max_h,
                                full_phys_h_cm, scale_bar_font)

        # Title
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", title_font_size)
        c.drawString(MARGIN, page_h - MARGIN - title_font_size,
                     f"{core_id} | {img_type.upper()}")

        label_font_size = max(8, title_font_size - 4)
        cur_x = SCALE_WIDTH

        # Compute global MSCL ranges for PDF export
        mscl_ranges_pdf = {}
        if show_mscl_pdf:
            mscl_ranges_pdf = self._get_mscl_ranges(
                items, self.get_selected_mscl_columns())

        for im in processed:
            # MSCL plot to the left of the image
            if show_mscl_pdf:
                self._draw_mscl_plot_pdf(
                    c, cur_x, top_y,
                    im['mscl_pt_w'], im['pt_h_full'],
                    im['sec'], im['full_phys_h_cm'],
                    ranges=mscl_ranges_pdf)
                cur_x += im['mscl_pt_w']

            # Image pasted at its correct physical offset within the full canvas
            im['buf'].seek(0)
            img_y = top_y - im['crop_top_pt'] - im['pt_h_img']
            c.drawImage(ImageReader(im['buf']), cur_x, img_y,
                        width=im['pt_w'], height=im['pt_h_img'])

            # Section label
            c.setFont("Helvetica", label_font_size)
            c.setFillColorRGB(1, 1, 1)
            c.drawCentredString(cur_x + im['pt_w'] / 2, MARGIN,
                                f"Section {im['sec']}")
            cur_x += im['pt_w'] + INTER_IMAGE_GAP


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoreCollectorApp()
    window.show()
    sys.exit(app.exec())