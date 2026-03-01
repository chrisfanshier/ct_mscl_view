import sys
import io
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from PIL import Image

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                             QListWidgetItem, QLabel, QScrollArea, QCheckBox, QFrame)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

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

# ── Scientific Image Logic ──────────────────────────────────────────────────

def auto_crop(arr):
    max_val = arr.max()
    if max_val == 0: return arr
    col_means = arr.mean(axis=0)
    row_means = arr.mean(axis=1)
    left   = int(np.argmax(col_means > max_val * CROP_COL_THRESH))
    right  = int(len(col_means) - np.argmax(col_means[::-1] > max_val * CROP_COL_THRESH) - 1)
    top    = int(np.argmax(row_means > max_val * CROP_ROW_THRESH))
    bottom = int(len(row_means) - np.argmax(row_means[::-1] > max_val * CROP_ROW_THRESH) - 1)
    return arr[top:bottom + 1, left:right + 1] if right > left and bottom > top else arr

def load_tif_scientific(tif_path):
    img = Image.open(tif_path)
    dpi = float(img.info.get("dpi", (300, 300))[1])
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
    return Image.fromarray(arr.astype(np.uint8), mode="L"), dpi

# ── Metadata Logic ──────────────────────────────────────────────────────────

def parse_xml_robust(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        def get_text(*tags):
            for tag in tags:
                el = root.find(f".//{tag}")
                if el is not None and el.text: return el.text.strip()
            return None
        sec_id = get_text("Section-ID", "section-id", "sectionID", "Section-Number", "section-number")
        core_name = get_text("Core-ID", "coreID")
        core_name = core_name.strip().replace("_", "-").upper() if core_name else "UNKNOWN"
        return {
            "core_id": core_name,
            "section_number": int(sec_id) if (sec_id and sec_id.isdigit()) else 0,
            "image_type": (get_text("Image-Type", "image-type") or "Unknown").lower()
        }
    except: return None

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
                    if meta: pairs.append({"tif": tif_path, "meta": meta})
        self.finished.emit(pairs)

class CoreCollectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Core Summary Viewer")
        self.resize(1300, 850)
        self.found_items = []

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        self.btn_browse = QPushButton("1. Select Core Directory")
        self.btn_browse.clicked.connect(self.browse_directory)
        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self.update_previews)
        self.chk_scale_bar = QCheckBox("Add Full Depth Scale")
        self.chk_scale_bar.setChecked(True)
        self.btn_export = QPushButton("2. Export Selected to PDF")
        self.btn_export.clicked.connect(self.export_pdf)
        self.btn_export.setEnabled(False)

        left_panel.addWidget(self.btn_browse)
        left_panel.addWidget(QLabel("Available Sections:"))
        left_panel.addWidget(self.list_widget)
        left_panel.addWidget(self.chk_scale_bar)
        left_panel.addWidget(self.btn_export)
        
        self.preview_area = QScrollArea()
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_area.setWidget(self.preview_container)
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setStyleSheet("background-color: #1a1a1a;")

        layout.addLayout(left_panel, 1)
        layout.addWidget(self.preview_area, 4)

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

    def update_previews(self):
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).checkState() == Qt.Checked:
                self.add_thumbnail(self.found_items[i])

    def add_thumbnail(self, data):
        try:
            pil_img, _ = load_tif_scientific(data['tif'])
            target_width = 150 
            w_percent = (target_width / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            resized_img = pil_img.resize((target_width, h_size), Image.Resampling.LANCZOS)
            img_data = resized_img.tobytes()
            bytes_per_line = resized_img.size[0]
            qimg = QImage(img_data, resized_img.size[0], resized_img.size[1], bytes_per_line, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg.copy())
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFrameShape(QFrame.Panel)
            self.preview_layout.addWidget(lbl, 0, Qt.AlignTop)
        except: pass

    def export_pdf(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF (*.pdf)")
        if not save_path: return
        selected = [self.found_items[i] for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.Checked]
        if not selected: return
        groups = defaultdict(list)
        for item in selected:
            groups[(item['meta']['core_id'], item['meta']['image_type'])].append(item)
        c = canvas.Canvas(save_path)
        for key in sorted(groups.keys()):
            self.generate_page(c, key, sorted(groups[key], key=lambda x: x['meta']['section_number']))
            c.showPage()
        c.save()

    def draw_full_scale(self, c, x, start_y, total_pt_height):
        """Draws a vertical scale from 0 to the total height of the longest section."""
        pts_per_cm = 72 / 2.54
        total_cm = total_pt_height / pts_per_cm
        
        c.setStrokeColorRGB(1, 1, 1)
        c.setLineWidth(1.2)
        
        # Main vertical line
        c.line(x, start_y, x, start_y - total_pt_height)
        
        # Draw Ticks and Labels
        for i in range(int(total_cm) + 1):
            tick_y = start_y - (i * pts_per_cm)
            
            if i % 10 == 0:
                # Major tick + Label every 10cm
                c.setLineWidth(1.5)
                c.line(x, tick_y, x + 8, tick_y)
                c.setFillColorRGB(1, 1, 1)
                c.setFont("Helvetica-Bold", 9)
                c.drawString(x + 12, tick_y - 3, f"{i} cm")
            elif i % 5 == 0:
                # Mid tick every 5cm
                c.setLineWidth(1.0)
                c.line(x, tick_y, x + 5, tick_y)
            else:
                # Small tick every 1cm
                c.setLineWidth(0.5)
                c.line(x, tick_y, x + 3, tick_y)

    def generate_page(self, c, key, items):
        core_id, img_type = key
        processed_images = []
        for item in items:
            pil_img, dpi = load_tif_scientific(item['tif'])
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            processed_images.append({
                'buf': buf, 
                'pt_w': (pil_img.size[0] / dpi) * 72.0, 
                'pt_h': (pil_img.size[1] / dpi) * 72.0, 
                'sec': item['meta']['section_number']
            })

        max_h = max(im['pt_h'] for im in processed_images)
        total_w = sum(im['pt_w'] for im in processed_images) + (len(processed_images)-1)*INTER_IMAGE_GAP
        
        # Extra margin on the left for the labels (3 cm instead of MARGIN)
        SCALE_WIDTH = 1.5 * cm
        page_w = SCALE_WIDTH + MARGIN + total_w
        page_h = 2 * MARGIN + HEADER_HEIGHT + LABEL_HEIGHT + max_h
        
        c.setPageSize((page_w, page_h))
        c.setFillColorRGB(*BG_COLOR)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        # Scale baseline is the top of the images
        top_y = page_h - MARGIN - HEADER_HEIGHT
        
        if self.chk_scale_bar.isChecked():
            self.draw_full_scale(c, MARGIN/2, top_y, max_h)

        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, page_h - MARGIN - 14, f"{core_id} | {img_type.upper()}")

        cur_x = SCALE_WIDTH # Start after scale area
        for im in processed_images:
            im['buf'].seek(0)
            # Draw from top_y downwards
            c.drawImage(ImageReader(im['buf']), cur_x, top_y - im['pt_h'], width=im['pt_w'], height=im['pt_h'])
            c.drawCentredString(cur_x + im['pt_w']/2, MARGIN, f"Section {im['sec']}")
            cur_x += im['pt_w'] + INTER_IMAGE_GAP

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoreCollectorApp()
    window.show()
    sys.exit(app.exec())