#!/usr/bin/env python3
import io
import cv2
import numpy as np
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image

# --------- 参数解析 ---------
parser = argparse.ArgumentParser(description='Generate a PDF with AprilTags on both sides')
parser.add_argument('--page_size', type=float, default=200, help='Page size in mm (default: 200)')
parser.add_argument('--margin', type=float, default=20, help='Margin from page edge in mm (default: 10)')
parser.add_argument('--tags_per_side', type=int, default=8, help='Number of tags on each side (default: 8)')
parser.add_argument('--tag_size', type=float, default=15, help='Size of each AprilTag in mm (default: 15)')
parser.add_argument('--checker_size', type=float, default=2, help='Space between tags in mm (default: 20)')
parser.add_argument('--start_id', type=int, default=0, help='Starting ID for tags (default: 0)')
parser.add_argument('--family', choices=['16h5','36h11','25h9'], default='16h5', help='AprilTag family (default: 16h5)')
parser.add_argument('--output', type=str, default='apriltags.pdf', help='Output PDF filename')
args = parser.parse_args()

# --------- 字典映射 ---------
fmap = {
    '16h5': cv2.aruco.DICT_APRILTAG_16H5,
    '36h11': cv2.aruco.DICT_APRILTAG_36H11,
    '25h9': cv2.aruco.DICT_APRILTAG_25H9,
}
aruco_dict = cv2.aruco.getPredefinedDictionary(fmap[args.family])

# --------- 计算垂直间距，确保上下边距一致 ---------
total_tag_height = args.tags_per_side * args.tag_size
available_space = args.page_size - 2 * args.margin - total_tag_height
vertical_spacing = available_space / (args.tags_per_side - 1) if args.tags_per_side > 1 else 0

# --------- 生成PDF ---------
buf = io.BytesIO()
c = canvas.Canvas(buf, pagesize=(args.page_size*mm, args.page_size*mm))

# 绘制黑色边框
page_mm = args.page_size * mm
# c.setLineWidth(1)
# c.rect(0, 0, page_mm, page_mm, stroke=1, fill=0)

# 绘制 AprilTag 标签
for side in ['left', 'right']:
    x_mm = args.margin if side == 'left' else (args.page_size - args.margin - args.tag_size)
    for i in range(args.tags_per_side):
        tag_id = args.start_id + i
        marker_px = int(args.tag_size * 10)
        img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, marker_px)
        pil = Image.fromarray(img)
        bio = io.BytesIO()
        pil.save(bio, format='PNG')
        bio.seek(0)
        reader = ImageReader(bio)
        # y 从顶部下移
        y_mm = args.page_size - args.margin - args.tag_size - i * (args.tag_size + vertical_spacing)
        c.drawImage(reader, x_mm * mm, y_mm * mm, width=args.tag_size * mm, height=args.tag_size * mm)

# 底部描述文字
desc = f"AprilTag {args.family}, IDs {args.start_id}-{args.start_id+args.tags_per_side-1}, tag size {args.tag_size}mm, checkerboard size {args.checker_size}mm"
c.setFont('Helvetica', 8)
c.drawCentredString(page_mm/2, 5 * mm, desc)

c.showPage()
c.save()
buf.seek(0)

# 写入文件
with open(args.output, 'wb') as f:
    f.write(buf.read())
print(f"Generated PDF: {args.output}")
