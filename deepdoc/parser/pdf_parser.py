#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import random
import re
import sys
import threading
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer

import numpy as np
import pdfplumber
import trio
import xgboost as xgb
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader as pdf2_read
from api import settings
from api.utils.file_utils import get_project_base_directory
from deepdoc.vision import OCR, LayoutRecognizer, Recognizer, TableStructureRecognizer
from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk
from rag.nlp import rag_tokenizer
from rag.prompts import vision_llm_describe_prompt
from rag.settings import PARALLEL_DEVICES

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class RAGFlowPdfParser:
    def __init__(self, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

        self.ocr = OCR()
        self.parallel_limiter = None
        if PARALLEL_DEVICES is not None and PARALLEL_DEVICES > 1:
            self.parallel_limiter = [trio.CapacityLimiter(1) for _ in range(PARALLEL_DEVICES)]

        if hasattr(self, "model_speciess"):
            self.layouter = LayoutRecognizer("layout." + self.model_speciess)
        else:
            self.layouter = LayoutRecognizer("layout")
        self.tbl_det = TableStructureRecognizer()
        self.updown_cnt_mdl = xgb.Booster()
        if not settings.LIGHTEN:
            try:
                import torch.cuda
                if torch.cuda.is_available():
                    self.updown_cnt_mdl.set_param({"device": "cuda"})
            except Exception:
                logging.exception("RAGFlowPdfParser __init__")
        try:
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
        except Exception:
            model_dir = snapshot_download(
                repo_id="InfiniFlow/text_concat_xgb_v1.0",
                local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))

        self.page_from = 0

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def _height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(
            self, a, b):
        return (
            b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"chapter[0-9]",
            r"article[0-9]",
            r"section[0-9]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any(re.match(p, b["text"]) for p in proj_patt)

    def _updown_concat_features(self, up, down):
        w = max(self._char_width(up), self._char_width(down))
        h = max(self._height(up), self._height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = up["text"][-LEN:].strip() \
            + (" " if re.match(r"[a-zA-Z0-9]+",
                               up["text"][-1] + down["text"][0]) else "") \
            + down["text"][:LEN].strip()
        tks_all = rag_tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(
                r"([。？！；!?;+)）]|[a-z]\.)$",
                up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            True if re.search(
                r"(^.?[/,?;:\]，。；：’”？！》】）-])",
                down["text"]) else False,
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[\(（][^\)）]+$", up["text"])
            and re.search(r"[\)）]", down["text"]) else False,
            self._match_proj(down),
            re.match(r"[A-Z]", down["text"]) is not None,
            re.match(r"[A-Z]", up["text"][-1]) is not None,
            re.match(r"[a-z0-9]", up["text"][-1]) is not None,
            re.match(r"[0-9.%,-]+$", down["text"]) is not None,
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],
            abs(self._height(up) - self._height(down)) / min(self._height(up), self._height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"])) / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threshold):
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold and arr[j + 1]["top"] < arr[j]["top"] and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def _table_transformer_job(self, zoom_factor):
        logging.debug("Table processing...")
        imgs, positions = [], []
        table_counts = [0]
        MARGIN = 10
        self.table_components = []
        assert len(self.page_layout) == len(self.page_images)
        for page_index, tables in enumerate(self.page_layout):
            tables = [table for table in tables if table["type"] == "table"]
            table_counts.append(len(tables))
            if not tables:
                continue
            for tb in tbls:  # for table
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, \
                    tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top))
                imgs.append(self.page_images[p].crop((left, top, right, bott)))

        assert len(self.page_images) == len(table_counts) - 1
        if not imgs:
            return
        recognitions = self.tbl_det(imgs)
        table_counts = np.cumsum(table_counts)
        for i in range(len(table_counts) - 1):
            page_tables = []
            for j, table_items in enumerate(recognitions[table_counts[i]: table_counts[i + 1]]):
                positions_page = positions[table_counts[i]: table_counts[i + 1]]
                for item in table_items:
                    item["x0"] = (item["x0"] + positions_page[j][0])
                    item["x1"] = (item["x1"] + positions_page[j][0])
                    item["top"] = (item["top"] + positions_page[j][1])
                    item["bottom"] = (item["bottom"] + positions_page[j][1])
                    for dimension in ["x0", "x1", "top", "bottom"]:
                        item[dimension] /= zoom_factor
                    item["top"] += self.page_cum_height[i]
                    item["bottom"] += self.page_cum_height[i]
                    item["pn"] = i
                    item["layoutno"] = j
                    page_tables.append(item)
            self.table_components.extend(page_tables)

        def gather(keyword, fuzzy=10, proportion=0.6):
            elements = Recognizer.sort_Y_firstly([r for r in self.table_components if re.match(keyword, r["label"])], fuzzy)
            elements = Recognizer.layouts_cleanup(self.boxes, elements, 5, proportion)
            return Recognizer.sort_Y_firstly(elements, 0)

        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        columns = sorted([r for r in self.table_components if re.match(r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        columns = Recognizer.layouts_cleanup(self.boxes, columns, 5, 0.5)
        for box in self.boxes:
            if box.get("layout_type", "") != "table":
                continue
            row_index = Recognizer.find_overlapped_with_threshold(box, rows, threshold=0.3)
            if row_index is not None:
                box["R"] = row_index
                box["R_top"] = rows[row_index]["top"]
                box["R_bottom"] = rows[row_index]["bottom"]

            header_index = Recognizer.find_overlapped_with_threshold(box, headers, threshold=0.3)
            if header_index is not None:
                box["H_top"] = headers[header_index]["top"]
                box["H_bottom"] = headers[header_index]["bottom"]
                box["H_left"] = headers[header_index]["x0"]
                box["H_right"] = headers[header_index]["x1"]
                box["H"] = header_index

            column_index = Recognizer.find_horizontally_tightest_fit(box, columns)
            if column_index is not None:
                box["C"] = column_index
                box["C_left"] = columns[column_index]["x0"]
                box["C_right"] = columns[column_index]["x1"]

            span_index = Recognizer.find_overlapped_with_threshold(box, spans, threshold=0.3)
            if span_index is not None:
                box["H_top"] = spans[span_index]["top"]
                box["H_bottom"] = spans[span_index]["bottom"]
                box["H_left"] = spans[span_index]["x0"]
                box["H_right"] = spans[span_index]["x1"]
                box["SP"] = span_index

    def _ocr(self, page_number, image, characters, zoom_factor=3):
    #Enhanced OCR processing with better error handling and performance.
    
        # Configurable thresholds
        SIZE_MISMATCH_THRESHOLD = 0.7  # Max size difference ratio
        SPACE_INSERTION_CHARS = r"[0-9a-zA-Zа-яА-Я,.?;:!%%]"
        MIN_BOX_AREA = 4  # Minimum area for a valid box (width*height in pixels)
        
        try:
            # Convert image to numpy array once
            image_np = np.array(image)
            
            # Detect text regions with error handling
            try:
                detected_lines = self.ocr.detect(image_np)
                if not detected_lines:
                    self.boxes.append([])
                    logging.debug(f"No text detected on page {page_number}")
                    return
            except Exception as e:
                logging.error(f"OCR detection failed on page {page_number}: {str(e)}")
                self.boxes.append([])
                return

            # Process detected boxes
            valid_boxes = []
            for line in detected_lines:
                b, t = line[0], line[1][0]
                # Validate box coordinates and minimum size
                if (b[0][0] <= b[1][0] and b[0][1] <= b[-1][1] and 
                    (b[1][0] - b[0][0]) * (b[-1][1] - b[0][1]) >= MIN_BOX_AREA):
                    valid_boxes.append((b, t))
            
            # Convert and sort boxes
            boxes = [
                {
                    "x0": b[0][0] / zoom_factor,
                    "x1": b[1][0] / zoom_factor,
                    "top": b[0][1] / zoom_factor,
                    "bottom": b[-1][1] / zoom_factor,
                    "text": "",
                    "txt": t,  # Temporary storage
                    "page_number": page_number,
                    "confidence": t[1] if isinstance(t, tuple) else 1.0  # Store confidence if available
                }
                for b, t in valid_boxes
            ]
            
            # Sort boxes with optimized threshold calculation
            sort_threshold = self.mean_height[-1] / 3 if self.mean_height else 10
            boxes = Recognizer.sort_Y_firstly(boxes, sort_threshold)

            # Pre-calculate mean height for current page if not available
            current_page_mean_height = (
                self.mean_height[page_number - 1] 
                if page_number - 1 < len(self.mean_height) and self.mean_height[page_number - 1] > 0
                else 12  # Default value
            )

            # Process characters in batches for better performance
            sorted_chars = Recognizer.sort_Y_firstly(characters, current_page_mean_height // 4)
            char_index = 0
            total_chars = len(sorted_chars)
            
            while char_index < total_chars:
                char = sorted_chars[char_index]
                char_index += 1
                
                # Find matching box
                box_idx = Recognizer.find_overlapped(char, boxes)
                if box_idx is None:
                    self.lefted_chars.append(char)
                    continue

                # Calculate sizes once
                box = boxes[box_idx]
                char_height = char["bottom"] - char["top"]
                box_height = box["bottom"] - box["top"]
                size_ratio = abs(char_height - box_height) / max(char_height, box_height)
                
                # Skip characters with significant size mismatch
                if size_ratio >= SIZE_MISMATCH_THRESHOLD and char["text"] != ' ':
                    self.lefted_chars.append(char)
                    continue

                # Handle space insertion more intelligently
                if char["text"] == " ":
                    if box["text"]:  # Only add space if there's existing text
                        last_char = box["text"][-1]
                        # Add space only between certain characters
                        if re.match(SPACE_INSERTION_CHARS, last_char):
                            # Don't add duplicate spaces
                            if not box["text"].endswith(" "):
                                box["text"] += " "
                else:
                    # Validate character before adding
                    if char["text"].strip():  # Skip empty/whitespace-only chars
                        box["text"] += char["text"]

            # Process empty boxes with fallback OCR
            for box in boxes:
                if not box["text"]:
                    try:
                        # Convert coordinates once
                        left = int(box["x0"] * zoom_factor)
                        top = int(box["top"] * zoom_factor)
                        right = int(box["x1"] * zoom_factor)
                        bottom = int(box["bottom"] * zoom_factor)
                        
                        # Validate coordinates
                        if right > left and bottom > top:
                            polygon = np.array([
                                [left, top],
                                [right, top],
                                [right, bottom],
                                [left, bottom]
                            ], dtype=np.float32)
                            
                            # Perform OCR with error handling
                            box["text"] = self.ocr.recognize(image_np, polygon) or ""
                    except Exception as e:
                        logging.warning(f"Fallback OCR failed for box on page {page_number}: {str(e)}")
                        box["text"] = ""

            # Clean up and filter boxes
            final_boxes = []
            for box in boxes:
                if "txt" in box:
                    del box["txt"]
                if box["text"].strip():  # Only keep boxes with actual content
                    final_boxes.append(box)
            
            # Update mean height if needed
            if not self.mean_height or self.mean_height[-1] == 0:
                heights = [box["bottom"] - box["top"] for box in final_boxes]
                if heights:
                    self.mean_height.append(np.median(heights))
                else:
                    self.mean_height.append(current_page_mean_height)  # Fallback
            
            self.boxes.append(final_boxes)
            
        except Exception as e:
            logging.error(f"Unexpected error in _ocr for page {page_number}: {str(e)}")
            self.boxes.append([])

    def _layouts_rec(self, zoom_factor, drop=True):
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, zoom_factor, drop=drop)
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _text_merge(self):
        boxes = self.boxes

        def end_with(box, text):
            text = text.strip()
            text_box = box.get("text", "").strip()
            return text_box and text_box.find(text) == len(text_box) - len(text)

        def start_with(box, texts):
            text_box = box.get("text", "").strip()
            return text_box and any(text_box.find(t.strip()) == 0 for t in texts)

        i = 0
        while i < len(boxes) - 1:
            box = boxes[i]
            next_box = boxes[i + 1]
            if box.get("layoutno", "0") != next_box.get("layoutno", "1") or box.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue
            if abs(self._y_dis(box, next_box)) < self.mean_height[boxes[i]["page_number"] - 1] / 3:
                boxes[i]["x1"] = next_box["x1"]
                boxes[i]["top"] = (box["top"] + next_box["top"]) / 2
                boxes[i]["bottom"] = (box["bottom"] + next_box["bottom"]) / 2
                boxes[i]["text"] += next_box["text"]
                boxes.pop(i + 1)
                continue
            i += 1

    def _naive_vertical_merge(self):
        boxes = Recognizer.sort_Y_firstly(self.boxes, np.median(self.mean_height) / 3)
        i = 0
        while i + 1 < len(boxes):
            box = boxes[i]
            next_box = boxes[i + 1]
            if box["page_number"] < next_box["page_number"] and re.match(r"[0-9  •一—-]+$", box["text"]):
                boxes.pop(i)
                continue
            if not box["text"].strip():
                boxes.pop(i)
                continue
            concatting_features = [
                box["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                len(box["text"].strip()) > 1 and box["text"].strip()[-2] in ",;:'\"，‘“、；：",
                next_box["text"].strip() and next_box["text"].strip()[0] in "。；？！?”）),，、：",
            ]
            feats = [
                box.get("layoutno", 0) != next_box.get("layoutno", 0),
                box["text"].strip()[-1] in "。？！?",
                self.is_english and box["text"].strip()[-1] in ".!?",
                box["page_number"] == next_box["page_number"] and next_box["top"] - box["bottom"] > self.mean_height[box["page_number"] - 1] * 1.5,
                box["page_number"] < next_box["page_number"] and abs(box["x0"] - next_box["x0"]) > self.mean_width[box["page_number"] - 1] * 4,
            ]
            # split features
            detach_feats = [b["x1"] < b_["x0"],
                            b["x0"] > b_["x1"]]
            if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                logging.debug("{} {} {} {}".format(
                    b["text"],
                    b_["text"],
                    any(feats),
                    any(concatting_feats),
                ))
                i += 1
                continue
            box["bottom"] = next_box["bottom"]
            box["text"] += next_box["text"]
            box["x0"] = min(box["x0"], next_box["x0"])
            box["x1"] = max(box["x1"], next_box["x1"])
            boxes.pop(i + 1)
        self.boxes = boxes

    def _concat_downward(self, concat_between_pages=True):
        for i in range(len(self.boxes)):
            mean_height = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                y_dis = self._y_dis(self.boxes[i], self.boxes[j]) / mean_height
                if abs(y_dis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif y_dis > 0:
                    break
                j += 1

        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, down_index):
                chunks.append(up)
                i = down_index
                while i < min(down_index + 12, len(boxes)):
                    y_dis = self._y_dis(up, boxes[i])
                    same_page = up["page_number"] == boxes[i]["page_number"]
                    mean_height = self.mean_height[up["page_number"] - 1]
                    mean_width = self.mean_width[up["page_number"] - 1]
                    if same_page and y_dis > mean_height * 4:
                        break
                    if not same_page and y_dis > mean_height * 16:
                        break
                    if not concat_between_pages and boxes[i]["page_number"] > up["page_number"]:
                        break
                    if up.get("R", "") != boxes[i].get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue
                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) or re.match(r"[0-9]{2,3}/[0-9]{3}$", boxes[i]["text"]) or not boxes[i]["text"].strip():
                        i += 1
                        continue
                    if not boxes[i]["text"].strip() or not up["text"].strip():
                        i += 1
                        continue
                    if up["x1"] < boxes[i]["x0"] - 10 * mean_width or up["x0"] > boxes[i]["x1"] + 10 * mean_width:
                        i += 1
                        continue
                    if i - down_index < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == boxes[i].get("layoutno", "2"):
                            dfs(boxes[i], i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue
                    features = self._updown_concat_features(up, boxes[i])
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([features]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(boxes[i], i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        boxes = []
        for block in blocks:
            if len(block) == 1:
                boxes.append(block[0])
                continue
            text = block[0]
            for chunk in block[1:]:
                text["text"] = text["text"].strip()
                chunk["text"] = chunk["text"].strip()
                if not chunk["text"]:
                    continue
                if text["text"] and re.match(r"[0-9\.a-zA-Z]+$", text["text"][-1] + chunk["text"][-1]):
                    text["text"] += " "
                text["text"] += chunk["text"]
                text["x0"] = min(text["x0"], chunk["x0"])
                text["x1"] = max(text["x1"], chunk["x1"])
                text["page_number"] = min(text["page_number"], chunk["page_number"])
                text["bottom"] = chunk["bottom"]
                if not text["layout_type"] and chunk["layout_type"]:
                    text["layout_type"] = chunk["layout_type"]
            boxes.append(text)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return
        find_it = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(contents|table of contents|schedules|exibits|acknowledge)$", re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            find_it = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if find_it:
            return

        page_dirty = [0] * len(self.page_images)
        for box in self.boxes:
            if re.search(r"(··|··|··)", box["text"]):
                page_dirty[box["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, count in enumerate(page_dirty) if count > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            box = self.boxes[i]
            next_box = self.boxes[i + 1]
            if not box["text"].strip():
                self.boxes.pop(i)
                continue
            if not next_box["text"].strip():
                self.boxes.pop(i + 1)
                continue
            if box["text"].strip()[0] != next_box["text"].strip()[0] or box["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm") or rag_tokenizer.is_chinese(box["text"].strip()[0]) or box["top"] > next_box["bottom"]:
                i += 1
                continue
            next_box["text"] = box["text"] + "\n" + next_box["text"]
            next_box["x0"] = min(box["x0"], next_box["x0"])
            next_box["x1"] = max(box["x1"], next_box["x1"])
            next_box["top"] = box["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, need_image, ZM, return_html, need_position, separate_tables_figures=False):
        tables = {}
        figures = {}
        i = 0
        last_layout_no = ""
        no_merge_layout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + \
                "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption",
                                                                                                      "title",
                                                                                                      "figure caption",
                                                                                                      "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(data|information|charts)*source[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if layout_no not in tables:
                    tables[layout_no] = []
                tables[layout_no].append(self.boxes[i])
                self.boxes.pop(i)
                last_layout_no = layout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(data|information|charts)*source[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if layout_no not in figures:
                    figures[layout_no] = []
                figures[layout_no].append(self.boxes[i])
                self.boxes.pop(i)
                last_layout_no = layout_no
                continue
            i += 1

        no_merge_layout_no = set(no_merge_layout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in no_merge_layout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mean_height = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mean_height * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        i = 0
        while i < len(self.boxes):
            caption = self.boxes[i]
            if not TableStructureRecognizer.is_caption(caption):
                i += 1
                continue

            def nearest(tbls):
                nonlocal caption
                min_key = ""
                min_value = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(caption, b)
                        x_dis = self._x_dis(caption, b) if not x_overlapped(caption, b) else 0
                        distance = y_dis * y_dis + x_dis * x_dis
                        if distance < min_value:
                            min_key = k
                            min_value = distance
                return min_key, min_value

            table_key, table_value = nearest(tables)
            figure_key, figure_value = nearest(figures)
            if table_value < figure_value and table_key:
                tables[table_key].insert(0, caption)
                logging.debug(f"TABLE: {self.boxes[i]['text']}; Cap: {table_key}")
            elif figure_key:
                figures[figure_key].insert(0, caption)
                logging.debug(f"FIGURE: {self.boxes[i]['text']}; Cap: {figure_key}")
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {
                    "x0": np.min([b["x0"] for b in bxs]),
                    "top": np.min([b["top"] for b in bxs]) - height,
                    "x1": np.max([b["x1"] for b in bxs]),
                    "bottom": np.max([b["bottom"] for b in bxs]) - height
                }
                layouts = [layout for layout in self.page_layout[page_number] if layout["type"] == layout_type]
                index = Recognizer.find_overlapped(bounding_box, layouts, naive=True)
                if index is not None:
                    bounding_box = layouts[index]
                else:
                    logging.warning(f"Missing layout match: {page_number + 1}, {bxs[0].get('layoutno', '')}")

                left, top, right, bottom = bounding_box["x0"], bounding_box["top"], bounding_box["x1"], bounding_box["bottom"]
                if right < left:
                    right = left + 1
                poss.append((page_number + self.page_from, left, right, top, bottom))
                return self.page_images[page_number].crop((left * zoom_factor, top * zoom_factor, right * zoom_factor, bottom * zoom_factor))
            page_numbers = sorted(page_numbers.items(), key=lambda x: x[0])
            images = [cropout(arr, layout_type, poss) for page_number, arr in page_numbers]
            image = Image.new("RGB", (int(np.max([img.size[0] for img in images])), int(np.sum([img.size[1] for img in images]))), (245, 245, 245))
            height = 0
            for img in images:
                image.paste(img, (0, int(height)))
                height += img.size[1]
            return image

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue
            poss = []

            if separate_tables_figures:
                figure_results.append(
                    (cropout(
                        bxs,
                        "figure", poss),
                     [txt]))
                figure_positions.append(poss)
            else:
                res.append(
                    (cropout(
                        bxs,
                        "figure", poss),
                     [txt]))
                positions.append(poss)

        for key, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean(
                [(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            res.append((cropout(bxs, "table", poss),
                        self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        patterns = [
            (r"chapter[0-9]", 1),
            (r"article[0-9]", 2),
            (r"section[0-9]", 3),
            (r"[0-9]+[, ]", 4),
            (r"[\(（][0-9]+[）\)]", 5),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 6),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 9),
            (r".{,48}[：:?？]$", 10),
            (r"[0-9]+）", 11),
            (r"[⚫•➢✓]", 12)
        ]
        for pattern, index in patterns:
            if re.match(pattern, line):
                return index
        return

    def _line_tag(self, box, zoom_factor):
        page_numbers = [box["page_number"]]
        top = box["top"] - self.page_cum_height[page_numbers[0] - 1]
        bottom = box["bottom"] - self.page_cum_height[page_numbers[0] - 1]
        page_images_count = len(self.page_images)
        if page_numbers[-1] - 1 >= page_images_count:
            return ""
        while bottom * zoom_factor > self.page_images[page_numbers[-1] - 1].size[1]:
            bottom -= self.page_images[page_numbers[-1] - 1].size[1] / zoom_factor
            page_numbers.append(page_numbers[-1] + 1)
            if page_numbers[-1] - 1 >= page_images_count:
                return ""
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join(map(str, page_numbers)), box["x0"], box["x1"], top, bottom)

    def __filterout_scraps(self, boxes, zoom_factor):
        def width(box):
            return box["x1"] - box["x0"]

        def height(box):
            return box["bottom"] - box["top"]

        def useful(box):
            if box.get("layout_type"):
                return True
            if width(box) > self.page_images[box["page_number"] - 1].size[0] / zoom_factor / 3:
                return True
            if height(box) > self.mean_height[box["page_number"] - 1]:
                return True
            return False

        results = []
        while boxes:
            lines = []
            widths = []
            page_width = self.page_images[boxes[0]["page_number"] - 1].size[0] / zoom_factor
            mean_height = self.mean_height[boxes[0]["page_number"] - 1]
            match = self.proj_match(boxes[0]["text"]) or boxes[0].get("layout_type", "") == "title"

            def dfs(line, start):
                nonlocal mean_height, page_width, lines, widths
                lines.append(line)
                widths.append(width(line))
                match = self.proj_match(line["text"]) or line.get("layout_type", "") == "title"
                for i in range(start + 1, min(start + 20, len(boxes))):
                    if boxes[i]["page_number"] - line["page_number"] > 0:
                        break
                    if not match and self._y_dis(line, boxes[i]) >= 3 * mean_height and height(line) < 1.5 * mean_height:
                        break
                    if not useful(boxes[i]):
                        continue
                    if match or self._x_dis(boxes[i], line) < page_width / 10:
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if useful(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug(f"WASTE: {boxes[0]['text']}")
            except Exception:
                pass
            boxes.pop(0)
            mean_width = np.mean(widths)
            if match or mean_width / page_width >= 0.35 or mean_width > 200:
                results.append("\n".join([box["text"] + self._line_tag(box, zoom_factor) for box in lines]))
            else:
                logging.debug(f"REMOVED: {'<<'.join(box['text'] for box in lines)}")

        return "\n\n".join(results)

    @staticmethod
    def total_page_number(filename, binary=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = pdfplumber.open(
                    fnm) if not binary else pdfplumber.open(BytesIO(binary))
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logging.exception("total_page_number")

    def __images__(self, filename, zoom_factor=3, page_from=0, page_to=299, callback=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        start = timer()
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(
                    fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                                    enumerate(self.pdf.pages[page_from:page_to])]
                try:
                    self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
                except Exception as e:
                    logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                    self.page_chars = [[] for _ in range(page_to - page_from)]  # If failed to extract, using empty list instead.

                self.total_page = len(self.pdf.pages)
        except Exception:
            logging.exception("RAGFlowPdfParser __images__")
        logging.info(f"__images__ dedupe_chars cost {timer() - start}s")

        self.outlines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            outlines = self.pdf.outline

            def dfs(arr, depth):
                for item in arr:
                    if isinstance(item, dict):
                        self.outlines.append((item["/Title"], depth))
                        continue
                    dfs(item, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        finally:
            self.pdf.close()
        if not self.outlines:
            logging.warning("Miss outlines")

        logging.debug("Images converted.")
        self.is_english = [re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(
            random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i]))))) for i in
            range(len(self.page_chars))]
        if sum([1 if e else 0 for e in self.is_english]) > len(
                self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        async def __img_ocr(i, id, img, chars, limiter):
            j = 0
            while j + 1 < len(chars):
                if chars[j]["text"] and chars[j + 1]["text"] and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"]) and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2:
                    chars[j]["text"] += " "
                j += 1

            if limiter:
                async with limiter:
                    await trio.to_thread.run_sync(lambda: self.__ocr(i + 1, img, chars, zoomin, id))
            else:
                self.__ocr(i + 1, img, chars, zoomin, id)

            if callback and i % 6 == 5:
                callback(prog=(i + 1) * 0.6 / len(self.page_images), msg="")

        async def __img_ocr_launcher():
            def __ocr_preprocess():
                chars = self.page_chars[i] if not self.is_english else []
                self.mean_height.append(
                    np.median(sorted([c["height"] for c in chars])) if chars else 0
                )
                self.mean_width.append(
                    np.median(sorted([c["width"] for c in chars])) if chars else 8
                )
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars

            if self.parallel_limiter:
                async with trio.open_nursery() as nursery:
                    for i, img in enumerate(self.page_images):
                        chars = __ocr_preprocess()

                        nursery.start_soon(__img_ocr, i, i % PARALLEL_DEVICES, img, chars,
                                           self.parallel_limiter[i % PARALLEL_DEVICES])
                        await trio.sleep(0.1)
            else:
                for i, img in enumerate(self.page_images):
                    chars = __ocr_preprocess()
                    await __img_ocr(i, 0, img, chars, None)

        start = timer()

        trio.run(__img_ocr_launcher)

        logging.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")

        if not self.is_english and not any([char for char in self.page_chars]) and self.boxes:
            boxes = [box for boxes in self.boxes for box in boxes]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices(boxes, k=min(30, len(boxes)))))

        logging.debug(f"Is it English: {self.is_english}")

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        if len(self.boxes) == 0 and zoom_factor < 9:
            self.__images__(filename, zoom_factor * 3, page_from, page_to, callback)

    def __call__(self, filename, need_image=True, zoom_factor=3, return_html=False):
        self.__images__(filename, zoom_factor)
        self._layouts_rec(zoom_factor)
        self._table_transformer_job(zoom_factor)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tables = self._extract_table_figure(need_image, zoom_factor, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoom_factor), tables

    def remove_tag(self, text):
        return re.sub(r"@@[\t0-9.-]+?##", "", text)

    def crop(self, text, zoom_factor=3, need_position=False):
        images = []
        positions = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", text):
            page_numbers, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            positions.append(([int(p) - 1 for p in page_numbers.split("-")], left, right, top, bottom))
        if not positions:
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in positions]), 6)
        GAP = 6
        pos = positions[0]
        positions.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = positions[-1]
        positions.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / zoom_factor, pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1] / zoom_factor, pos[4] + 120)))

        positions_result = []
        for ii, (pns, left, right, top, bottom) in enumerate(positions):
            right = left + max_width
            bottom *= zoom_factor
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(
                self.page_images[pns[0]].crop((left * ZM, top * ZM,
                                               right *
                                               ZM, min(
                                                   bottom, self.page_images[pns[0]].size[1])
                                               ))
            )
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(
                    bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                images.append(self.page_images[pn].crop((left * zoom_factor, 0, right * zoom_factor, min(bottom, self.page_images[pn].size[1]))))
                if 0 < ii < len(positions) - 1:
                    positions_result.append((pn + self.page_from, left, right, 0, min(bottom, self.page_images[pn].size[1]) / zoom_factor))
                bottom -= self.page_images[pn].size[1]

        if not images:
            if need_position:
                return None, None
            return

        height = 0
        for img in images:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([img.size[0] for img in images]))
        image = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(images):
            if ii == 0 or ii + 1 == len(images):
                img = img.convert('RGBA')
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            image.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return image, positions_result
        return image

    def get_position(self, box, zoom_factor):
        positions = []
        page_number = box["page_number"]
        top = box["top"] - self.page_cum_height[page_number - 1]
        bottom = box["bottom"] - self.page_cum_height[page_number - 1]
        positions.append((page_number, box["x0"], box["x1"], top, min(bottom, self.page_images[page_number - 1].size[1] / zoom_factor)))
        while bottom * zoom_factor > self.page_images[page_number - 1].size[1]:
            bottom -= self.page_images[page_number - 1].size[1] / zoom_factor
            top = 0
            page_number += 1
            positions.append((page_number, box["x0"], box["x1"], top, min(bottom, self.page_images[page_number - 1].size[1] / zoom_factor)))
        return positions

class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([text for text in page.extract_text().split("\n") if text.strip()])
            outlines = self.pdf.outline

            def dfs(arr, depth):
                for item in arr:
                    if isinstance(item, dict):
                        self.outlines.append((item["/Title"], depth))
                        continue
                    dfs(item, depth + 1)

            dfs(outlines, 0)
        except Exception:
            logging.exception("Outlines exception")
        if not self.outlines:
            logging.warning("Miss outlines")

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(
                    fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                                    enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        self.__images__(fnm=filename, zoomin=3, page_from=from_page, page_to=to_page, **kwargs)

        total_pdf_pages = self.total_page

        start_page = max(0, from_page)
        end_page = min(to_page, total_pdf_pages)

        all_docs = []

        for idx, img_binary in enumerate(self.page_images or []):
            pdf_page_num = idx  # 0-based
            if pdf_page_num < start_page or pdf_page_num >= end_page:
                continue

            docs = picture_vision_llm_chunk(
                binary=img_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_describe_prompt(page=pdf_page_num+1),
                callback=callback,
            )

            if docs:
                all_docs.append(docs)
        return [(doc, "") for doc in all_docs], []


if __name__ == "__main__":
    pass
