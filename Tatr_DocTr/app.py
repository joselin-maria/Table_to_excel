import asyncio
import string
import random
from collections import Counter
from itertools import count, tee
import base64
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import pytesseract
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from io import BytesIO

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')
st.title("Table Detection and Table Structure Recognition (Tesseract)")
st.write("Implemented by MSFT team: https://github.com/microsoft/table-transformer, with Tesseract for OCR")

table_detection_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection")
table_recognition_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition")

def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_PIL(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

async def pytess(cell_pil_img, threshold: float = 0.5):
    cell_pil_img = TableExtractionPipeline.add_padding(
        pil_img=cell_pil_img, top=50, right=50, bottom=50, left=50, color=(255, 255, 255))
    cell_image_np = np.array(cell_pil_img.convert("L"))
    text = pytesseract.image_to_string(cell_image_np, config='--psm 6')
    return text.strip().replace('\n', ' ')

def sharpen_image(pil_img):
    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img

def uniquify(seq, suffs=count(1)):
    not_unique = [k for k, v in Counter(seq).items() if v > 1]
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix
    return seq

def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)

def td_postprocess(pil_img):
    img = PIL_to_cv(pil_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))
    mask = mask & nzmask
    new_img = img.copy()
    new_img[np.where(mask)] = 255
    return cv_to_PIL(new_img)

def table_detector(image, THRESHOLD_PROBA):
    feature_extractor = DetrImageProcessor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = table_detection_model(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    for box in bboxes_scaled:
        print("Table detector bbox:", box.tolist())
        if len(box) != 4:
            raise ValueError(f"Expected 4 values in bbox, got {len(box)}: {box.tolist()}")
    return probas[keep], bboxes_scaled

def table_struct_recog(image, THRESHOLD_PROBA):
    feature_extractor = DetrImageProcessor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = table_recognition_model(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    for box in bboxes_scaled:
        print("Table structure bbox:", box.tolist())
        if len(box) != 4:
            raise ValueError(f"Expected 4 values in bbox, got {len(box)}: {box.tolist()}")
    return probas[keep], bboxes_scaled

class TableExtractionPipeline:
    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    @staticmethod
    def add_padding(pil_img, top, right, bottom, left, color=(255, 255, 255)):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    @staticmethod
    def dynamic_delta(xmin, ymin, xmax, ymax, delta_xmin, delta_ymin, delta_xmax, delta_ymax, pil_img):
        offset_x = (xmax - xmin) * 0.05
        offset_y = (ymax - ymin) * 0.05
        w_img, h_img = pil_img.size
        doxmin = max(0, xmin - (delta_xmin + offset_x))
        doymin = max(0, ymin - (delta_ymin + offset_y))
        doxmax = min(w_img, xmax + (delta_xmax + offset_x))
        doymax = min(h_img, ymax + (delta_ymax + offset_y))
        return doxmin, doymin, doxmax, doymax

    def plot_results_detection(self, c1, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, box in zip(prob, boxes.tolist()):
            print("Plot detection bbox:", box)
            xmin, ymin, xmax, ymax = box
            cl = p.argmax()
            xmin, ymin, xmax, ymax = self.dynamic_delta(xmin, ymin, xmax, ymax, delta_xmin, delta_ymin, delta_xmax, delta_ymax, pil_img)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin - 20, ymin - 50, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        c1.pyplot()

    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        cropped_img_list = []
        for p, box in zip(prob, boxes.tolist()):
            print("Crop tables bbox:", box)
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = self.dynamic_delta(xmin, ymin, xmax, ymax, delta_xmin, delta_ymin, delta_xmax, delta_ymax, pil_img)
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)
        return cropped_img_list

    def generate_structure(self, c2, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        plt.figure(figsize=(32, 20))
        plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0
        for p, box in zip(prob, boxes.tolist()):
            print("Generate structure bbox:", box)
            xmin, ymin, xmax, ymax = box
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            text = f'{class_text}: {p[cl]:0.2f}'
            if class_text in ['table row', 'table projected row header', 'table column']:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=2))
                ax.text(xmin - 10, ymin - 10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))
            if class_text == 'table row':
                rows['table row.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)
            idx += 1
        plt.axis('on')
        c2.pyplot()
        return rows, cols

    def sort_table_featuresv2(self, rows: dict, cols: dict):
        rows_ = {table_feature: 
                 (xmin, ymin, xmax, ymax) 
                 for table_feature, 
                 (xmin, ymin, xmax, ymax) 
                 in sorted(rows.items(), 
                           key=lambda tup: tup[1][1])}
        cols_ = {table_feature: (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}
        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows: dict, cols: dict):
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img
        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img
        return rows, cols

    def object_to_cellsv2(self, master_row: dict, cols: dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row
        for k_row, v_row in new_master_row.items():
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols) - 1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb
                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)
            cells_img[k_row + '.' + str(row_idx)] = row_img_list
            row_idx += 1
        return cells_img, len(new_cols), len(new_master_row) - 1

    def clean_dataframe(self, df):
        for col in df.columns:
            df[col] = df[col].str.replace("'", '', regex=True)
            df[col] = df[col].str.replace('"', '', regex=True)
            df[col] = df[col].str.replace(']', '', regex=True)
            df[col] = df[col].str.replace('[', '', regex=True)
            df[col] = df[col].str.replace('{', '', regex=True)
            df[col] = df[col].str.replace('}', '', regex=True)
        return df

    @st.cache_data
    def convert_df(_self, df):
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return output.getvalue()

    def create_dataframe(self, c3, cell_ocr_res: list, max_cols: int, max_rows: int):
        headers = cell_ocr_res[:max_cols]
        new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0
        cells_list = cell_ocr_res[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)
        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1
        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]
        df = self.clean_dataframe(df)
        c3.dataframe(df)
        excel_data = self.convert_df(df)
        try:
            numkey = str(df.iloc[0, 0])
        except IndexError:
            numkey = str(0)
        filename = f"table_{numkey}.xlsx"
        output_path = f"output/{filename}"
        with open(output_path, "wb") as f:
            f.write(excel_data)
        c3.success(f"Excel saved to {output_path}")
        return df

    async def start_process(self, image_path: str, TD_THRESHOLD, TSR_THRESHOLD,
                            OCR_THRESHOLD, padd_top, padd_left, padd_bottom,
                            padd_right, delta_xmin, delta_ymin, delta_xmax,
                            delta_ymax, expand_rowcol_bbox_top,
                            expand_rowcol_bbox_bottom):
        image = Image.open(image_path).convert("RGB")
        probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)
        if bboxes_scaled.nelement() == 0:
            st.write('No table found in the image')
            return ''
        c1, c2, c3 = st.columns((1, 1, 1))
        self.plot_results_detection(c1, table_detection_model, image, probas,
                                    bboxes_scaled, delta_xmin, delta_ymin,
                                    delta_xmax, delta_ymax)
        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled,
                                            delta_xmin, delta_ymin, delta_xmax,
                                            delta_ymax)
        for idx, unpadded_table in enumerate(cropped_img_list):
            table = self.add_padding(unpadded_table, padd_top, padd_right,
                                     padd_bottom, padd_left)
            probas, bboxes_scaled = table_struct_recog(
                table, THRESHOLD_PROBA=TSR_THRESHOLD)
            rows, cols = self.generate_structure(c2, table_recognition_model,
                                                 table, probas, bboxes_scaled,
                                                 expand_rowcol_bbox_top,
                                                 expand_rowcol_bbox_bottom)
            rows, cols = self.sort_table_featuresv2(rows, cols)
            master_row, cols = self.individual_table_featuresv2(table, rows, cols)
            cells_img, max_cols, max_rows = self.object_to_cellsv2(
                master_row, cols, expand_rowcol_bbox_top,
                expand_rowcol_bbox_bottom, padd_left)
            sequential_cell_img_list = []
            for k, img_list in cells_img.items():
                for img in img_list:
                    sequential_cell_img_list.append(
                        pytess(cell_pil_img=img, threshold=OCR_THRESHOLD))
            cell_ocr_res = await asyncio.gather(*sequential_cell_img_list)
            self.create_dataframe(c3, cell_ocr_res, max_cols, max_rows)
            st.write('Errors in OCR may be due to image quality or OCR performance')

if __name__ == "__main__":
    st.title("Process Static Table Image")
    input_image = "static/2.png"
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(input_image):
        st.image(input_image, caption="Input Image")
        if st.button("Convert to Excel"):
            with st.spinner("Processing..."):
                try:
                    te = TableExtractionPipeline()
                    asyncio.run(
                        te.start_process(
                            input_image,
                            TD_THRESHOLD=0.8,
                            TSR_THRESHOLD=0.7,
                            OCR_THRESHOLD=0.5,
                            padd_top=90,
                            padd_left=40,
                            padd_bottom=90,
                            padd_right=40,
                            delta_xmin=10,
                            delta_ymin=3,
                            delta_xmax=10,
                            delta_ymax=3,
                            expand_rowcol_bbox_top=0,
                            expand_rowcol_bbox_bottom=0
                        )
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.error("Input image 'static/2.png' not found.")