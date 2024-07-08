import numpy as np
from pathlib import Path
import openvino as ov
from IPython import display
import ipywidgets as widgets
from ultralytics import YOLO
import psutil
import cv2
import os
from PIL import Image
from run_ocr_single_image import run_paddle_ocr_single_image_ver2

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='CPU',
    description='Device:',
    disabled=False,
)

rec_model_file_path = "../Text_reg/namhoai96.xml"

# Read the model and corresponding weights from a file.
rec_model = core.read_model(model=rec_model_file_path)

# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")
# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

def run_paddle_ocr_folder(input_folder, output_folder, rec_compiled_model = '', rec_output_layer = ''):
    true_texts = []
    predicted_texts = []
    plate_det_model = YOLO('../best_openvino_model_det_plate/best_openvino_model')
    txt_det_model = YOLO('../best_openvino_model_text_det/best_openvino_model')

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        cpu_use_t = psutil.cpu_percent(interval=None)
        mem_t = psutil.virtual_memory().used / (1024 ** 1)
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)


        txts, plate_coor, txt_boxes = run_paddle_ocr_single_image_ver2(image_path, use_popup=False, plate_det_model = plate_det_model, txt_det_model = txt_det_model, rec_compiled_model= rec_compiled_model, rec_output_layer= rec_output_layer)
        predicted_texts.append(txts)


        if len(txts) == 2:
          cv2.putText(frame, str(txts[1] + " " + txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        elif len(txts) == 1:
          cv2.putText(frame, str(txts[0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        else:
          cv2.putText(frame, str(""), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


        if len(plate_coor) == 4:
          x1, y1, x2, y2 = plate_coor
          cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)


        x_offset, y_offset = plate_coor[0], plate_coor[1]
        adjusted_boxes = [box + np.array([x_offset, y_offset]) for box in txt_boxes]
        

        for box in adjusted_boxes:
          box = box.astype(int)  # Convert coordinates to integers
          cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)


        # Save thez modified frame to the output folder
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, frame)
        cpu_use_s = psutil.cpu_percent(interval=None)
        mem_s = psutil.virtual_memory().used / (1024 ** 1)

        print(f"Muc su dung: {cpu_use_s-cpu_use_t}%")
        print(f"Muc su dung mem: {mem_s-mem_t}MB")

    

    return true_texts, predicted_texts


input_folder = "../images"
output_folder = "../output"
true_texts, predicted_texts = run_paddle_ocr_folder(input_folder, output_folder, rec_compiled_model= rec_compiled_model, rec_output_layer= rec_output_layer)
