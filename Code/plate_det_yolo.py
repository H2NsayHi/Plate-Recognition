from image_cv2 import pil_to_cv2

def plate_det_yolo(model, image):
  
  pd = 20

  prediction = model.predict(source=image, conf=0.5, iou=0.5)

  # Convert PIL image to OpenCV format
  img_cv2 = pil_to_cv2(image)

  # Iterate over each prediction result
  for result in prediction:
      boxes = result.boxes.cpu().numpy()
      xyxy = boxes.xyxy

      # Draw bounding boxes on the predicted boxes
      for box in xyxy:
          x1, y1, x2, y2 = int(box[0]-pd), int(box[1]-pd), int(box[2]+pd), int(box[3]+pd)
          cropped_image = img_cv2[y1:y2, x1:x2]
          print(box)
          return cropped_image, [x1, y1, x2, y2]