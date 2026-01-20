import cv2
import math
import numpy as np
import onnxruntime

class Segment:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.onnx_path = path

        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.get_input_output_details()
    
    def get_input_output_details(self):
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()

        self.input_names = [input_.name for input_ in model_inputs]
        self.output_names = [output.name for output in model_outputs]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        # print(boxes[indices].shape, scores[indices].shape, class_ids[indices].shape, mask_predictions[indices].shape)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]


    def process_box_output_26(self, box_output):
        detections = box_output[0]
        # print(detections.shape)
        boxes = []
        scores = []
        class_ids = []
        mask_predictions = []

        for det in detections:
            x1, y1, x2, y2, score, class_id = det[:6]

            mask_prediction = det[6:]

            if score < self.conf_threshold:
                continue

            # x1 = (x1 - pad_w) / scale
            # y1 = (y1 - pad_h) / scale
            # x2 = (x2 - pad_w) / scale
            # y2 = (y2 - pad_h) / scale
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(score))
            class_ids.append(int(class_id))
            mask_predictions.append(mask_prediction)

        boxes = np.array(boxes)
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # # Convert boxes to xyxy format
        # boxes = xywh2xyxy(boxes)
        #
        # # Check the boxes are within the image
        # boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        # boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        # boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        # boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)


        scores = np.array(scores)
        class_ids = np.array(class_ids)
        mask_predictions = np.array(mask_predictions)


        # print(boxes.shape, scores.shape, class_ids.shape, mask_predictions.shape)
        return boxes, scores, class_ids, mask_predictions

    # def process_mask_output(self, mask_predictions, mask_output):
    #     if mask_predictions.shape[0] == 0:
    #         return []
    #
    #     mask_output = np.squeeze(mask_output)
    #
    #     num_mask, mask_height, mask_width = mask_output.shape
    #     masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
    #     masks = masks.reshape((-1, mask_height, mask_width))
    #
    #     # Avoid redundant calculations
    #     scale_boxes = self.rescale_boxes(self.boxes,
    #                                      (self.img_height, self.img_width),
    #                                      (mask_height, mask_width))
    #
    #     mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
    #     blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
    #
    #     for i, (box, scale_box) in enumerate(zip(self.boxes, scale_boxes)):
    #         scale_y1, scale_y2, scale_x1, scale_x2 = map(int, map(math.floor, scale_box))
    #         y1, y2, x1, x2 = map(int, map(math.floor, box))
    #
    #         scale_crop_mask = masks[i, scale_y1:scale_y2, scale_x1:scale_x2]
    #         crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    #         crop_mask = cv2.blur(crop_mask, blur_size)
    #         crop_mask = (crop_mask > 0.5).astype(np.uint8)
    #         mask_maps[i, y1:y2, x1:x2] = crop_mask
    #
    #     return mask_maps

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)
        # print(mask_output.shape)  # (32, 160, 160)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))


        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps


    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        if 'yolo26' in self.onnx_path:

            self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output_26(outputs[0])

        else:
            self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])

        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def __call__(self, image):
        return self.segment_objects(image)
    
    
    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))