import contextlib
import io


class YoloBoundingBox:
    def __init__(self, image_processor, load_params):
        self.image_processor = image_processor
        self.load_params = load_params
        self._yolo_model = None
        self._yolo_segmentation_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = self.load_params.get_detection_model()
        return self._yolo_model

    @property
    def yolo_segmentation_model(self):
        if self._yolo_segmentation_model is None:
            self._yolo_segmentation_model = self.load_params.get_segmentation_model()
        return self._yolo_segmentation_model

    def get_confidence_threshold(self):
        return self.load_params.get_confidence_threshold()

    def get_tags_and_boxes(self, confidence_threshold=None):
        if confidence_threshold is None:
            confidence_threshold = self.get_confidence_threshold()

        self.target_label = self.get_target_label()
        self.image = self.image_processor.get_rgb_cv_image()
        if self.image is None:
            return []

        detection_results = self._yolo_msg_filter(self.image)

        detected_objects = []
        for result in detection_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                confidence = float(box.conf)

                if confidence < confidence_threshold:
                    continue

                if self.target_label and class_name != self.target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append(
                    {
                        "label": class_name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    }
                )

        return detected_objects

    def get_segmentation_data(self, confidence_threshold=None):
        """
        Returns segmentation masks for detected objects.
        """
        if confidence_threshold is None:
            confidence_threshold = self.get_confidence_threshold()

        self.image = self.image_processor.get_rgb_cv_image()
        if self.image is None:
            print("Error: No image received from image_processor")
            return []

        segmentation_results = self._yolo_segmentation_filter(self.image)

        if not segmentation_results or segmentation_results[0].masks is None:
            # print("Warning: No segmentation masks detected.")
            return []  # Return empty list if no masks found

        segmentation_objects = []
        for result in segmentation_results:
            if result.masks is None or result.boxes is None:
                continue

            # Convert masks to usable format
            masks_np = result.masks.data.cpu().numpy()  # Convert to NumPy array

            for i, (box, cls, conf) in enumerate(
                zip(
                    result.boxes.xyxy.cpu().numpy(),  # Bounding box in pixel coordinates
                    result.boxes.cls.cpu().numpy(),  # Class IDs
                    result.boxes.conf.cpu().numpy(),  # Confidence scores
                )
            ):
                if float(conf) < confidence_threshold:
                    continue

                class_id = int(cls)
                class_name = self.yolo_segmentation_model.names[class_id]

                segmentation_objects.append(
                    {
                        "label": class_name,
                        "confidence": float(conf),
                        "box": tuple(map(int, box)),
                        "mask": masks_np[i],
                    }
                )

        # print("Segmentation Data:", segmentation_objects)  # Debugging Output
        return segmentation_objects

    def _yolo_msg_filter(self, img):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.yolo_model(img, verbose=False)
        return results

    def _yolo_segmentation_filter(self, img):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.yolo_segmentation_model(img, verbose=False)
        return results

    def get_target_label(self):
        target_label = self.image_processor.get_yolo_target_label()
        if target_label in [None, "None"]:
            target_label = None
        return target_label
