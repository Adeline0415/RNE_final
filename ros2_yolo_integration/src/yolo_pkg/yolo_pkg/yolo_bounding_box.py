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


# # 簡化的 yolo_bounding_box.py - 只檢測真正的pikachu

# import contextlib
# import io

# class YoloBoundingBox:
#     def __init__(self, image_processor, load_params):
#         self.image_processor = image_processor
#         self.load_params = load_params
#         self._yolo_model = None
#         self._yolo_segmentation_model = None
        
#         # === 簡化配置 - 只有pikachu是目標 ===
#         self.target_pikachu = "pikachu"  # 唯一真正的目標
#         self.fake_pikachu = ["kirito_pikachu", "gengar_pikachu", "fluorescent_pikachu"]  # 干擾物
#         self.other_objects = ["yoda", "beer", "gundam", "furniture"]
        
#         # Hell模式開關
#         self.hell_mode = False

#     def set_hell_mode(self, enabled=True):
#         """設置Hell模式"""
#         self.hell_mode = enabled
#         if enabled:
#             print("🔥 Hell模式已啟用 - 專注於真正的pikachu")
            
#     def get_tags_and_boxes(self, confidence_threshold=None):
#         """獲取檢測框 - 只關注真正的pikachu"""
#         if confidence_threshold is None:
#             confidence_threshold = self.get_confidence_threshold()

#         self.target_label = self.get_target_label()
#         self.image = self.image_processor.get_rgb_cv_image()
#         if self.image is None:
#             return []

#         detection_results = self._yolo_msg_filter(self.image)
#         detected_objects = []
        
#         for result in detection_results:
#             for box in result.boxes:
#                 class_id = int(box.cls[0])
#                 class_name = self.yolo_model.names[class_id]
#                 confidence = float(box.conf)
                
#                 # confidence檢查
#                 if confidence < confidence_threshold:
#                     continue
                
#                 # 判斷是否應該包含這個物體
#                 should_include = False
#                 object_type = "unknown"
                
#                 if class_name == self.target_pikachu:
#                     # 真正的目標pikachu
#                     should_include = True
#                     object_type = "target"
#                 elif class_name in self.fake_pikachu:
#                     # 假皮卡丘 - Hell模式下也要檢測（用於調試和避開）
#                     if self.hell_mode:
#                         should_include = True
#                         object_type = "fake_pikachu"
#                 elif class_name == "furniture":
#                     # 家具 - Hell模式下檢測（用於導航參考）
#                     if self.hell_mode:
#                         should_include = True
#                         object_type = "furniture"
#                 elif class_name in ["yoda", "beer", "gundam"]:
#                     # 其他干擾物 - Hell模式下也檢測
#                     if self.hell_mode:
#                         should_include = True
#                         object_type = "distractor"
#                 else:
#                     # 正常模式下檢查target_label
#                     if not self.target_label or class_name == self.target_label:
#                         should_include = True
#                         object_type = "other"
                
#                 if should_include:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
#                     detected_objects.append({
#                         "label": class_name,
#                         "confidence": confidence,
#                         "box": (x1, y1, x2, y2),
#                         "object_type": object_type,
#                         "is_target_pikachu": class_name == self.target_pikachu,
#                         "is_fake_pikachu": class_name in self.fake_pikachu
#                     })
        
#         # 排序：真pikachu優先，然後按confidence
#         detected_objects.sort(key=lambda x: (x["is_target_pikachu"], x["confidence"]), reverse=True)
            
#         return detected_objects

#     def get_target_pikachu_objects(self):
#         """獲取真正的pikachu目標"""
#         all_objects = self.get_tags_and_boxes()
#         return [obj for obj in all_objects if obj['label'] == self.target_pikachu]

#     def get_fake_pikachu_objects(self):
#         """獲取假pikachu（干擾物）"""
#         all_objects = self.get_tags_and_boxes()
#         return [obj for obj in all_objects if obj['label'] in self.fake_pikachu]

#     def get_best_target(self):
#         """獲取最佳目標 - 只考慮真正的pikachu"""
#         pikachu_objects = self.get_target_pikachu_objects()
        
#         if not pikachu_objects:
#             return None
        
#         # 選擇confidence最高的真pikachu
#         return max(pikachu_objects, key=lambda x: x['confidence'])

#     def get_furniture_objects(self):
#         """獲取家具檢測結果"""
#         all_objects = self.get_tags_and_boxes()
#         return [obj for obj in all_objects if obj['label'] == 'furniture']

#     def get_detection_summary(self):
#         """獲取檢測摘要"""
#         all_objects = self.get_tags_and_boxes()
#         target_pikachu = self.get_target_pikachu_objects()
#         fake_pikachu = self.get_fake_pikachu_objects()
        
#         summary = {
#             'total_objects': len(all_objects),
#             'target_pikachu_count': len(target_pikachu),  # 真正的pikachu數量
#             'fake_pikachu_count': len(fake_pikachu),      # 假pikachu數量
#             'furniture_count': len(self.get_furniture_objects()),
#             'best_target': self.get_best_target(),
#             'fake_pikachu_types': list(set([obj['label'] for obj in fake_pikachu]))
#         }
        
#         return summary

#     # === 保留原有核心方法 ===
#     @property
#     def yolo_model(self):
#         if self._yolo_model is None:
#             self._yolo_model = self.load_params.get_detection_model()
#         return self._yolo_model

#     @property
#     def yolo_segmentation_model(self):
#         if self._yolo_segmentation_model is None:
#             self._yolo_segmentation_model = self.load_params.get_segmentation_model()
#         return self._yolo_segmentation_model

#     def get_confidence_threshold(self):
#         return self.load_params.get_confidence_threshold()

#     def _yolo_msg_filter(self, img):
#         with contextlib.redirect_stdout(io.StringIO()):
#             results = self.yolo_model(img, verbose=False)
#         return results

#     def _yolo_segmentation_filter(self, img):
#         with contextlib.redirect_stdout(io.StringIO()):
#             results = self.yolo_segmentation_model(img, verbose=False)
#         return results

#     def get_target_label(self):
#         target_label = self.image_processor.get_yolo_target_label()
#         if target_label in [None, "None"]:
#             target_label = None
#         return target_label

#     def get_segmentation_data(self, confidence_threshold=None):
#         """語義分割數據獲取"""
#         if confidence_threshold is None:
#             confidence_threshold = self.get_confidence_threshold()

#         self.image = self.image_processor.get_rgb_cv_image()
#         if self.image is None:
#             return []

#         segmentation_results = self._yolo_segmentation_filter(self.image)

#         if not segmentation_results or segmentation_results[0].masks is None:
#             return []

#         segmentation_objects = []
#         for result in segmentation_results:
#             if result.masks is None or result.boxes is None:
#                 continue

#             masks_np = result.masks.data.cpu().numpy()

#             for i, (box, cls, conf) in enumerate(
#                 zip(
#                     result.boxes.xyxy.cpu().numpy(),
#                     result.boxes.cls.cpu().numpy(),
#                     result.boxes.conf.cpu().numpy(),
#                 )
#             ):
#                 if float(conf) < confidence_threshold:
#                     continue

#                 class_id = int(cls)
#                 class_name = self.yolo_segmentation_model.names[class_id]

#                 segmentation_objects.append({
#                     "label": class_name,
#                     "confidence": float(conf),
#                     "box": tuple(map(int, box)),
#                     "mask": masks_np[i],
#                 })

#         return segmentation_objects