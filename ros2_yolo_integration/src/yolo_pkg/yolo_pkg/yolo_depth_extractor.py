import numpy as np


class YoloDepthExtractor:
    def __init__(self, yolo_boundingbox, image_processor, ros_communication):
        self.yolo_boundingbox = yolo_boundingbox
        self.ros_communicator = ros_communication
        self.image_processor = image_processor

    def get_yolo_object_depth(self, radius_increment=2, max_iterations=10):
        """
        Calculates the depth for each detected object.
        If the center pixel depth is invalid (<=0 or NaN), it iteratively expands
        a search window outwards from the center, staying within the bounding box,
        until a valid depth (mean of valid pixels in the window) is found or
        the window reaches the box limits or max iterations.

        Args:
            radius_increment (int): The amount to increase the search radius in each
                                    iteration (e.g., 2 means radius grows 2, 4, 6...).
            max_iterations (int): Maximum number of expansion steps to prevent infinite loops.

        Returns:
            list: A list of dictionaries, each containing 'label', 'box', and 'depth'.
                  Depth is in meters (float) or the original invalid value if no valid
                  depth is found.
        """
        depth_cv_image = self.image_processor.get_depth_cv_image()
        if depth_cv_image is None or not isinstance(depth_cv_image, np.ndarray):
            print("Depth image is invalid.")
            return []

        detected_objects = self.yolo_boundingbox.get_tags_and_boxes()
        if not detected_objects:
            return []

        objects_with_depth = []
        img_h, img_w = depth_cv_image.shape[:2]

        for obj in detected_objects:
            label = obj["label"]
            x1, y1, x2, y2 = map(int, obj["box"])  # Ensure box coords are int

            # Clamp box coordinates to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

            if x1 >= x2 or y1 >= y2:
                print(f"Invalid box dimensions after clamping for {label}. Skipping.")
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # --- Initial depth check at the center ---
            depth_value = depth_cv_image[center_y, center_x]
            is_invalid = depth_value <= 0 or np.isnan(depth_value)

            # --- Iterative Expansion Search (if center is invalid) ---
            if is_invalid:
                found_valid_depth = False
                current_radius = 0
                for iteration in range(max_iterations):
                    current_radius += radius_increment

                    # Define search window boundaries, clamped by bounding box
                    min_x = max(x1, center_x - current_radius)
                    max_x = min(x2, center_x + current_radius)
                    min_y = max(y1, center_y - current_radius)
                    max_y = min(y2, center_y + current_radius)

                    # Check if window covers the entire bounding box - stop if it does
                    if (
                        min_x == x1
                        and max_x == x2
                        and min_y == y1
                        and max_y == y2
                        and iteration > 0
                    ):
                        # print(f"Window reached bbox limits for {label} at radius {current_radius}. Stopping search.") # Optional debug
                        break  # No point expanding further

                    # Extract the window (ensure valid slice indices)
                    if (
                        min_y > max_y or min_x > max_x
                    ):  # Should not happen with clamping, but safety check
                        continue
                    depth_window = depth_cv_image[min_y : max_y + 1, min_x : max_x + 1]

                    # Find valid depths within the window
                    valid_depths = depth_window[
                        (depth_window > 0) & (~np.isnan(depth_window))
                    ]

                    if valid_depths.size > 0:
                        # Calculate the mean of valid depths
                        depth_value = np.mean(valid_depths)
                        found_valid_depth = True
                        # print(f"Found valid depth for {label} at radius {current_radius}: {depth_value:.3f}m") # Optional debug
                        break  # Exit loop once valid depth is found

                # if not found_valid_depth: # Optional debug
                # print(f"No valid depth found for {label} after {max_iterations} iterations.")

            # --- Append Result ---
            # depth_value will be the original center value (valid or invalid)
            # or the mean from the first successful window expansion.
            # If expansion failed, it remains the original invalid value.
            objects_with_depth.append(
                {
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "depth": (
                        float(depth_value) if not np.isnan(depth_value) else np.nan
                    ),  # Handle potential NaN persistence
                }
            )

        return objects_with_depth

    # ... rest of the class (get_depth_camera_center_value) ...
    # Note: get_depth_camera_center_value still uses a fixed window search
    def get_depth_camera_center_value(self):
        """
        Returns the depth value at the center point of the depth camera,
        along with the center coordinates. Uses a fixed window search if center is invalid.

        Returns:
            dict: Contains 'center' (x,y) coordinates and 'depth' value.
                  Returns None if depth image is invalid or no valid depth found near center.
        """
        depth_cv_image = self.image_processor.get_depth_cv_image()
        is_invalid_depth_image = depth_cv_image is None or not isinstance(
            depth_cv_image, np.ndarray
        )
        if is_invalid_depth_image:
            print("Depth image is invalid.")
            return None

        height, width = depth_cv_image.shape[:2]
        center_x = width // 2
        center_y = height // 2

        center_depth = depth_cv_image[center_y, center_x]

        if center_depth <= 0 or np.isnan(center_depth):
            window_size = 5  # Fixed window size (e.g., 11x11) for camera center
            min_r, max_r = max(0, center_y - window_size), min(
                height, center_y + window_size + 1
            )
            min_c, max_c = max(0, center_x - window_size), min(
                width, center_x + window_size + 1
            )
            window = depth_cv_image[min_r:max_r, min_c:max_c]

            non_zero_values = window[(window > 0) & (~np.isnan(window))]
            if non_zero_values.size > 0:
                center_depth = np.mean(non_zero_values)
            else:
                print("No valid depth value found near the camera center point.")
                return None

        # Ensure float return, handle potential NaN
        final_depth = float(center_depth) if not np.isnan(center_depth) else np.nan
        if np.isnan(final_depth):
            print("Final center depth is NaN.")
            return None  # Return None if depth remains NaN

        return {"center": (center_x, center_y), "depth": final_depth}
