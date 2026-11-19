
class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def calculate_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def update(self, detections):
        # Age existing tracks
        for track in self.tracks.values():
            track["age"] += 1

        matches = []
        candidates = []

        # Compare detections with existing tracks
        for detection_index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():

                if track["class_id"] != detection["class_id"]:
                    continue

                iou = self.calculate_iou(
                    detection["bbox"],
                    track["bbox"]
                )

                if iou >= self.iou_threshold:
                    candidates.append(
                        (iou, detection_index, track_id)
                    )

        # Highest IoU matches first
        candidates.sort(reverse=True)

        used_detections = set()
        used_tracks = set()

        for iou, detection_index, track_id in candidates:
            if detection_index in used_detections:
                continue

            if track_id in used_tracks:
                continue

            matches.append((detection_index, track_id))
            used_detections.add(detection_index)
            used_tracks.add(track_id)

        # Update matched tracks
        tracked_detections = [None] * len(detections)

        for detection_index, track_id in matches:
            detection = dict(detections[detection_index])

            self.tracks[track_id]["bbox"] = detection["bbox"]
            self.tracks[track_id]["age"] = 0
            self.tracks[track_id]["hits"] += 1

            detection["track_id"] = track_id
            tracked_detections[detection_index] = detection

        # Create tracks for unmatched detections
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "class_id": detection["class_id"],
                "age": 0,
                "hits": 1
            }

            tracked = dict(detection)
            tracked["track_id"] = track_id

            tracked_detections[index] = tracked

        # Remove stale tracks
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["age"] > self.max_age
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return tracked_detections

    def get_active_count(self):
        return sum(
            1 for track in self.tracks.values()
            if track["age"] == 0
        )

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
