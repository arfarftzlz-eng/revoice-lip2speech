#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

import numpy as np

warnings.filterwarnings("ignore")


class _MediaPipeDetector:
    """Wrapper for MediaPipe FaceLandmarker with proper cleanup."""

    # MediaPipe FaceMesh produces 478 landmarks; we map a subset to the
    # 68-point scheme expected by the downstream mouth-cropping pipeline.
    # Mapping from: https://github.com/google/mediapipe/issues/1615
    _MP_TO_68 = [
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288,
        323, 454, 389,  # jaw 0-16
        70, 63, 105, 66, 107,  # left eyebrow 17-21
        336, 296, 334, 293, 301,  # right eyebrow 22-26
        168, 197, 5, 4, 75,  # nose bridge 27-30  (approx)
        97, 2, 326, 305,  # nose bottom 31-34  (approx)
        33, 160, 158, 133, 153, 144,  # left eye 36-41
        362, 385, 387, 263, 373, 380,  # right eye 42-47
        61, 39, 37, 0, 267, 269, 291,  # outer lip top 48-54
        321, 314, 17, 84, 91,  # outer lip bottom 55-59
        78, 82, 13, 312, 308,  # inner lip top 60-64
        317, 14, 87,  # inner lip bottom 65-67
    ]

    def __init__(self):
        import os
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe not found. Install it with:\n"
                "  pip install mediapipe"
            )

        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions

        self._mp = mp

        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Download it with:\n"
                "  wget -O preprocessing/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        with open(model_path, "rb") as model_file:
            model_asset_buffer = model_file.read()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_asset_buffer),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            # Frames from torchvision.io.read_video are already RGB
            h, w = frame.shape[:2]
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
            results = self._landmarker.detect(mp_image)
            if not results.face_landmarks:
                landmarks.append(None)
            else:
                face_lm = results.face_landmarks[0]
                pts = np.array(
                    [(face_lm[i].x * w, face_lm[i].y * h)
                     for i in self._MP_TO_68],
                    dtype=np.float32,
                )
                landmarks.append(pts)
        return landmarks

    def close(self):
        """Explicitly close the MediaPipe landmarker to avoid shutdown errors."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class _RetinaFaceDetector:
    """Wrapper for RetinaFace + FAN landmark detection (ibug packages).

    This provides higher-accuracy 68-point landmarks compared to MediaPipe,
    matching the detector used in the auto_avsr repository.
    Requires GPU and the ibug packages (see README for full instructions).
    """

    def __init__(self, device="cuda:0"):
        try:
            from ibug.face_detection import RetinaFacePredictor
            from ibug.face_alignment import FANPredictor
        except ImportError:
            raise ImportError(
                "ibug face_detection/face_alignment not found. "
                "These packages require cloning and installing in editable mode. "
                "See the README 'RetinaFace+FAN' section for full instructions."
            )

        self._face_detector = RetinaFacePredictor(
            threshold=0.8,
            device=device,
            model=RetinaFacePredictor.get_model("resnet50"),
        )
        self._landmark_detector = FANPredictor(
            device=device,
            model=FANPredictor.get_model("2dfan2"),
        )

    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            detected_faces = self._face_detector(frame, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                # Select the largest face by bounding-box area
                face_areas = (detected_faces[:, 2] - detected_faces[:, 0]) * (
                    detected_faces[:, 3] - detected_faces[:, 1]
                )
                largest_idx = np.argmax(face_areas)
                face = detected_faces[largest_idx : largest_idx + 1]
                pts, _ = self._landmark_detector(frame, face, rgb=True)
                landmarks.append(pts[0].astype(np.float32))
        return landmarks

    def close(self):
        """No-op — ibug predictors don't require explicit cleanup."""
        pass


class LandmarksDetector:
    def __init__(self, detector="mediapipe", device="cuda:0", **kwargs):
        if detector == "mediapipe":
            self._detect = _MediaPipeDetector()
        elif detector == "retinaface":
            self._detect = _RetinaFaceDetector(device=device)
        else:
            raise ValueError(
                f"Unknown detector '{detector}'. Choose 'mediapipe' or 'retinaface'."
            )

    def __call__(self, video_frames):
        return self._detect(video_frames)

    def close(self):
        """Release resources. Call this when done to avoid shutdown errors."""
        if hasattr(self._detect, "close"):
            self._detect.close()
