import numpy as np
from dlib import rectangle
import dlib

from ..base import FaceBoxPredictorBase, LandmarksPredictorBase



### The wrapper on landmarks predictors

class LandmarksPredictorDlib(LandmarksPredictorBase):

    def __init__(self, path_to_shape: str):
        self._landmark_predictor = dlib.shape_predictor(path_to_shape)

    def predict_landmarks(self, face_predictor: FaceBoxPredictorBase, images: list) -> list:
        pass

    def predict_landmarks_in_certain_space(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        face = rectangle(x1, y1, x2, y2)
        grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self._landmark_predictor(grayFrame, face)
        landmarks_np = face_utils.shape_to_np(landmarks)

        return landmarks_np

