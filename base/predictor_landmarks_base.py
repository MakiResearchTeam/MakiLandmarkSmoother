import numpy as np

from .predictor_face_base import FaceBoxPredictorBase



### The wrapper on landmarks predictors

class LandmarksPredictorBase:

    def __init__(self):
        pass

    def predict_landmarks(self, face_predictor: FaceBoxPredictorBase, images: list) -> list:
        pass

    def predict_landmarks_in_certain_space(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        pass
