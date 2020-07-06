### The wrapper on face box predictors
import numpy as np


class FaceBoxPredictorBase:


    def __init__(self):
        pass

    def predict_face_box(self, image: list) -> list:
        pass

    def predict_face_box_single(self, image: np.ndarray) -> np.ndarray:
        pass

