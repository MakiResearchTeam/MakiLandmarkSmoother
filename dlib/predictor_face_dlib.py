### The wrapper on face box predictors
import numpy as np
import dlib
import cv2

from ..base import FaceBoxPredictorBase


class FaceBoxPredictorDlib(FaceBoxPredictorBase):


    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def predict_face_box(self, image: list) -> list:
        pass

    def predict_face_box_single(self, image: np.ndarray) -> list:
        grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector(grayFrame)
        if len(faces) == 0:
            return False

        face = faces[0]
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        return [x1, y1, x2, y2]

