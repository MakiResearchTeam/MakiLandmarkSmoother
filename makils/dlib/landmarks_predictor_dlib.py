# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiLandmarkSmoother.
#
# MakiLandmarkSmoother is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiLandmarkSmoother is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import cv2
from tqdm import tqdm

from .. import dlib
from imutils import face_utils

from ..base import LandmarksPredictorBase


class LandmarksPredictorDlib(LandmarksPredictorBase):

    def __init__(self, path_to_shape: str):
        """
        Create wrapper of the landmark predictor from Dlib library

        Parameters
        ----------
        path_to_shape : str
            Path to shape file which is used for initialize landmark predictor,
            for more information visit original Dlib site
        """
        self._detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(path_to_shape)

    def predict_landmarks(self, images: list) -> list:
        """
        Predict landmarks at `images`

        Parameters
        ----------
        images : list
            List of images

        Returns
        -------
        list
            List of np.ndarrays which is landmarks to certain image from `images`,
            return in the same order as input `images`
        """
        iterator = tqdm(range(len(images)))
        landmarks_np_list = []

        for i in iterator:
            image = images[i]
            # Find facebox
            grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detector(grayFrame)
            if len(faces) == 0:
                print(f'On the image at {i} index was found 0 faces.')
                continue
            # Take only one face and predict landmarks
            face = faces[0]
            landmarks = self._landmark_predictor(grayFrame, face)
            landmarks_np = face_utils.shape_to_np(landmarks)
            landmarks_np_list.append(landmarks_np)

        return landmarks_np_list

