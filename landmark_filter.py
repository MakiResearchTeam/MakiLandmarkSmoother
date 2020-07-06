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


from .base import LandmarksPredictorBase
import numpy as np

from tqdm import tqdm


### Landmarks smoother with global and local filter
### LSWGL


class LSWGL:

    def __init__(self, noc=0.06, t_value=None, lambda_value=None
                 ):
        # variables for smoothe stuff
        self._gsv = None
        self._lsv = None
        self._noc = noc
        self._w = None

        if t_value is None:
            self._t = 1.0
        else:
            self._t = t_value

        if lambda_value is None:
            self._lamda = 2.0
        else:
            self._lamda = lambda_value

        self._is_landmarks_variable_restored = False

        self._fb_landmarks = None
        self._dfb_landmarks = None

        self.number_of_restore = 0

    def smoothe_landmarks(self, landmarks: list) -> list:
        landmarks_list = []

        iterator = tqdm(range(len(landmarks)))

        for i in iterator:
            if i == 0:
                # array with shape [68, 2]
                self._fb_landmarks = landmarks[i]
                self._recalculate_variables_landmarks()
                self._is_landmarks_variable_restored = False

                landmarks_list.append(self._fb_landmarks)
                continue

            # array with shape [68, 2]
            dfb_landmarks = landmarks[i]

            if len(self._dfb_landmarks) == 0:
                continue
            else:
                self._dfb_landmarks = dfb_landmarks[0]

            # Calculate GSV and compare movement of face according to landmarks
            self._calculate_GSV()
            need_to_restore_landmarks = self._gsv > self._noc
            if type(need_to_restore_landmarks) == np.ndarray:
                need_to_restore_landmarks = need_to_restore_landmarks.max()

            if need_to_restore_landmarks:
                print(f'Landmarks are restored at {i} iteration')

            if need_to_restore_landmarks:
                self._recalculate_variables_landmarks()
            else:
                if not self._is_landmarks_variable_restored:
                    self._recalculate_landmark_variables()

                self._calc_new_landmarks()

            landmarks_list.append(self._fb_landmarks)

        iterator.close()
        self.__reset_variables()
        return landmarks_list

    def _calculate_GSV(self):
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks
        # Choose 7 landmarks
        seven_landmarks_old = np.stack([
            old[30],  # top of the nose

            old[45],  # right eye, right corner
            old[42],  # right eye, left corner
            old[39],  # left eye, right corner
            old[36],  # left eye, left corner

            old[48],  # mouth, left corner
            old[54],  # mouth, right corner
        ], axis=0)
        seven_landmarks_new = np.stack([
            new[30],  # top of the nose

            new[45],  # right eye, right corner
            new[42],  # right eye, left corner
            new[39],  # left eye, right corner
            new[36],  # left eye, left corner

            new[48],  # mouth, left corner
            new[54],  # mouth, right corner
        ], axis=0)

        # Calclate distance between centre of eyes
        mid_left_eye = (old[39] + old[36]) / 2.0
        mid_right_eye = (old[45] + old[42]) / 2.0
        eye_dist = np.sum(np.square(mid_left_eye - mid_right_eye))

        self._gsv = np.sqrt(
            np.sum(np.square(seven_landmarks_new - seven_landmarks_old)) / eye_dist
        )

    def _calculate_lsv(self):
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks
        # Calclate distance between centre of eyes
        mid_left_eye = (old[39] + old[36]) / 2.0
        mid_right_eye = (old[45] + old[42]) / 2.0
        eye_dist = np.sum(np.square(mid_left_eye - mid_right_eye))

        cur_distance = np.sqrt(np.sum(np.square(old - new), axis=1) / eye_dist)

        self._lsv = np.abs(cur_distance - self._gsv)

    def _calculate_W(self):
        self._calculate_lsv()
        if self._gsv > self._noc:
            self._is_landmarks_variable_restored = False
            self._w = np.zeros(68).astype(np.float32)
        else:
            self._w = self._lamda * self._lsv + self._t * self._gsv

    def _calc_new_landmarks(self):
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks

        self._calculate_W()
        self._fb_landmarks = np.expand_dims(self._w, axis=1) * old + (1 - np.expand_dims(self._w, axis=1)) * new

    def _recalculate_variables_landmarks(self):
        if self._dfb_landmarks is not None:
            self._fb_landmarks = self._dfb_landmarks
            self._dfb_landmarks = None

        self.number_of_restore += 1

    def _recalculate_landmark_variables(self):
        self._calculate_lsv()
        self._is_landmarks_variable_restored = True

    def __reset_variables(self):
        self._lsv = None
        self._gsv = None
        self._w = None

