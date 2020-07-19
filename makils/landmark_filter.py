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


import numpy as np

from tqdm import tqdm


class LandmarkSmoother:

    def __init__(self, noc=0.06, t_value=None, lambda_value=None,
                 index_of_top_nose=30,
                 index_of_right_eye_right_corner=45,
                 index_of_right_eye_left_corner=42,
                 index_of_left_eye_right_corner=39,
                 index_of_left_eye_left_corner=36,
                 index_of_mouth_left_corner=48,
                 index_of_mouth_right_corner=54,
                 number_of_landmarks=68
                 ):
        """
        Create Landmarks smoother

        Parameters
        ----------
        noc : float
            Noc (or Number Of Chaos) responsible for a measure of shift landmarks,
            between previous and current landmarks, usually 0.06 +- 0.1 number is good
        t_value : float
            Scale of the GSV (Global shaking value) responsible for movement of the main points,
            in our case is 4 points for two eyes corners, 1 point for tip of nose and two for mouth corners
        lambda_value : float
            Scale of the LSV (Local shaking value) responsible for movement of all landmarks
        index_of_top_nose : int
            Index of the landmarks which is responsible for top nose position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_right_eye_right_corner : int
            Index of the landmarks which is responsible for right corner of the right eye position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_right_eye_left_corner : int
            Index of the landmarks which is responsible for left corner of the right eye position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_left_eye_right_corner : int
            Index of the landmarks which is responsible for right corner of the left eye position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_left_eye_left_corner : int
            Index of the landmarks which is responsible for left corner of the left eye position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_mouth_left_corner : int
            Index of the landmarks which is responsible for left corner of the mouth position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        index_of_mouth_right_corner : int
            Index of the landmarks which is responsible for right corner of the mouth position,
            this index us used for calculate LSV/GSV (e. i. for landmarks shift),
            By default set index of the Dlib library
        number_of_landmarks : int
            Maximum number of landmarks,
            By default equal to 68, which is usually use
        """
        self.__reset_variables()
        self._noc = noc

        self._index_of_top_nose = index_of_top_nose
        self._index_of_right_eye_right_corner = index_of_right_eye_right_corner
        self._index_of_right_eye_left_corner = index_of_right_eye_left_corner
        self._index_of_left_eye_right_corner = index_of_left_eye_right_corner
        self._index_of_left_eye_left_corner = index_of_left_eye_left_corner
        self._index_of_mouth_left_corner = index_of_mouth_left_corner
        self._index_of_mouth_right_corner = index_of_mouth_right_corner

        self._number_of_landmakrs = number_of_landmarks

        if t_value is None:
            self._t = 1.0
        else:
            self._t = t_value

        if lambda_value is None:
            self._lamda = 2.0
        else:
            self._lamda = lambda_value

        self._is_landmarks_variable_restored = False
        # fb - means main landmarks (which we "smoothe", i. e. we accumulating this value)
        # dfb - current landmarks at the image (which is not "smoothe")
        self._fb_landmarks = None
        self._dfb_landmarks = None

    def smoothe_landmarks(self, landmarks: list) -> list:
        """
        Apply smoothe algorithm on `landmarks`

        Parameters
        ----------
        landmarks : list
            List of landmarks, which are should go in the order of the video

        Returns
        -------
        list
            Smoother landmarks in same order as input `landmarks`
        """
        landmarks_list = []

        iterator = tqdm(range(len(landmarks)))

        for i in iterator:
            if i == 0:
                # array with shape [number_of_landmarks, 2]
                self._fb_landmarks = landmarks[i]
                self._recalculate_variables_landmarks()
                self._is_landmarks_variable_restored = False

                landmarks_list.append(self._fb_landmarks)
                continue

            # array with shape [number_of_landmarks, 2]
            self._dfb_landmarks = landmarks[i]

            # Calculate GSV and compare movement of face according to landmarks
            self._calculate_GSV()
            need_to_restore_landmarks = self._gsv > self._noc
            if type(need_to_restore_landmarks) == np.ndarray:
                need_to_restore_landmarks = need_to_restore_landmarks.max()

            if need_to_restore_landmarks:
                print(f'Landmarks are restored at {i} iteration')
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
        """
        Calculate GSV or Global Shaking Value, which is responsible for movement of the main points,
        in our case is 4 points for two eyes corners, 1 point for tip of nose and two for mouth corners

        """
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks
        # Choose 7 landmarks
        seven_landmarks_old = np.stack([
            old[self._index_of_top_nose],  # top of the nose

            old[self._index_of_right_eye_right_corner],  # right eye, right corner
            old[self._index_of_right_eye_left_corner],  # right eye, left corner
            old[self._index_of_left_eye_right_corner],  # left eye, right corner
            old[self._index_of_left_eye_left_corner],  # left eye, left corner

            old[self._index_of_mouth_left_corner],  # mouth, left corner
            old[self._index_of_mouth_right_corner],  # mouth, right corner
        ], axis=0)

        seven_landmarks_new = np.stack([
            new[self._index_of_top_nose],  # top of the nose

            new[self._index_of_right_eye_right_corner],  # right eye, right corner
            new[self._index_of_right_eye_left_corner],  # right eye, left corner
            new[self._index_of_left_eye_right_corner],  # left eye, right corner
            new[self._index_of_left_eye_left_corner],  # left eye, left corner

            new[self._index_of_mouth_left_corner],  # mouth, left corner
            new[self._index_of_mouth_right_corner],  # mouth, right corner
        ], axis=0)

        # Calculate distance between centre of eyes
        mid_left_eye = (old[39] + old[36]) / 2.0
        mid_right_eye = (old[45] + old[42]) / 2.0
        eye_dist = np.sum(np.square(mid_left_eye - mid_right_eye))

        self._gsv = np.sqrt(
            np.sum(np.square(seven_landmarks_new - seven_landmarks_old)) / eye_dist
        )

    def _calculate_lsv(self):
        """
        Calculate GSV or Global Shaking Value, which is responsible for movement of all landmarks

        """
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks
        # Calclate distance between centre of eyes
        mid_left_eye = (old[self._index_of_left_eye_right_corner] + old[self._index_of_left_eye_left_corner]) / 2.0
        mid_right_eye = (old[self._index_of_right_eye_right_corner] + old[self._index_of_right_eye_left_corner]) / 2.0
        eye_dist = np.sum(np.square(mid_left_eye - mid_right_eye))

        cur_distance = np.sqrt(np.sum(np.square(old - new), axis=1) / eye_dist)

        self._lsv = np.abs(cur_distance - self._gsv)

    def _calculate_W(self):
        """
        In our case w - is some sort of alpha (or decay) of the Moving average

        """
        self._calculate_lsv()
        if self._gsv > self._noc:
            # Drop our smoother value, and set value to current landmark
            self._is_landmarks_variable_restored = False
            self._w = np.zeros(self._number_of_landmakrs).astype(np.float32)
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

    def _recalculate_landmark_variables(self):
        self._calculate_lsv()
        self._is_landmarks_variable_restored = True

    def __reset_variables(self):
        """
        Set variables, to default value

        """
        self._lsv = None
        self._gsv = None
        self._w = None

