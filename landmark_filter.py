# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiLandmarkSmoother.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.


from .base import LandmarksPredictorBase, FaceBoxPredictorBase
import numpy as np

from tqdm import tqdm

### Landmarks smoother with global and local filter
### LSWGL


class LSWGL:

    def __init__(self, predict_face: FaceBoxPredictorBase, predict_landmarks: LandmarksPredictorBase, 
                 image_w: int, image_h: int, 
                 noc=0.06, shift_x=20, shift_y=20, 
                 area_diff=0.2, t_value=None, lambda_value=None,
                 box_update_using_decay=False, 
                 decay_box=0.9
    ):
        self._predict_face = predict_face
        self._predict_landmarks = predict_landmarks

        self._image_w = image_w
        self._image_h = image_h

        # variables for smoothe stuff
        self._yo = None
        self._gsv = None
        self._lsv = None 
        self._noc = noc
        self._w = None

        if t_value is None:
            #self._t = - (1 / noc)
            self._t = 1.0
            self._t_update_in_opt = False
        else:
            self._t = t_value
            self._t_update_in_opt = True

        if lambda_value is None:
            self._lamda = 1.0
            self._lamda_update_in_opt = False
        else:
            self._lamda = lambda_value
            self._lamda_update_in_opt = True

        self._is_landmarks_variable_restored = False

        self._fb = None
        self._fb_landmarks = None
        self._dfb = None
        self._dfb_landmarks = None

        self._shift_x = shift_x
        self._shift_y = shift_y
        self._area_diff = area_diff

        self._box_update_using_decay = box_update_using_decay
        self._decay_box = decay_box

        # counter for images
        self._counter = 0
        self.number_of_restore = 0

    def smoothe_faceboxes(self, images:list) -> list:
        return self._smoothe_optimization(images, 
            return_smoothe_faceboxes=True, 
            return_smoothe_landmarks=False
        )

    def smoothe_landmarks(self, images:list, return_smoothe_faceboxes=False) -> list:
        return self._smoothe_optimization(images, 
            return_smoothe_faceboxes=return_smoothe_faceboxes, 
            return_smoothe_landmarks=True
        )


    def _smoothe_optimization(self, images:list, 
        return_smoothe_faceboxes=False, 
        return_smoothe_landmarks=False
    ) -> list:
        new_boxes = []
        new_landmarks = []

        # for debagings
        self.test_gsv = []
        self.test_area = []
        self.test_shift_x = []
        self.test_shift_y = []
        self.w_list = []
        self.lsv_list = []

        iterator = tqdm(range(len(images)))

        for _ in iterator:
            if self._counter == 0:
                # array with shape [x1, y1, x2, y2]
                self._fb = self._predict_face.predict_face_box_single(images[self._counter])
                # array with shape [68, 2]
                self._fb_landmarks = self._predict_landmarks.predict_landmarks_in_certain_space(images[self._counter], *self._fb)
                self._recalculate_variables_landmarks()
                self._is_landmarks_variable_restored = False

                new_boxes.append(self._fb)
                new_landmarks.append(self._fb_landmarks)
                self._counter += 1
                continue

            # array with shape [x1, y1, x2, y2]
            self._dfb = self._predict_face.predict_face_box_single(images[self._counter])
            # array with shape [68, 2]
            self._dfb_landmarks = self._predict_landmarks.predict_landmarks_in_certain_space(images[self._counter], *self._dfb)

            if self._dfb is None or self._dfb_landmarks is None:
                self._counter += 1
                continue

            need_to_restore_box = False

            ### 1. Different in area if window
            diff_area = abs(self._norm_s(*self._dfb) - self._norm_s(*self._fb)) / self._norm_s(*self._dfb)
            diff_area_bool = diff_area > self._area_diff
            if type(diff_area_bool) == np.ndarray:
                diff_area_bool = diff_area_bool.max()

            need_to_restore_box = need_to_restore_box or diff_area_bool
            if need_to_restore_box:
                print('area')
            self.test_area.append(diff_area)

            ### 2. Shift in coordinate
            if not need_to_restore_box:
                shift_x = abs(self._norm_x(self._dfb[0]) - self._norm_x(self._fb[0]))
                shift_y = abs(self._norm_y(self._dfb[1]) - self._norm_x(self._fb[1]))

                shift_x_bool = shift_x > self._shift_x
                shift_y_bool = shift_y > self._shift_y

                if type(shift_x_bool) == np.ndarray:
                    shift_x_bool = shift_x_bool.max()

                if type(shift_y_bool) == np.ndarray:
                    shift_y_bool = shift_y_bool.max()

                need_to_restore_box = need_to_restore_box or shift_x_bool or shift_y_bool
                self.test_shift_x.append(shift_x)
                self.test_shift_y.append(shift_y)

                if need_to_restore_box:
                    print('shift')

            ### 3. Calculate GSV and compare movement of face according to landmarks
            self._calculate_GSV()
            self.test_gsv.append(self._gsv)
            need_to_restore_landmarks = self._gsv > self._noc
            if type(need_to_restore_landmarks) == np.ndarray:
                need_to_restore_landmarks = need_to_restore_landmarks.max()

            if need_to_restore_landmarks:
                print('gsv')


            if need_to_restore_box:
                self._recalculate_variables_box()
            else:
                ### Smoothe facebox, if previous checks statements is good
                self._calc_new_box()

            if need_to_restore_landmarks:
            	self._recalculate_variables_landmarks()
            else:
                if not self._is_landmarks_variable_restored:
                    self._recalculate_landmark_variables()
                
                self._calc_new_landmarks()


            self.w_list.append(self._w)
            self.lsv_list.append(self._lsv)

            new_boxes.append(self._fb)
            new_landmarks.append(self._fb_landmarks)
            self._counter += 1

        iterator.close()
        return new_boxes, new_landmarks

    def _norm_s(self, x1, y1, x2, y2):
        half_w = self._image_w / 2.0
        half_h = self._image_h / 2.0
        x2 = (x2 / half_w) - 1.0
        x1 = (x1 / half_w) - 1.0
        y2 = (y2 / half_h) - 1.0
        y1 = (y1 / half_h) - 1.0
        return (x2 - x1) * (y2 - y1)

    def _norm(self, x1, y1, x2, y2):
        half_w = self._image_w / 2.0
        half_h = self._image_h / 2.0
        x2 = (x2 / half_w) - 1.0
        x1 = (x1 / half_w) - 1.0
        y2 = (y2 / half_h) - 1.0
        y1 = (y1 / half_h) - 1.0
        
        return x1, y1, x2, y2

    def _norm_x(self, x):
        half_w = self._image_w / 2.0
        return (x / half_w) - 1.0

    def _norm_y(self, y):
        half_h = self._image_h / 2.0
        return (y / half_h) - 1.0

    def _calculate_GSV(self):
        # for easy access
        old = self._fb_landmarks
        new = self._dfb_landmarks
        # Choose 7 landmarks
        seven_landmarks_old = np.stack([
            old[30], # top of the nose
            
            old[45], # right eye, right corner
            old[42], # right eye, left corner
            old[39], # left eye, right corner
            old[36], # left eye, left corner
            
            old[48], # mouth, left corner
            old[54], # mouth, right corner
        ], axis=0)
        seven_landmarks_new = np.stack([
            new[30], # top of the nose
            
            new[45], # right eye, right corner
            new[42], # right eye, left corner
            new[39], # left eye, right corner
            new[36], # left eye, left corner
            
            new[48], # mouth, left corner
            new[54], # mouth, right corner
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

    def _calc_new_box(self):
        if self._box_update_using_decay:
            self._fb = self._decay_box * self._fb + (1 - self._decay_box) * self._dfb
        else:
            ### Smoothe facebox, if previous checks statements is good
            new_box = np.matmul(self._dfb_landmarks.transpose(), self._yo)

            self._fb = [
                new_box[0][0], # x1
                new_box[0][1], # y1
                new_box[1][0], # x2
                new_box[1][1], # y2
            ]

    def _recalculate_variables_box(self):
        if self._dfb is not None:
            self._fb = self._dfb
            self._dfb = None

        self.number_of_restore += 1

    def _recalculate_variables_landmarks(self):
        if self._dfb_landmarks is not None:
            self._fb_landmarks = self._dfb_landmarks
            self._dfb_landmarks = None

        box = np.array([
            [self._fb[0], self._fb[1]],
            [self._fb[2], self._fb[3]],
        ]).astype(np.float32)

        ### Final product: matrix [68, 2]
        ### Solve: a * x = b, where x is `yo`, a is `fb_landmarks`, b is `box`
        self._yo, _, _, _ = np.linalg.lstsq(self._fb_landmarks.transpose(), box)

        self.number_of_restore += 1


    def _recalculate_landmark_variables(self):
        self._calculate_lsv()
        if self._lamda_update_in_opt:
            #self._lamda = 1.0 / self._lsv
            pass

        if self._t_update_in_opt:
            #self._t = - (1 / self._noc)
            pass

        self._is_landmarks_variable_restored = True

    def reset_variables(self):
        self._yo = None
        self._gsv = None
        self._counter = 0
