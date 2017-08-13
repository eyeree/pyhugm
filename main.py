from __future__ import print_function
from __future__ import division

print('loading...')

import collections
import datetime
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import threading

from colormath import color_objects, color_diff, color_conversions
import cv2
import numpy as np
import opc
import ui
import yaml

from pylepton import Lepton
from PyQt4 import QtGui, QtCore

import config_util


#### Color Stuff

R = 0
G = 1
B = 2

R_SLICE = slice(R, R+1)
G_SLICE = slice(G, G+1)
B_SLICE = slice(B, B+1)

def rgb(r, g, b):
    return np.uint8([r, g, b])

COLOR_BLACK  = rgb(0, 0, 0)
COLOR_GRAY   = rgb(128, 128, 128)
COLOR_WHITE  = rgb(255, 255, 255)
COLOR_RED    = rgb(255, 0, 0)
COLOR_GREEN  = rgb(0, 255, 0)
COLOR_BLUE   = rgb(0, 0, 255)
COLOR_YELLOW = COLOR_RED + COLOR_GREEN
COLOR_PURPLE = COLOR_RED + COLOR_BLUE
COLOR_MASK   = COLOR_PURPLE

COLOR_BEGIN  = COLOR_YELLOW
COLOR_END    = COLOR_PURPLE

def gamma_lut(correction):
    return np.array([((i / 255.0) ** (1.0 / correction)) * 255 for i in np.arange(0, 256)]).astype("uint8")

print('generating GAMMA_LUT')
GAMMA_CORRECTIONS = [ x / 100.0 for x in range(70, 130) ]
#print('GAMMA_CORRECTIONS', GAMMA_CORRECTIONS)
#[0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29]
NO_CORRECTION = GAMMA_CORRECTIONS.index(1.0)
GAMMA_LUT = [ gamma_lut(correction) for correction in GAMMA_CORRECTIONS ]

def rgb_adjust(r, g, b):
    return np.uint8([r, g, b])

NO_COLOR_ADJUSTMENT = rgb_adjust(NO_CORRECTION, NO_CORRECTION, NO_CORRECTION)

DEFAULT_COLOR_ADJUSTMENT = rgb_adjust(GAMMA_CORRECTIONS.index(0.85), GAMMA_CORRECTIONS.index(0.85), GAMMA_CORRECTIONS.index(0.85))


#### Pixel Stuff

NUM_PIXELS = 1024

S_COUNT = 10
R_COUNT = 16

S_HALF = 12
S_CUT = 1.0 + (1.0 / S_HALF)

R_LEN = 32
R_MIN = 0
R_MAX = R_LEN - 1

S_WIDTH = 14.5 / S_HALF
S_HEIGHT = 14.0 / S_COUNT

S_WH_RATIO = S_HEIGHT / S_WIDTH
#print('S_WH_RATIO', S_WH_RATIO, S_WIDTH, S_HEIGHT, S_HALF, S_CUT)

Strand = collections.namedtuple('Strand',
    [
        'index',
        'begin',
        'end',
        'length',
        'slice',
        'color_adjustment',
        'distance',
        'distance_inverse',
        'lepton_slice'
    ]
)


def strands(distance_function, end_point_list):

    result = []

    for end_points in end_point_list:

        index = len(result)

        begin, end, l, t, r, b, color_adjustment = end_points

        if begin < end:
            length = end - begin + 1
            slice_ = slice(begin, end + 1, 1)
        else:
            length = begin - end + 1
            slice_ = slice(begin, end - 1, -1)

        distance = distance_function(index, length)
        distance_inverse = 1.0 - distance

        result.append(
            Strand(
                index,
                begin,
                end,
                length,
                slice_,
                color_adjustment.copy(),
                distance,
                distance_inverse,
                (slice(t,b), slice(l,r))
            )
        )

    return result

def s_index_to_y_dist(index):
    return float(index + (S_HALF - S_COUNT)) * 1.025

S_DIST_NORM = math.hypot(0, s_index_to_y_dist(0)) / (S_HALF - 1)

def make_s_dist(index, length):
    y = s_index_to_y_dist(index)
    half = int((length - 1) / 2)
    result = np.float16([math.hypot(x, y) / (S_HALF - 1) for x in range(-half, half + 1)])
    #print('s_dist', index, result)
    return result

def make_r_dist(index, length):
    result = np.float16([ i / (length - 1) for i in range(0, length) ])
    #print('r_dist', index, result)
    return result

S_STRANDS = strands(make_s_dist,
    [
        (384, 406,  5, 56, 75, 60, DEFAULT_COLOR_ADJUSTMENT),    # 0
        (192, 214,  5, 52, 75, 56, DEFAULT_COLOR_ADJUSTMENT),    # 1
        (256, 276,  5, 48, 75, 52, DEFAULT_COLOR_ADJUSTMENT),    # 2
        (724, 704,  5, 44, 75, 48, DEFAULT_COLOR_ADJUSTMENT),    # 3
        (658, 640, 10, 40, 70, 44, DEFAULT_COLOR_ADJUSTMENT),    # 4
        (448, 464, 15, 36, 65, 40, DEFAULT_COLOR_ADJUSTMENT),    # 5
        (479, 465, 20, 32, 60, 36, DEFAULT_COLOR_ADJUSTMENT),    # 6
        (659, 671, 25, 28, 55, 32, DEFAULT_COLOR_ADJUSTMENT),    # 7
        (223, 215, 30, 24, 50, 28, NO_COLOR_ADJUSTMENT),         # 8
        (725, 729, 35, 20, 45, 24, DEFAULT_COLOR_ADJUSTMENT)     # 9
    ]
)

R_STRANDS = strands(make_r_dist,
    [
        (128, 159,  0,  0,  5, 60, NO_COLOR_ADJUSTMENT),       # 0 (L)
        ( 64,  95,  5,  0, 10, 44, NO_COLOR_ADJUSTMENT),       # 1
        (320, 351, 10,  0, 15, 40, DEFAULT_COLOR_ADJUSTMENT),  # 2
        (960, 991, 15,  0, 20, 36, NO_COLOR_ADJUSTMENT),       # 3
        (  0,  31, 20,  0, 25, 32, NO_COLOR_ADJUSTMENT),       # 4
        (224, 255, 25,  0, 30, 28, NO_COLOR_ADJUSTMENT),       # 5
        (576, 607, 30,  0, 35, 28, NO_COLOR_ADJUSTMENT),       # 6
        (480, 511, 35,  0, 40, 24, NO_COLOR_ADJUSTMENT),       # 7
        (768, 799, 40,  0, 45, 24, NO_COLOR_ADJUSTMENT),       # 8
        (896, 927, 45,  0, 50, 28, DEFAULT_COLOR_ADJUSTMENT),  # 9
        (832, 863, 50,  0, 55, 28, NO_COLOR_ADJUSTMENT),       # 10
        (512, 543, 55,  0, 60, 32, NO_COLOR_ADJUSTMENT),       # 11
        (730, 761, 60,  0, 65, 36, NO_COLOR_ADJUSTMENT),       # 12
        (672, 703, 65,  0, 70, 40, NO_COLOR_ADJUSTMENT),       # 13
        (277, 308, 70,  0, 75, 44, DEFAULT_COLOR_ADJUSTMENT),  # 14
        (407, 438, 75,  0, 80, 60, DEFAULT_COLOR_ADJUSTMENT)   # 15 (R)
    ]
)

#print('\n***************************************************************************\n')
#print('R_STRANDS\n ', '\n  '.join([str(strand) for strand in R_STRANDS]))
#print('\n***************************************************************************\n')
#print('S_STRANDS\n ', '\n  '.join([str(strand) for strand in S_STRANDS]))
#print('\n***************************************************************************\n')

ALL_STRANDS = []
ALL_STRANDS.extend(R_STRANDS)
ALL_STRANDS.extend(S_STRANDS)

COLOR_ADJUSTED_STRANDS = [ strand for strand in ALL_STRANDS if not np.all(strand.color_adjustment == 1.0) ]

PIXEL_DIST = np.zeros(NUM_PIXELS, dtype=np.float16)
PIXEL_DIST_INVERSE = np.zeros(NUM_PIXELS, dtype=np.float16)
TOTAL_PIXEL_DIST = np.zeros(NUM_PIXELS, dtype=np.float16)
TOTAL_PIXEL_DIST_INVERSE = np.zeros(NUM_PIXELS, dtype=np.float16)

TOTAL_PIXEL_DIST_S_TO_R_RATIO = 1.0 / 3.0
TOTAL_PIXEL_DIST_S_TO_R_RATIO_INVERSE = 1.0 - TOTAL_PIXEL_DIST_S_TO_R_RATIO

S_MASK = np.zeros(NUM_PIXELS, dtype=np.bool)
R_MASK = np.zeros(NUM_PIXELS, dtype=np.bool)

for strand in R_STRANDS:
    PIXEL_DIST[strand.slice] = strand.distance
    PIXEL_DIST_INVERSE[strand.slice] = strand.distance_inverse
    TOTAL_PIXEL_DIST[strand.slice] = TOTAL_PIXEL_DIST_S_TO_R_RATIO + (strand.distance * TOTAL_PIXEL_DIST_S_TO_R_RATIO_INVERSE)
    TOTAL_PIXEL_DIST_INVERSE[strand.slice] = strand.distance_inverse * TOTAL_PIXEL_DIST_S_TO_R_RATIO_INVERSE
    R_MASK[strand.slice] = True

s_dist_min = 1000
s_dist_max = -1000

s_dist_min_inverse = 1000
s_dist_max_inverse = -1000

for strand in S_STRANDS:
    strand_min = strand.distance.min()
    strand_max = strand.distance.max()
    if strand_min < s_dist_min: s_dist_min = strand_min
    if strand_max > s_dist_max: s_dist_max = strand_max
    strand_min_inverse = strand.distance_inverse.min()
    strand_max_inverse = strand.distance_inverse.max()
    if strand_min_inverse < s_dist_min_inverse: s_dist_min_inverse = strand_min_inverse
    if strand_max_inverse > s_dist_max_inverse: s_dist_max_inverse = strand_max_inverse

s_dist_range = s_dist_max - s_dist_min
s_dist_range_inverse = s_dist_max_inverse - s_dist_min_inverse

for strand in S_STRANDS:
    PIXEL_DIST[strand.slice] = (strand.distance - s_dist_min) / s_dist_range
    PIXEL_DIST_INVERSE[strand.slice] = (strand.distance_inverse - s_dist_min_inverse) / s_dist_range_inverse
    TOTAL_PIXEL_DIST[strand.slice] = strand.distance * TOTAL_PIXEL_DIST_S_TO_R_RATIO
    TOTAL_PIXEL_DIST_INVERSE[strand.slice] = TOTAL_PIXEL_DIST_S_TO_R_RATIO_INVERSE + (strand.distance_inverse * TOTAL_PIXEL_DIST_S_TO_R_RATIO)
    S_MASK[strand.slice] = True

S_PIXEL_COUNT = np.sum(S_MASK)
R_PIXEL_COUNT = np.sum(R_MASK)


def diff_colors(start_color, end_color):
    return np.int16(end_color) - start_color


def interp_color_fn(start_color, end_color, ease = lambda x : x):

    diff_color = diff_colors(start_color, end_color)

    #print('interp_color_fn', start_color, end_color, diff_color, ease)

    def interp_color(i):
        result = np.clip(start_color + diff_color * ease(i), 0, 255, np.empty(3, dtype=np.uint8))
        #print('interp_color', i, ease(i), start_color, end_color, result)
        return result

    return interp_color


def ubound_color_fn(fn, ubound, end_color, ubound_color, ease = lambda x: x, lbound = 1.0):

    #print('ubound_color_fn', ubound, end_color, ubound_color, ease, lbound)

    interp_ubound_color = interp_color_fn(end_color, ubound_color, ease)

    def apply_ubound_color(i):
        if i <= lbound:
            #print('ubound_color i <= lbound', i, lbound, fn(i))
            return fn(i)
        elif i >= ubound:
            #print('ubound_color i >= ubound', i, ubound, ubound_color)
            return ubound_color
        else:
            #print('ubound_color', i, lbound, ubound, 1.0 - ((i - lbound) / (ubound - lbound)), (i - lbound) / (ubound - lbound), i - lbound, ubound - lbound)
            return interp_ubound_color((i - lbound) / (ubound - lbound))

    return apply_ubound_color


def lbound_color_fn(fn, lbound, lbound_color, start_color, ease = lambda x: x, ubound = 0.0):

    interp_lbound_color = interp_color_fn(lbound_color, start_color, ease)

    def apply_lbound_color(i):
        if i >= ubound:
            return fn(i)
        elif i <= lbound:
            return lbound_color
        else:
            return interp_lbound_color(1.0 - ((i - ubound) / (ubound - lbound)))

    return apply_lbound_color


def easing_curve_fn(easing_curve_type):
    easing_curve = QtCore.QEasingCurve(easing_curve_type)
    def ease(i):
        return easing_curve.valueForProgress(i)
    return ease


MIN_COLOR_SUM = 16
MAX_COLOR_SUM = 3 * 192
MIN_COLOR_DELTA = 16
MIN_DELTA_E = 30.0

def random_color(other_than = None):
    for i in range(1,10):

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        s = r + g + b

        if s < MIN_COLOR_SUM or s > MAX_COLOR_SUM:
            print('rejected sum', s, r, g, b)
            continue

        if other_than is not None:
            other_rgb = color_objects.sRGBColor(other_than[R], other_than[G], other_than[B], is_upscaled = True)
            check_rgb = color_objects.sRGBColor(r, g, b, is_upscaled = True)
            other_lab = color_conversions.convert_color(other_rgb, color_objects.LabColor)
            check_lab = color_conversions.convert_color(check_rgb, color_objects.LabColor)
            delta_e = color_diff.delta_e_cie2000(other_lab, check_lab)
            if delta_e <= MIN_DELTA_E:
                print('rejected delta_e', delta_e, r, g, b, other_than)
                continue

        break

    return rgb(r, g, b)


class DisplayBase(QtCore.QObject):

    def __init__(self):
        super(DisplayBase, self).__init__()


def identity(x):
    return x


EASING_FN = [ np.vectorize(QtCore.QEasingCurve(i).valueForProgress, doc = 'QEasingCurve({})'.format(i), otypes=[np.float32]) for i in range(0, 41) ]


def choice_fn(options):

    weights, choices = zip(*options)

    sum_weights = sum(weights)
    probabilities = [ w / sum_weights for w in weights ]

    #print('choice_fn', choices, probabilities)

    def choose():
        return np.random.choice(choices, p=probabilities)

    return choose


class ConfiguredDisplay(DisplayBase):

    PICK_NEW_DISPLAY_DELTA = 3

    MODE_SHOW_NORMAL      = 'SHOW_NORMAL'
    MODE_NORMAL_TO_RANDOM = 'NORMAL_TO_RANDOM'
    MODE_SHOW_RANDOM      = 'SHOW_RANDOM'
    MODE_RANDOM_TO_NORMAL = 'RANDOM_TO_NORMAL'
    MODE_RANDOM_TO_RANDOM = 'RANDOM_TO_RANDOM'
    MODE_SHOW_LEPTON      = 'SHOW_LEPTON'
    MODE_ECLIPSE          = 'ECLIPSE'

    MODE_DURATION = {
        MODE_SHOW_NORMAL:       5.0,
        MODE_NORMAL_TO_RANDOM: 15.0,
        MODE_SHOW_RANDOM:       0.1,
        MODE_RANDOM_TO_NORMAL:  5.0,
        MODE_RANDOM_TO_RANDOM: 10.0,
        MODE_SHOW_LEPTON:      10.0
    }

    MIN_SPEED = 0.10   # fast
    MAX_SPEED = 1.00    # slow

    MIN_SPEED_DELTA = 0.01
    MAX_SPEED_DELTA = 0.10

    PULSE_TIME = 1.0

    def __init__(self, completeCutoff):
        super(ConfiguredDisplay, self).__init__()

        self.speed = 0.5
        self.mode_change_time = 0
        self.mode_start_time = 0
        self.mode = self.MODE_SHOW_LEPTON
        self.next_mode = None
        self.percent_complete_time = None
        self.pulse_end_time = None
        self.completeCutoff = completeCutoff

        self.normal_sun_pixels = self.make_sun_pixels(center_color = rgb(255, 223, 147), edge_color = rgb(255, 180, 0))
        self.from_pixels = self.normal_sun_pixels

        self.color_choice = choice_fn(
            [
                (1, self.color_normal_prepare),
                (3, self.color_pulling_prepare),
                (3, self.color_pushing_prepare),
                (2, self.color_random_prepare)
            ]
        )

        self.trans_choice = choice_fn(
            [
                (1, self.trans_blend_prepare),
                (1, self.trans_push_prepare),
                (1, self.trans_pull_prepare),
                (1, self.trans_noise_prepare)
            ]
        )

        self.make_random_sun_pixels()

        self.eclipse_sun_pixels = self.make_sun_pixels(center_color = COLOR_BLACK, edge_color = rgb(128, 0, 128), easing_fn_index = QtCore.QEasingCurve.InExpo)

        self.eclipse_effect = self.make_sun_pixels(center_color = COLOR_BLACK, edge_color = rgb(255, 0, 255), easing_fn_index = QtCore.QEasingCurve.InExpo)
        for strand in R_STRANDS:
            self.eclipse_effect[strand.slice] = COLOR_WHITE

        self.eclipse_color_diff = self.eclipse_effect - self.eclipse_sun_pixels


    def update(self, frame_time, pixels, lepton_data, movement):

        if self.mode == self.MODE_SHOW_LEPTON:

            pixels[:] = self.normal_sun_pixels

            if lepton_data is None:
                print('no lepton data')
            else:

                count_complete = 0

                for strand in S_STRANDS:
                    strip = lepton_data[strand.lepton_slice]
                    strip = cv2.resize(strip, (strand.length, 1))[0]
                    mask = (strip != COLOR_WHITE).all(1)
                    count_complete += np.sum(mask)
                    np.copyto(pixels[strand.slice], strip, where = mask[:,None])

                percent_complete = count_complete / S_PIXEL_COUNT
                percent_complete_inverse = 1.0 - percent_complete

                for strand in R_STRANDS:
                    strip = lepton_data[strand.lepton_slice]
                    strip = cv2.resize(strip, (1, strand.length))[:,0]
                    strip = np.flipud(strip)
                    mask = (strip != COLOR_WHITE).all(1)
                    np.copyto(pixels[strand.slice], strip, where = mask[:,None])
                    pixels[strand.slice] *= percent_complete_inverse

                mode_time = frame_time.current - self.mode_start_time
                if mode_time < 1.0:

                    percent_mode_time = mode_time / 1.0

                    cv2.addWeighted(pixels, percent_mode_time, self.normal_sun_pixels, 1.0 - percent_mode_time, 0.0, pixels)

                else:

                    if percent_complete >= self.completeCutoff.value():
                        if self.percent_complete_time:
                            if frame_time.current >= self.percent_complete_time:

                                pixels[:] = self.eclipse_sun_pixels

                                effecttive_pulse_time = self.PULSE_TIME * self.speed

                                if self.pulse_end_time:

                                    pulse_percent = (effecttive_pulse_time - (self.pulse_end_time - frame_time.current)) / effecttive_pulse_time

                                    pulse_dist = np.abs(TOTAL_PIXEL_DIST - pulse_percent)
                                    pulse_dist_i = 1.0 - pulse_dist

                                    fn = EASING_FN[QtCore.QEasingCurve.InElastic]
                                    eased_pulse_dist = fn(pulse_dist)
                                    eased_pulse_dist_i = fn(pulse_dist_i)

                                    pixels += self.eclipse_color_diff * eased_pulse_dist[:,None]
                                    pixels += self.eclipse_color_diff * eased_pulse_dist_i[:,None]

                                    if frame_time.current >= self.pulse_end_time:
                                        self.pulse_end_time = frame_time.current + effecttive_pulse_time

                                else:

                                    self.pulse_end_time = frame_time.current + effecttive_pulse_time
                        else:
                            #print(frame_time.current, 'complete')
                            self.percent_complete_time = frame_time.current + 1
                    else:
                        self.percent_complete_time = None

                if not movement:
                    if frame_time.current >= self.mode_change_time:
                        self.set_mode(frame_time, self.MODE_SHOW_NORMAL)
                        self.start(self.make_random_sun_pixels)
                else:
                    self.mode_change_time = frame_time.current + self.MODE_DURATION[self.MODE_SHOW_LEPTON]

        elif self.mode == self.MODE_SHOW_NORMAL:

            pixels[:] = self.normal_sun_pixels

            if movement:
                self.set_mode(frame_time, self.MODE_SHOW_LEPTON)
            elif frame_time.current >= self.mode_change_time:
                self.set_mode(frame_time, self.MODE_NORMAL_TO_RANDOM)

        elif self.mode == self.MODE_SHOW_RANDOM:

            pixels[:] = self.from_pixels

            if movement:
                self.set_mode(frame_time, self.MODE_RANDOM_TO_NORMAL)
            elif frame_time.current >= self.mode_change_time:
                if self.next_mode is None:
                    print('next_mode is None')
                else:
                    self.set_mode(frame_time, self.next_mode)

        else:

            percent = (frame_time.current - self.mode_start_time) / (self.mode_change_time - self.mode_start_time)

            #percents = np.array([ 1.0 if d <= percent else self.trans_ease(percent / d) for d in self.trans_dist ])
            percents = self.trans_ease(percent / self.trans_dist)
            colors = self.from_pixels + (self.pixel_diffs * percents[:,None])
            np.clip(colors, 0, 255, pixels)

            if movement:
                if self.mode == self.MODE_NORMAL_TO_RANDOM or self.mode == self.MODE_RANDOM_TO_RANDOM:
                    inverse_duratation = (self.mode_change_time - self.mode_start_time) - (self.mode_change_time - frame_time.current)
                    #print(frame_time.current, 'inverse_duration', inverse_duratation, (self.mode_change_time - self.mode_start_time), (self.mode_change_time - frame_time.current))
                    self.set_mode(frame_time, self.MODE_RANDOM_TO_NORMAL)
                    self.mode_change_time = frame_time.current + inverse_duratation
                    self.from_pixels = self.to_pixels
                    self.to_pixels = self.normal_sun_pixels
                    self.pixel_diffs = np.int16(self.to_pixels) - self.from_pixels
                self.mode_change_time -= frame_time.delta # double speed

            if frame_time.current >= self.mode_change_time:
                self.from_pixels = self.to_pixels
                if self.mode == self.MODE_NORMAL_TO_RANDOM or self.mode == self.MODE_RANDOM_TO_RANDOM:
                    self.set_mode(frame_time, self.MODE_SHOW_RANDOM)
                    self.start(self.pick_next_color)
                else:
                    self.set_mode(frame_time, self.MODE_SHOW_NORMAL)
                    self.start(self.make_random_sun_pixels)

    def start(self, fn):
        if False:
            threading.Thread(target=fn).start()
        else:
            fn()


    def make_random_sun_pixels(self):
        self.center_color = random_color()
        self.edge_color = random_color(other_than = self.center_color)
        self.make_next_sun_pixels()


    def make_next_sun_pixels(self):
        self.to_pixels = self.make_sun_pixels(self.center_color, self.edge_color)
        self.pick_next_trans()


    def make_sun_pixels(self, center_color, edge_color, easing_fn_index = QtCore.QEasingCurve.InQuad):

        pixels = np.zeros([NUM_PIXELS, 3], dtype=np.uint8)

        r_ease = easing_curve_fn(QtCore.QEasingCurve.Linear)
        r_start_color = edge_color
        r_end_color = COLOR_BLACK
        r_compute = interp_color_fn(r_start_color, r_end_color, r_ease)
        r_first = R_STRANDS[0]
        pixels[r_first.slice] = [ r_compute(i) for i in r_first.distance ]
        for r in R_STRANDS[1:]: pixels[r.slice] = pixels[r_first.slice]

        s_ease = easing_curve_fn(easing_fn_index)
        s_start_color = center_color
        s_end_color = edge_color
        s_compute = ubound_color_fn(
            interp_color_fn(s_start_color, s_end_color, s_ease),
            S_CUT, s_end_color, COLOR_BLACK
        )
        for s in S_STRANDS:
            pixels[s.slice] = [ s_compute(i) for i in s.distance ]

        return pixels


    def pick_next_color(self):
        color_prepare = self.color_choice()
        #print(color_prepare.__name__)
        color_prepare()


    def color_normal_prepare(self):
        self.next_mode = self.MODE_RANDOM_TO_NORMAL
        #print('next_mode', self.next_mode)
        self.to_pixels = self.normal_sun_pixels
        self.pick_next_trans()


    def color_random_prepare(self):
        self.next_mode = self.MODE_RANDOM_TO_RANDOM
        #print('next_mode', self.next_mode)
        self.start(self.make_random_sun_pixels)


    def color_pulling_prepare(self):
        self.next_mode = self.MODE_RANDOM_TO_RANDOM
        #print('next_mode', self.next_mode)
        self.center_color = self.edge_color
        self.edge_color = random_color(other_than = self.center_color)
        self.start(self.make_next_sun_pixels)


    def color_pushing_prepare(self):
        self.next_mode = self.MODE_RANDOM_TO_RANDOM
        #print('next_mode', self.next_mode)
        self.edge_color = self.center_color
        self.center_color = random_color(other_than = self.edge_color)
        self.start(self.make_next_sun_pixels)


    def pick_next_trans(self):
        #self.trans_ease = random.choice(EASING_FN)
        #print(self.trans_ease.__doc__)
        self.trans_ease = EASING_FN[QtCore.QEasingCurve.Linear]
        self.pixel_diffs = np.int16(self.to_pixels) - self.from_pixels
        trans_prepare = self.trans_choice()
        #print(trans_prepare.__name__)
        trans_prepare()


    def trans_blend_prepare(self):
        self.trans_dist = np.ones(NUM_PIXELS, dtype=np.float16)


    def trans_push_prepare(self):
        self.trans_dist = TOTAL_PIXEL_DIST


    def trans_pull_prepare(self):
        self.trans_dist = TOTAL_PIXEL_DIST_INVERSE


    def trans_noise_prepare(self):
        self.trans_dist = PIXEL_DIST * 0.05
        self.trans_dist += 0.95
        self.trans_dist[S_MASK] = np.random.random(S_PIXEL_COUNT)


    def set_mode(self, frame_time, mode):

        self.mode = mode
        self.mode_change_time = frame_time.current + (self.MODE_DURATION[mode] * self.speed)
        self.mode_start_time = frame_time.current

        old_speed = self.speed
        new_speed = old_speed + ((EASING_FN[QtCore.QEasingCurve.InOutExpo](random.random()) * (self.MAX_SPEED_DELTA * 2)) - self.MAX_SPEED_DELTA)
        if new_speed > self.MAX_SPEED:
            new_speed = self.MAX_SPEED - (new_speed - self.MAX_SPEED)
        elif new_speed < self.MIN_SPEED:
            new_speed = self.MIN_SPEED + (self.MIN_SPEED - new_speed)
        self.speed = new_speed

        print(frame_time.current, '******* mode *******', mode, old_speed, new_speed)


class DisplaySun(DisplayBase):


    def __init__(self, center_color = rgb(255, 223, 147), edge_color = rgb(255, 180, 0)):
        super(DisplaySun, self).__init__()

        pixels = np.zeros([NUM_PIXELS, 3], dtype=np.uint8)

        r_ease = easing_curve_fn(QtCore.QEasingCurve.Linear)
        r_start_color = edge_color
        r_end_color = COLOR_BLACK
        r_compute = interp_color_fn(r_start_color, r_end_color, r_ease)
        r_first = R_STRANDS[0]
        pixels[r_first.slice] = [ r_compute(i) for i in r_first.distance ]
        for r in R_STRANDS[1:]: pixels[r.slice] = pixels[r_first.slice]

        s_ease = easing_curve_fn(QtCore.QEasingCurve.InQuad)
        s_start_color = center_color
        s_end_color = edge_color
        s_compute = ubound_color_fn(
            interp_color_fn(s_start_color, s_end_color, s_ease),
            S_CUT, s_end_color, COLOR_BLACK
        )
        for s in S_STRANDS:
            pixels[s.slice] = [ s_compute(i) for i in s.distance ]

        self.__pixels = pixels


    def update(self, frame_time, pixels, lepton_data, movement):
        pixels[:] = self.__pixels


class DisplayColor(DisplayBase):

    TARGET_R = 1
    TARGET_S = 2

    color_changed = QtCore.pyqtSignal(np.ndarray)
    adjustment_changed = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, initial_color, initial_r_index, initial_s_index):
        super(DisplayColor, self).__init__()
        self.__color = initial_color.copy()
        self.__r_index = initial_r_index
        self.__s_index = initial_s_index
        self.__target = self.TARGET_R

    def set_r_index(self, index):
        self.__r_index = index
        self.__target = self.TARGET_R
        self.adjustment_changed.emit(self.get_adjust())

    def set_s_index(self, index):
        self.__s_index = index
        self.__target = self.TARGET_S
        self.adjustment_changed.emit(self.get_adjust())

    def set_red_color(self, red):
        self.__color[R] = red
        self.color_changed.emit(self.__color)

    def set_green_color(self, green):
        self.__color[G] = green
        self.color_changed.emit(self.__color)

    def set_blue_color(self, blue):
        self.__color[B] = blue
        self.color_changed.emit(self.__color)

    def set_red_adjust(self, correction):
        self.__set_adjust(correction, R)

    def set_green_adjust(self, correction):
        self.__set_adjust(correction, G)

    def set_blue_adjust(self, correction):
        self.__set_adjust(correction, B)

    def __set_adjust(self, correction, index):

        target_strand = self.__get_target_strand()

        #print('__set_adjust', target_strand.color_adjustment, index, correction)

        target_strand.color_adjustment[index] = correction

        if np.all(target_strand.color_adjustment == NO_CORRECTION):
            if target_strand in COLOR_ADJUSTED_STRANDS:
                COLOR_ADJUSTED_STRANDS.remove(target_strand)
        else:
            if target_strand not in COLOR_ADJUSTED_STRANDS:
                COLOR_ADJUSTED_STRANDS.append(target_strand)

        self.adjustment_changed.emit(self.get_adjust())

    def __get_target_strand(self):
        if self.__target == self.TARGET_S:
            return S_STRANDS[self.__s_index]
        else:
            return R_STRANDS[self.__r_index]

    def get_adjust(self):
        return self.__get_target_strand().color_adjustment

    def set_all_off(self):
        self.__color[:] = COLOR_BLACK
        self.color_changed.emit(self.__color)

    def set_all_half(self):
        self.__color[:] = COLOR_GRAY
        self.color_changed.emit(self.__color)

    def set_all_full(self):
        self.__color[:] = COLOR_WHITE
        self.color_changed.emit(self.__color)

    def update(self, frame_time, pixels, lepton_data, movement):

        pixels[:] = self.__color

        if not np.all(self.__color ==  0):
            strand = self.__get_target_strand()
            pixels[strand.begin] = COLOR_BEGIN
            pixels[strand.end] = COLOR_END



class DisplayIndex(DisplayBase):

    def __init__(self, initial_pixel_index, initial_r_index, initial_s_index):
        super(DisplayIndex, self).__init__()
        self.__pixel_index = initial_pixel_index
        self.__r_index = initial_r_index
        self.__s_index = initial_s_index

    def set_r_index(self, r_index):
        self.__r_index = r_index

    def set_s_index(self, s_index):
        self.__s_index = s_index

    def set_pixel_index(self, pixel_index):
        self.__pixel_index = pixel_index

    def update(self, frame_time, pixels, lepton_data, movement):

        r = R_STRANDS[self.__r_index]
        pixels[r.slice] = COLOR_GRAY
        pixels[r.begin] = COLOR_BEGIN
        pixels[r.end] = COLOR_END

        s = S_STRANDS[self.__s_index]
        pixels[s.slice] = COLOR_GRAY
        pixels[s.begin] = COLOR_BEGIN
        pixels[s.end] = COLOR_END

        if self.__pixel_index > 0:
            pixels[self.__pixel_index - 1] = COLOR_RED
        pixels[self.__pixel_index] = COLOR_WHITE
        if self.__pixel_index < 1023:
            pixels[self.__pixel_index + 1] = COLOR_BLUE



class FrameTime(object):

    SLEEP_TIME = 1.0 / 30.0

    def __init__(self):
        self.__last_time = time.time()
        self.__last_delta = 0.0
        self.__last_display_time = self.__last_time
        self.__frame_count = 0
        self.__total_free_time = 0.0

    @property
    def current(self):
        return self.__last_time

    @property
    def delta(self):
        return self.__last_delta

    def tick(self):

        free_time = self.SLEEP_TIME - (time.time() - self.__last_time)
        self.__total_free_time += free_time
        time.sleep(max(free_time, 0.0))

        current_time = time.time()

        self.__last_delta = current_time - self.__last_time
        self.__last_time = current_time

        self.__frame_count += 1
        display_delta_time = current_time - self.__last_display_time
        if display_delta_time >= 30:
            if True:
                print(
                    'update fps', self.__frame_count / display_delta_time,
                    'avg free %', int((self.__total_free_time / display_delta_time) * 100))
            self.__frame_count = 0
            self.__last_display_time = current_time
            self.__total_free_time = 0.0


def lepton_process(connection, lepton_device):
    print('lepton process running')
    with Lepton(lepton_device) as lepton:
        while not connection.poll():
            lepton_frame = lepton.capture()
            connection.send(lepton_frame)
    print('lepton process exiting')
    connection.close()
    print('lepton process exited')


class LeptonThread(QtCore.QThread):

    LEPTON_DEVICE = "/dev/spidev0.1"

    lepton_frame_captured = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super(LeptonThread, self).__init__(parent)
        parent_conn, child_conn = multiprocessing.Pipe()
        self.__lepton_process = multiprocessing.Process(target=lepton_process, args=(child_conn, self.LEPTON_DEVICE))
        self.__lepton_process.start()
        self.__connection = parent_conn

    @QtCore.pyqtSlot()
    def stop(self):
        print('lepton thread stop requested')
        self.__connection.send({'stop': True})
        self.__connection.close()

    def run(self):

        print('lepton thread running')

        try:

            while True:
                lepton_frame = self.__connection.recv()
                self.lepton_frame_captured.emit(lepton_frame)

        except IOError:
            pass

        print('lepton thread exiting')
        self.__lepton_process.join()
        print('lepton thread exited')


class UpdateThread(QtCore.QThread):

    OPC_ADDRESS = 'localhost:7890'
    OPC_HEADER_SIZE = 4

    def __init__(self, parent, initial_display):
        QtCore.QThread.__init__(self, parent)
        self.__exiting = False
        self.__display = initial_display
        self.__opc_client = opc.Client(self.OPC_ADDRESS)
        self.__lepton_frame = None
        self.__movement = None
        self.__color_adjustment_enabled = False

    @QtCore.pyqtSlot()
    def stop(self):
        print('update thread stop requested')
        self.__exiting = True

    @QtCore.pyqtSlot()
    def set_display(self, display):
        self.__display = display

    @QtCore.pyqtSlot()
    def set_lepton_frame(self, lepton_frame, movement):
        self.__lepton_frame = lepton_frame
        self.__movement = movement

    @QtCore.pyqtSlot()
    def set_color_adjustment_enabled(self, enabled):
        self.__color_adjustment_enabled = enabled

    def run(self):

        print('update thread running')

        opc_buffer = np.empty([(NUM_PIXELS * 3) + self.OPC_HEADER_SIZE], dtype=np.uint8)
        opc_buffer[0] = 0                          # channel
        opc_buffer[1] = 0                          # command = set color
        opc_buffer[2] = int(NUM_PIXELS * 3 / 256)  # len hi byte
        opc_buffer[3] = NUM_PIXELS * 3 % 256       # len lo byte

        pixels = opc_buffer[self.OPC_HEADER_SIZE:].reshape([NUM_PIXELS, 3])

        frame_time = FrameTime()

        while not self.__exiting:

            frame_time.tick()

            pixels.fill(0)

            self.__display.update(frame_time, pixels, self.__lepton_frame, self.__movement)

            if self.__color_adjustment_enabled:
                for strand in COLOR_ADJUSTED_STRANDS:
                    if strand.color_adjustment[R] != NO_CORRECTION:
                        pixels[strand.slice, R_SLICE] = GAMMA_LUT[strand.color_adjustment[R]][pixels[strand.slice, R_SLICE]]
                    if strand.color_adjustment[G] != NO_CORRECTION:
                        pixels[strand.slice, G_SLICE] = GAMMA_LUT[strand.color_adjustment[G]][pixels[strand.slice, G_SLICE]]
                    if strand.color_adjustment[B] != NO_CORRECTION:
                        pixels[strand.slice, B_SLICE] = GAMMA_LUT[strand.color_adjustment[B]][pixels[strand.slice, B_SLICE]]

            self.__opc_client.put_message(opc_buffer.tostring())

        print('update thread exited')


class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):

    def __init__(self):

        super(self.__class__, self).__init__()

        print('main window starting')

        self.setupUi(self)

        # buttons
        self.performFFCButton.clicked.connect(self.__perform_ffc)

        # configured display

        self.__configured_display = ConfiguredDisplay(self.completeCutoff)

        # index display

        self.__display_index = DisplayIndex(
            self.pixelIndexSpinBox.value(),
            self.rIndexSpinBox.value(),
            self.sIndexSpinBox.value())

        self.pixelIndexSpinBox.valueChanged.connect(
            lambda index: self.__display_index.set_pixel_index(index))

        self.pixelIndexStepSpinBox.valueChanged.connect(
            lambda value: self.pixelIndexSpinBox.setSingleStep(value))

        self.sIndexSpinBox.valueChanged.connect(
            lambda index: self.__display_index.set_s_index(index))

        self.rIndexSpinBox.valueChanged.connect(
            lambda index: self.__display_index.set_r_index(index))

        # color display

        self.__display_color = DisplayColor(
            rgb(self.redColorSpinBox.value(), self.greenColorSpinBox.value(), self.blueColorSpinBox.value()),
            self.rAdjustmentSpinBox.value(),
            self.sAdjustmentSpinBox.value())

        self.allFullPushButton.clicked.connect(
            self.__display_color.set_all_full)

        self.allHalfPushButton.clicked.connect(
            self.__display_color.set_all_half)

        self.allOffPushButton.clicked.connect(
            self.__display_color.set_all_off)

        self.redColorSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_red_color(value))

        self.greenColorSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_green_color(value))

        self.blueColorSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_blue_color(value))

        self.redColorSlider.valueChanged.connect(
            lambda value: self.__display_color.set_red_color(value))

        self.greenColorSlider.valueChanged.connect(
            lambda value: self.__display_color.set_green_color(value))

        self.blueColorSlider.valueChanged.connect(
            lambda value: self.__display_color.set_blue_color(value))

        self.config_adjustment_widgets(self.redAdjustmentDoubleSpinBox, self.redAdjustmentSlider)
        self.config_adjustment_widgets(self.greenAdjustmentDoubleSpinBox, self.greenAdjustmentSlider)
        self.config_adjustment_widgets(self.blueAdjustmentDoubleSpinBox, self.blueAdjustmentSlider)

        self.redAdjustmentDoubleSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_red_adjust(GAMMA_CORRECTIONS.index(round(value, 2))))

        self.greenAdjustmentDoubleSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_green_adjust(GAMMA_CORRECTIONS.index(round(value, 2))))

        self.blueAdjustmentDoubleSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_blue_adjust(GAMMA_CORRECTIONS.index(round(value, 2))))

        self.redAdjustmentSlider.valueChanged.connect(
            lambda value: self.__display_color.set_red_adjust(value))

        self.greenAdjustmentSlider.valueChanged.connect(
            lambda value: self.__display_color.set_green_adjust(value))

        self.blueAdjustmentSlider.valueChanged.connect(
            lambda value: self.__display_color.set_blue_adjust(value))

        self.rAdjustmentSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_r_index(value))

        self.sAdjustmentSpinBox.valueChanged.connect(
            lambda value: self.__display_color.set_s_index(value))

        self.adjustmentGroupBox.toggled.connect(
            lambda enabled: self.__update_thread.set_color_adjustment_enabled(enabled))

        self.__display_color.color_changed.connect(
            self.__on_color_changed)

        self.__display_color.adjustment_changed.connect(
            self.__on_adjustment_changed)

        # tool box
        self.__tool_box_displays = [self.__configured_display, self.__display_color, self.__display_index]
        self.toolBox.currentChanged.connect(
            lambda index: self.__update_thread.set_display(self.__tool_box_displays[index]))

        # update thread
        self.__update_thread = UpdateThread(self, self.__tool_box_displays[self.toolBox.currentIndex()])
        self.__update_thread.start()

        # lepton thread
        self.__lepton_thread = LeptonThread(self)
        self.__lepton_thread.lepton_frame_captured.connect(self.__lepton_frame_captured)
        self.__lepton_thread.start()

        self.fast_image = np.zeros([Lepton.ROWS, Lepton.COLS], dtype=np.float32)
        self.slow_image = np.zeros([Lepton.ROWS, Lepton.COLS], dtype=np.float32)
        self.movement_count = 0
        self.movement = False

        print('main window started')


    def __apply_config(self):

        text = str(self.configTextEdit.document().toPlainText())

        config = yaml.load(text)
        config = config_util.process_config(config)
        self.__configured_display.configure(config)

        return text


    def __save_config(self):

        try:

            text = self.__apply_config()

            old_name = str(self.pathEdit.text())
            new_name = 'config {}.yaml'.format(str(datetime.datetime.now()))
            new_path = os.path.join(os.path.dirname(old_name), 'backup', new_name)
            print('renaming {} to {}'.format(old_name, new_name))
            shutil.move(old_name, new_path)

            with open(old_name, 'w') as f:
                f.write(text)
                print('saved', old_name)

        except Exception as e:
            traceback.print_exc()
            QtGui.QMessageBox.critical(self, 'Could Not Save Config', e.message)


    def __edit_color(self):

        origional_text = str(self.configTextEdit.textCursor().selectedText())
        initial_color_parts = [ int(part.strip()) for part in origional_text.split(',') ]
        initial_color = QtGui.QColor(initial_color_parts[0], initial_color_parts[1], initial_color_parts[2])
        colorDialog = QtGui.QColorDialog(initial_color, self)

        def update_config(text):
            cursor = self.configTextEdit.textCursor()
            old_anchor= cursor.anchor()
            old_position= cursor.position()
            cursor.insertText(text)
            if old_anchor < old_position:
                new_anchor = old_anchor
                new_position = cursor.position()
            else:
                new_anchor = cursor.position()
                new_position = old_position
            cursor.setPosition(new_anchor, QtGui.QTextCursor.MoveAnchor)
            cursor.setPosition(new_position, QtGui.QTextCursor.KeepAnchor)
            self.configTextEdit.setTextCursor(cursor)
            self.__apply_config()

        def color_changed(color):
            text = '{}, {}, {}'.format(color.red(), color.green(), color.blue())
            update_config(text)

        def color_selected(color):
            colorDialog.close()
            color_changed(color)
            self.__save_config()

        def color_rejected():
            update_config(origional_text)
            colorDialog.close()

        colorDialog.currentColorChanged.connect(color_changed)
        colorDialog.colorSelected.connect(color_selected)
        colorDialog.rejected.connect(color_rejected)
        colorDialog .open()


    def __load_config(self):

        try:

            print('loading', self.pathEdit.text())

            with open(self.pathEdit.text(), 'r') as f:
                text = f.read()

            self.configTextEdit.document().setPlainText(text)

            self.__apply_config()

        except Exception as e:
            traceback.print_exc()
            QtGui.QMessageBox.critical(self, 'Could Not Load Config', e.message)


    def config_adjustment_widgets(self, spin_box, slider):
        spin_box.setDecimals(2)
        spin_box.setSingleStep(0.01)
        spin_box.setMinimum(GAMMA_CORRECTIONS[0])
        spin_box.setMaximum(GAMMA_CORRECTIONS[-1])
        spin_box.setValue(1.0)
        slider.setMinimum(0)
        slider.setMaximum(len(GAMMA_CORRECTIONS) - 1)
        slider.setValue(NO_CORRECTION)
        slider.tickInterval = 5

    def closeEvent(self, event):
        print('main window stopping')
        self.__lepton_thread.stop()
        self.__update_thread.stop()
        self.__lepton_thread.wait()
        self.__update_thread.wait()
        print('main window stopped')

    def __lepton_frame_captured(self, lepton_frame):
        self.__display_image(lepton_frame)

    def __perform_ffc(self):
        subprocess.call(['/home/pi/LeptonModule-master/software/flir_ffc/flir_ffc'])

    def __display_image(self, lepton_frame):

        clip_lower_bound = self.clipLower.value()
        clip_upper_bound = self.clipUpper.value()
        if clip_upper_bound < clip_lower_bound:
            clip_upper_bound = clip_lower_bound
        clip_range = clip_upper_bound - clip_lower_bound

        blur_size = self.blurSize.value()
        blur_sigma = self.blurSigma.value()

        zero_cutoff = self.zeroCutoff.value()
        mask_cutoff = self.maskCutoff.value()

        detect_top = self.detectTop.value()
        detect_left = self.detectLeft.value()
        detect_bottom = self.detectBottom.value()
        detect_right = self.detectRight.value()

        detect_slice = (slice(detect_top, detect_bottom + 1), slice(detect_left, detect_right + 1))
        detect_rect = ((detect_left, detect_top), (detect_right, detect_bottom))

        ease_fn = EASING_FN[self.easeFn.value()]

        # half assed filter that seems to work....
        np.clip(lepton_frame, clip_lower_bound, clip_upper_bound, lepton_frame)
        lepton_frame -= clip_lower_bound
        cv2.GaussianBlur(lepton_frame, (blur_size, blur_size), blur_sigma)
        lepton_frame[ lepton_frame < zero_cutoff ] = 0
        #print('lepton_frame', lepton_frame.min(), lepton_frame.max(), np.sum(lepton_frame < 100), np.sum(lepton_frame < 250), np.sum(lepton_frame < 400))

        gray = np.float32(lepton_frame) / clip_range

        cv2.accumulateWeighted(gray, self.slow_image, 0.05)
        cv2.accumulateWeighted(gray, self.fast_image, 0.7)

        mask = np.fliplr(self.fast_image <= mask_cutoff)
        #print('mask', mask.sum())

        #eased = np.fliplr(EASING_FN[QtCore.QEasingCurve.Linear](1.0 - self.fast_image))
        eased = np.subtract(1.0, np.fliplr(self.fast_image), dtype=np.float32)
        #print('eased fi', eased.dtype, eased.min(), eased.max(), np.sum(np.isnan(eased)))
        eased_min = eased.min()
        eased_max = eased.max()
        eased_ptp = eased_max - eased_min
        if eased_ptp:
            eased -= eased_min
            eased *= (1.0 / eased_ptp)
            #print('eased s', eased.dtype, eased.min(), eased.max(), np.sum(np.isnan(eased)))
        eased = ease_fn(eased)
        eased *= 255
        #print('eased c', eased.dtype, eased.min(), eased.max(), np.sum(np.isnan(eased)))
        color = cv2.cvtColor(eased, cv2.COLOR_GRAY2RGB)
        #print('color', color.shape, color.min(), color.max())
        rgb_data = np.uint8(color)
        #print('rgb', rgb_data.shape, rgb_data.dtype, rgb_data.min(), rgb_data.max())

        rgb_data[:,:,G] = 0
        rgb_data[mask] = COLOR_WHITE

        delta = cv2.absdiff(self.slow_image[detect_slice], self.fast_image[detect_slice])
        thresh = np.sum(cv2.threshold(delta, 0.3, 1.0, cv2.THRESH_BINARY)[1])
        #print('thresh', np.sum(thresh), np.sum(delta), delta.min(), delta.max())
        if self.movement_count > 0 and thresh < 3:
            self.movement_count -= 1
            #print('dec movement', self.movement_count)
            if self.movement_count == 0 and self.movement:
                print('movement stopped')
                self.movement = False
        elif self.movement_count < 10 and thresh > 10:
            self.movement_count += 1
            #print('inc movement', self.movement_count)
            if self.movement_count == 10 and not self.movement:
                print('movement started')
                self.movement = True

        self.__update_thread.set_lepton_frame(rgb_data, self.movement)

        image_data = rgb_data.copy()
        cv2.rectangle(image_data, detect_rect[0], detect_rect[1], (0, 255, 0), 1)

        bytesPerLine = 3 * Lepton.COLS
        image = QtGui.QImage(image_data.data, Lepton.COLS, Lepton.ROWS, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)

    def __on_color_changed(self, color):
        self.redColorSlider.setValue(color[R])
        self.redColorSpinBox.setValue(color[R])
        self.greenColorSlider.setValue(color[G])
        self.greenColorSpinBox.setValue(color[G])
        self.blueColorSlider.setValue(color[B])
        self.blueColorSpinBox.setValue(color[B])

    def __on_adjustment_changed(self, adjustment):
        #print('__on_adjustment_changed', adjustment)
        self.redAdjustmentSlider.setValue(adjustment[R])
        self.redAdjustmentDoubleSpinBox.setValue(GAMMA_CORRECTIONS[adjustment[R]])
        self.greenAdjustmentSlider.setValue(adjustment[G])
        self.greenAdjustmentDoubleSpinBox.setValue(GAMMA_CORRECTIONS[adjustment[G]])
        self.blueAdjustmentSlider.setValue(adjustment[B])
        self.blueAdjustmentDoubleSpinBox.setValue(GAMMA_CORRECTIONS[adjustment[B]])


def main():
    print('application starting')
    app = QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    result = app.exec_()
    print('application exiting')
    sys.exit(result)

if __name__ == '__main__':
    main()
