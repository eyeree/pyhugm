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

COLOR_BEGIN  = COLOR_YELLOW
COLOR_END    = COLOR_PURPLE

def gamma_lut(correction):
    return np.array([((i / 255.0) ** (1.0 / correction)) * 255 for i in np.arange(0, 256)]).astype("uint8")

print('generating GAMMA_LUT')
GAMMA_CORRECTIONS = [ x / 100.0 for x in range(70, 130) ]
#[ 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50 ]
NO_CORRECTION = GAMMA_CORRECTIONS.index(1.0)
GAMMA_LUT = [ gamma_lut(correction) for correction in GAMMA_CORRECTIONS ]

def rgb_adjust(r, g, b):
    return np.uint8([r, g, b])

NO_COLOR_ADJUSTMENT = rgb_adjust(NO_CORRECTION, NO_CORRECTION, NO_CORRECTION)

DEFAULT_COLOR_ADJUSTMENT = rgb_adjust(GAMMA_CORRECTIONS.index(1.02), GAMMA_CORRECTIONS.index(0.81), GAMMA_CORRECTIONS.index(0.98))


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
print('S_WH_RATIO', S_WH_RATIO, S_WIDTH, S_HEIGHT, S_HALF, S_CUT)

Strand = collections.namedtuple('Strand',
    [
        'index',
        'begin',
        'end',
        'length',
        'slice',
        'color_adjustment',
        'distance',
        'inverse_distance'
    ]
)


def strands(distance_function, end_point_list):

    result = []

    for end_points in end_point_list:

        index = len(result)

        begin, end, color_adjustment = end_points

        if begin < end:
            length = end - begin + 1
            slice_ = slice(begin, end + 1, 1)
        else:
            length = begin - end + 1
            slice_ = slice(begin, end - 1, -1)

        distance = distance_function(index, length)
        inverse_distance = 1.0 - distance

        result.append(
            Strand(
                index,
                begin,
                end,
                length,
                slice_,
                color_adjustment.copy(),
                distance,
                inverse_distance
            )
        )

    return result

def make_s_dist(index, length):
    y = float(index + (S_HALF - S_COUNT)) * 1.025
    half = int((length - 1) / 2)
    result = np.float16([math.hypot(x, y) / (S_HALF - 1) for x in range(-half, half + 1)])
    return result

def make_r_dist(index, length):
    return np.float16([ i / (length - 1) for i in range(0, length) ])

S_STRANDS = strands(make_s_dist,
    [
        (384, 406, DEFAULT_COLOR_ADJUSTMENT),    # 0
        (192, 214, DEFAULT_COLOR_ADJUSTMENT),    # 1
        (256, 276, DEFAULT_COLOR_ADJUSTMENT),    # 2
        (724, 704, DEFAULT_COLOR_ADJUSTMENT),    # 3
        (658, 640, DEFAULT_COLOR_ADJUSTMENT),    # 4
        (448, 464, DEFAULT_COLOR_ADJUSTMENT),    # 5
        (479, 465, DEFAULT_COLOR_ADJUSTMENT),    # 6
        (659, 671, DEFAULT_COLOR_ADJUSTMENT),    # 7
        (223, 215, NO_COLOR_ADJUSTMENT),         # 8
        (725, 729, DEFAULT_COLOR_ADJUSTMENT)     # 9
    ]
)

R_STRANDS = strands(make_r_dist,
    [
        (128, 159, NO_COLOR_ADJUSTMENT),       # 0 (L)
        ( 64,  95, NO_COLOR_ADJUSTMENT),       # 1
        (320, 351, DEFAULT_COLOR_ADJUSTMENT),  # 2
        (960, 991, NO_COLOR_ADJUSTMENT),       # 3
        (  0,  31, NO_COLOR_ADJUSTMENT),       # 4
        (224, 255, NO_COLOR_ADJUSTMENT),       # 5
        (576, 607, NO_COLOR_ADJUSTMENT),       # 6
        (480, 511, NO_COLOR_ADJUSTMENT),       # 7
        (768, 799, NO_COLOR_ADJUSTMENT),       # 8
        (896, 927, DEFAULT_COLOR_ADJUSTMENT),  # 9
        (832, 863, NO_COLOR_ADJUSTMENT),       # 10
        (512, 543, NO_COLOR_ADJUSTMENT),       # 11
        (730, 761, NO_COLOR_ADJUSTMENT),       # 12
        (672, 703, NO_COLOR_ADJUSTMENT),       # 13
        (277, 308, DEFAULT_COLOR_ADJUSTMENT),  # 14
        (407, 438, DEFAULT_COLOR_ADJUSTMENT)   # 15 (R)
    ]
)

#print('R_STRANDS\n ', '\n  '.join([str(strand) for strand in R_STRANDS]))
#print('S_STRANDS\n ', '\n  '.join([str(strand) for strand in S_STRANDS]))

ALL_STRANDS = []
ALL_STRANDS.extend(R_STRANDS)
ALL_STRANDS.extend(S_STRANDS)

COLOR_ADJUSTED_STRANDS = [ strand for strand in ALL_STRANDS if not np.all(strand.color_adjustment == 1.0) ]

PIXEL_DIST = np.zeros(NUM_PIXELS, dtype=np.float16)
PIXEL_DIST_INVERSE = np.zeros(NUM_PIXELS, dtype=np.float16)
for strand in ALL_STRANDS:
    PIXEL_DIST[strand.slice] = strand.distance
    PIXEL_DIST_INVERSE[strand.slice] = strand.inverse_distance


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
MAX_COLOR_SUM = (255 - MIN_COLOR_SUM) * 3
MIN_COLOR_DELTA = 16

def random_color(other_than = None):
    for i in range(1,10):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        s = r + g + b
        if s >= MIN_COLOR_SUM and s <= MAX_COLOR_SUM:
            if other_than is not None:
                count = 0
                count += 1 if abs(other_than[R] - r) > MIN_COLOR_DELTA else 0
                count += 2 if abs(other_than[G] - g) > MIN_COLOR_DELTA else 0
                count += 3 if abs(other_than[B] - b) > MIN_COLOR_DELTA else 0
                if count >= 2:
                    break
                print('rejected count', count, r, g, b, other_than)
            else:
                break
        print('rejected sum', s, r, g, b)
    return rgb(r, g, b)


class DisplayBase(QtCore.QObject):

    def __init__(self):
        super(DisplayBase, self).__init__()


def identity(x):
    return x


class ConfiguredDisplay(DisplayBase):

    PICK_NEW_DISPLAY_DELTA = 3

    MODE_SHOW_NORMAL      = 0
    MODE_NORMAL_TO_RANDOM = 1
    MODE_SHOW_RANDOM      = 2
    MODE_RANDOM_TO_NORMAL = 3
    MODE_RANDOM_TO_RANDOM = 4

    MODE_DURATION = {
        MODE_SHOW_NORMAL:      5,
        MODE_NORMAL_TO_RANDOM: 10,
        MODE_SHOW_RANDOM:      2,
        MODE_RANDOM_TO_NORMAL: 5,
        MODE_RANDOM_TO_RANDOM: 2
    }


    def __init__(self):
        super(ConfiguredDisplay, self).__init__()

        self.__mode_change_time = 0
        self.__mode_start_time = 0
        self.__mode = self.MODE_SHOW_NORMAL

        self.__normal_sun_pixels = self.__make_sun_pixels(center_color = rgb(255, 223, 147), edge_color = rgb(255, 180, 0))
        self.__from_pixels = self.__normal_sun_pixels
        self.__make_random_sun_pixels()

        self.__trans_choices = [
            (self.trans_blend, self.trans_blend_prepare),
            (self.trans_push, self.trans_push_prepare),
            (self.trans_pull, self.trans_pull_prepare),
            (self.trans_noise, self.trans_noise_prepare)
        ]
        self.__trans = self.__trans_choices[0][0]
        self.__trans_prepare = self.__trans_choices[0][1]


    def __make_random_sun_pixels(self):
        self.__center_color = random_color()
        self.__edge_color = random_color(other_than = self.__center_color)
        self.__make_next_sun_pixels()


    def __make_next_sun_pixels(self):
        self.__to_pixels = self.__make_sun_pixels(self.__center_color, self.__edge_color)
        self.__pick_next_trans()
        self.__trans_prepare()


    def __make_sun_pixels(self, center_color, edge_color):

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

        return pixels


    def __set_mode(self, frame_time, mode):
        self.__mode = mode
        self.__mode_change_time = frame_time.current + self.MODE_DURATION[mode]
        self.__mode_start_time = frame_time.current


    def update(self, frame_time, pixels, lepton_data):

        if self.__mode == self.MODE_SHOW_NORMAL:

            pixels[:] = self.__normal_sun_pixels

            if frame_time.current >= self.__mode_change_time:
                self.__set_mode(frame_time, self.MODE_NORMAL_TO_RANDOM)

        elif self.__mode == self.MODE_SHOW_RANDOM:

            pixels[:] = self.__from_pixels

            if frame_time.current >= self.__mode_change_time:
                self.__set_mode(frame_time, self.__next_mode)

        else:

            percent = (frame_time.current - self.__mode_start_time) / (self.__mode_change_time - self.__mode_start_time)

            self.__trans[self.__trans_type](percent, pixels)

            if frame_time.current >= self.__mode_change_time:
                self.__from_pixels = self.__to_pixels
                if self.__mode == self.MODE_NORMAL_TO_RANDOM or self.__mode == self.MODE_RANDOM_TO_RANDOM:
                    self.__set_mode(frame_time, self.MODE_SHOW_RANDOM)
                    self.__pick_next_mode()
                    self.__pick_next_trans()
                else:
                    self.__set_mode(frame_time, self.MODE_SHOW_NORMAL)
                    threading.Thread(target=self.__make_random_sun_pixels).start()


    def pick_next_mode(self):

        roll = random.randint(0, 99)
        if roll < 25:   #  0 - 24

            print('normal')

            self.__next_mode = self.MODE_RANDOM_TO_NORMAL
            self.__to_pixels = self.__normal_sun_pixels
            self.__pick_next_trans()

        elif roll < 50: # 25 - 49

            print('random')

            self.__next_mode = self.MODE_RANDOM_TO_RANDOM
            threading.Thread(target=self.__make_random_sun_pixels).start()

        elif roll < 75: # 50 - 74

            print('pulling')

            self.__next_mode = self.MODE_RANDOM_TO_RANDOM

            self.__center_color = self.__edge_color
            self.__edge_color = random_color(other_than = self.__center_color)
            self.__next_mode = self.MODE_RANDOM_TO_RANDOM

            threading.Thread(target=self.__make_next_sun_pixels).start()

        else:          # 75 - 99

            print('pushing')

            self.__next_mode = self.MODE_RANDOM_TO_RANDOM

            self.__edge_color = self.__center_color
            self.__center_color = random_color(other_than = self.__edge_color)
            self.__next_mode = self.MODE_RANDOM_TO_RANDOM

            threading.Thread(target=self.__make_next_sun_pixels).start()


    def pick_next_trans(self):
        choice = random.choice(self.__trans_choices)
        self.__trans = choice[0]
        self.__trans_prepare = choice[1]


    def diff_pixels(self):
        self.__pixel_diffs = np.int16(self.__to_pixels) - self.__from_pixels


    def trans_blend_prepare(self):
        self.diff_pixels()


    def trans_blend(self, percent, pixels):
        np.clip(
            self.__from_pixels + (self.__pixel_diffs * percent),
            0,
            255,
            pixels
        )


    def trans_push_prepare(self):
        pass


    def trans_push(self, percent, pixels):



        if percent < 33:
            # push s
        else:
            # push r


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


    def update(self, frame_time, pixels, lepton_data):
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

        print('__set_adjust', target_strand.color_adjustment, index, correction)

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

    def update(self, frame_time, pixels, lepton_data):

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

    def update(self, frame_time, pixels, lepton_data):

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
        if display_delta_time >= 10:
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
                cv2.normalize(lepton_frame, lepton_frame, 0, 255, cv2.NORM_MINMAX)
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
        self.__color_adjustment_enabled = True

    @QtCore.pyqtSlot()
    def stop(self):
        print('update thread stop requested')
        self.__exiting = True

    @QtCore.pyqtSlot()
    def set_display(self, display):
        self.__display = display

    @QtCore.pyqtSlot()
    def set_lepton_frame(self, lepton_frame):
        self.__lepton_frame = lepton_frame

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

            self.__display.update(frame_time, pixels, self.__lepton_frame)

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

        self.__configured_display = ConfiguredDisplay()

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

        # config

        self.loadButton.clicked.connect(self.__load_config)
        self.saveButton.clicked.connect(self.__save_config)
        self.editColorButton.clicked.connect(self.__edit_color)
        self.pathEdit.setText('/home/pi/pyhugm/config.yaml')
        #self.__load_config()

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
        self.__update_thread.set_lepton_frame(lepton_frame)

    def __perform_ffc(self):
        subprocess.call(['/home/pi/LeptonModule-master/software/flir_ffc/flir_ffc'])

    def __display_image(self, lepton_frame):
        rgb_data = np.uint8(cv2.cvtColor(lepton_frame, cv2.COLOR_GRAY2RGB))
        rows, columns, channel = rgb_data.shape
        bytesPerLine = 3 * columns
        image = QtGui.QImage(rgb_data.data, columns, rows, bytesPerLine, QtGui.QImage.Format_RGB888)
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
        print('__on_adjustment_changed', adjustment)
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
