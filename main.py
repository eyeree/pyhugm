from __future__ import print_function
from __future__ import division

print('loading...')

# python
import collections
import math
import multiprocessing
import subprocess
import sys
import time

# fadecandy
import opc

# qt
from PyQt4 import QtGui, QtCore

# project
import ui

# lepton
import numpy as np
from pylepton import Lepton

# image processsing
import cv2


class DisplayBase(QtCore.QObject):

    def __init__(self):
        super(DisplayBase, self).__init__()


Strand = collections.namedtuple('Strand', ['index', 'begin', 'end', 'length', 'slice', 'color_adjustment'])


def strands(end_point_list):

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

        result.append(
            Strand(
                index,
                begin,
                end,
                length,
                slice_,
                color_adjustment.copy()
            )
        )

    return result

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
print('GAMMA_CORRECTIONS', GAMMA_CORRECTIONS)
NO_CORRECTION = GAMMA_CORRECTIONS.index(1.0)
GAMMA_LUT = [ gamma_lut(correction) for correction in GAMMA_CORRECTIONS ]
print('done', len(GAMMA_LUT))
print(GAMMA_LUT)

def rgb_adjust(r, g, b):
    return np.uint8([r, g, b])

NO_COLOR_ADJUSTMENT = rgb_adjust(NO_CORRECTION, NO_CORRECTION, NO_CORRECTION)

DEFAULT_COLOR_ADJUSTMENT = rgb_adjust(GAMMA_CORRECTIONS.index(1.02), GAMMA_CORRECTIONS.index(0.81), GAMMA_CORRECTIONS.index(0.98))

S_STRANDS = strands(
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

R_STRANDS = strands(
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

print('R_STRANDS\n ', '\n  '.join([str(strand) for strand in R_STRANDS]))
print('S_STRANDS\n ', '\n  '.join([str(strand) for strand in S_STRANDS]))

ALL_STRANDS = []
ALL_STRANDS.extend(R_STRANDS)
ALL_STRANDS.extend(S_STRANDS)

COLOR_ADJUSTED_STRANDS = [ strand for strand in ALL_STRANDS if not np.all(strand.color_adjustment == 1.0) ]

NUM_PIXELS = 1024
MAX_S_HALF = 11


def interp_rgb(x, indexes, colors):
    r = np.interp(x, indexes, [c[R] for c in colors])
    g = np.interp(x, indexes, [c[G] for c in colors])
    b = np.interp(x, indexes, [c[B] for c in colors])
    return rgb(r, g, b)


def iter_rgb(count, colors):
    indexes = np.linspace(0, count - 1, len(colors))
    return [interp_rgb(x, indexes, colors) for x in range(0, count)]


def iter_rgb_s_dist(s, colors):

    indexes = np.linspace(-MAX_S_HALF, MAX_S_HALF, len(colors))

    y = s.index + 2  # offset for rows one less than MAX_S_HALF

    half = int((s.length - 1) / 2)

    return [interp_rgb(math.hypot(x, y), indexes, colors) for x in range(-half, half + 1)]


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


class DisplaySun(DisplayBase):

    S_CENTER_COLOR = rgb(255, 223, 147)
    S_HALF_COLOR = rgb(252, 217, 133)
    S_EDGE_COLOR = rgb(255, 180, 0)
    S_COLORS = [S_CENTER_COLOR, S_HALF_COLOR, S_HALF_COLOR, S_HALF_COLOR, S_EDGE_COLOR]

    R_START_COLOR = rgb(255, 180, 0)
    R_END_COLOR = COLOR_BLACK
    R_COLORS = [R_START_COLOR, R_END_COLOR]

    def __init__(self, initial_speed):
        super(DisplaySun, self).__init__()
        self.__frame_number = 0
        self.__pixels = np.zeros([NUM_PIXELS, 3], dtype=np.uint8)
        self.__speed = initial_speed

        for r in R_STRANDS:
            self.__pixels[r.slice] = iter_rgb(r.length, self.R_COLORS)

        for s in S_STRANDS:
            self.__pixels[s.slice] = iter_rgb_s_dist(s, self.S_COLORS)

    def set_speed(self, speed):
        self.__speed = speed

    def update(self, frame_time, pixels, lepton_data):
        pixels[:] = self.__pixels

        for r in R_STRANDS:
            for i in range(int(self.__frame_number / 4), r.length, int(r.length / 4)):

                p = r.begin + (i - 1) if i > 0 else r.end
                pixels[p] *= 0.87

                p = r.begin + i
                pixels[p] *= 0.83

                p = r.begin + (i + 1) if i < 31 else r.begin
                pixels[p] *= 0.87

        self.__frame_number += self.__speed
        if self.__frame_number >= 32:
            self.__frame_number = 0


class FrameTime(object):

    SLEEP_TIME = 1.0 / 61.0

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

        # sun display

        self.__display_sun = DisplaySun(
            self.sunSpeedSpinBox.value())

        self.sunSpeedSpinBox.valueChanged.connect(
            lambda value: self.__display_sun.set_speed(value))

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
        self.__tool_box_displays = [self.__display_sun, self.__display_color, self.__display_index]
        self.toolBox.currentChanged.connect(
            lambda index: self.__update_thread.set_display(self.__tool_box_displays[index]))

        # update thread
        self.__update_thread = UpdateThread(self, self.__tool_box_displays[self.toolBox.currentIndex()])
        self.__update_thread.start()

        # lepton thread
        self.__lepton_thread = LeptonThread(self)
        self.__lepton_thread.lepton_frame_captured.connect(self.__lepton_frame_captured)
        self.__lepton_thread.start()

        print('main window started')

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
