from __future__ import print_function
from __future__ import division

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


class DisplayBase(object):

    def __init__(self):
        pass


Strand = collections.namedtuple('Strand', ['index', 'begin', 'end', 'length', 'slice'])


def strands(end_point_list):
    result = []
    for end_points in end_point_list:
        if end_points[0] < end_points[1]:
            result.append(
                Strand(
                    len(result),
                    end_points[0], end_points[1],
                    (end_points[1] - end_points[0]) + 1,
                    slice(end_points[0], end_points[1] + 1, 1)
                )
            )
        else:
            result.append(
                Strand(
                    len(result),
                    end_points[0], end_points[1],
                    (end_points[0] - end_points[1]) + 1,
                    slice(end_points[0], end_points[1] - 1, -1)
                )
            )
    return result


S_STRANDS = strands(
    [
        (384, 406),    # 0
        (192, 214),    # 1
        (256, 276),    # 2
        (724, 704),    # 3
        (658, 640),    # 4
        (448, 464),    # 5
        (479, 465),    # 6
        (659, 671),    # 7
        (223, 215),    # 8
        (725, 729)     # 9
    ]
)

R_STRANDS = strands(
    [
        (128, 159),  # 0 (L)
        (64, 95),    # 1
        (320, 351),  # 2
        (960, 991),  # 3
        (0, 31),     # 4
        (224, 255),  # 5
        (576, 607),  # 6
        (480, 511),  # 7
        (768, 799),  # 8
        (896, 927),  # 9
        (832, 863),  # 10
        (512, 543),  # 11
        (730, 761),  # 12
        (672, 703),  # 13
        (277, 308),  # 14
        (407, 438)   # 15 (R)
    ]
)

print('R_STRANDS\n ', '\n  '.join([str(strand) for strand in R_STRANDS]))
print('S_STRANDS\n ', '\n  '.join([str(strand) for strand in S_STRANDS]))

S_COLOR_ADJUST = [
    (0, 0, 0),  # 0
    (0, 0, 0),  # 1
    (0, 0, 0),  # 2
    (0, 0, 0),  # 3
    (0, 0, 0),  # 4
    (0, 0, 0),  # 5
    (0, 0, 0),  # 6
    (0, 0, 0),  # 7
    (0, 0, 0),  # 8
    (0, 0, 0)   # 9
]

R_COLOR_ADJUST = [
    (0, 0, 0),  # 0
    (0, 0, 0),  # 1
    (0, 0, 0),  # 2
    (0, 0, 0),  # 3
    (0, 0, 0),  # 4
    (0, 0, 0),  # 5
    (0, 0, 0),  # 6
    (0, 0, 0),  # 7
    (0, 0, 0),  # 8
    (0, 0, 0),  # 9
    (0, 0, 0),  # 10
    (0, 0, 0),  # 11
    (0, 0, 0),  # 12
    (0, 0, 0),  # 13
    (0, 0, 0),  # 14
    (0, 0, 0)   # 15
]


class DisplayAllOn(DisplayBase):

    def __init__(self):
        super(DisplayAllOn, self).__init__()

    def update(self, frame_time, pixels, lepton_data):
        pixels.fill(128)


class DisplayAllOff(DisplayBase):

    def __init__(self):
        super(DisplayAllOff, self).__init__()

    def update(self, frame_time, pixels, lepton_data):
        pass


class DisplayRS(DisplayBase):

    def __init__(self, initial_r_index, initial_s_index):
        super(DisplayRS, self).__init__()
        self.__r_index = initial_r_index
        self.__s_index = initial_s_index

    def set_r_index(self, r_index):
        self.__r_index = r_index

    def set_s_index(self, s_index):
        self.__s_index = s_index

    def update(self, frame_time, pixels, lepton_data):

        r = R_STRANDS[self.__r_index]
        pixels[r.slice] = [123, 31, 173]
        pixels[r.begin] = [244, 152, 66]
        pixels[r.end] = [35, 234, 21]

        s = S_STRANDS[self.__s_index]
        pixels[s.slice] = [123, 31, 173]
        pixels[s.begin] = [244, 152, 66]
        pixels[s.end] = [35, 234, 21]


class DisplayPixel(DisplayBase):

    def __init__(self, initial_index):
        super(DisplayPixel, self).__init__()
        self.__index = initial_index

    def set_index(self, index):
        self.__index = index

    def update(self, frame_time, pixels, lepton_data):
        if self.__index > 0:
            pixels[self.__index - 1] = [128, 0, 0]
        pixels[self.__index] = [255, 255, 255]
        if self.__index < 1023:
            pixels[self.__index + 1] = [0, 0, 128]


def interp_rgb(x, indexes, colors):
    r = np.interp(x, indexes, [c[0] for c in colors])
    g = np.interp(x, indexes, [c[1] for c in colors])
    b = np.interp(x, indexes, [c[2] for c in colors])
    return [r, g, b]


def iter_rgb(count, colors):
    indexes = np.linspace(0, count - 1, len(colors))
    return [interp_rgb(x, indexes, colors) for x in range(0, count)]


MAX_S_HALF = 11


def iter_rgb_s_dist(s, colors):

    indexes = np.linspace(-MAX_S_HALF, MAX_S_HALF, len(colors))

    y = s.index + 2  # offset for rows one less than MAX_S_HALF

    half = int((s.length - 1) / 2)

    return [interp_rgb(math.hypot(x, y), indexes, colors) for x in range(-half, half + 1)]


def gamma_adjust_rgb(rgb, adjustment):
    return rgb * adjustment
    #r = int(rgb[0] * adjustment)
    #g = int(rgb[1] * adjustment)
    #b = int(rgb[2] * adjustment)
    #return [r, g, b]


class DisplaySun(DisplayBase):

    S_CENTER_COLOR = [255, 223, 147]
    S_HALF_COLOR = [252, 217, 133]
    S_EDGE_COLOR = [255, 180, 0]
    S_COLORS = [S_CENTER_COLOR, S_HALF_COLOR, S_HALF_COLOR, S_HALF_COLOR, S_EDGE_COLOR]

    R_START_COLOR = [255, 180, 0]
    R_END_COLOR = [0, 0, 0]
    R_COLORS = [R_START_COLOR, R_END_COLOR]

    ALIAS_GAMMA_ADJUSTMENT = 0.5

    def __init__(self, initial_speed):
        super(DisplaySun, self).__init__()
        self.__frame_number = 0
        self.__pixels = np.zeros([UpdateThread.NUM_PIXELS, 3], dtype=np.uint8)
        self.__speed = initial_speed

        for r in R_STRANDS:
            self.__pixels[r.slice] = iter_rgb(r.length, self.R_COLORS)

        for s in S_STRANDS:
            self.__pixels[s.slice] = iter_rgb_s_dist(s, self.S_COLORS)

        for i in [1, 3, 5]:
            s = S_STRANDS[i]
            self.__pixels[s.begin] = gamma_adjust_rgb(self.__pixels[s.begin], self.ALIAS_GAMMA_ADJUSTMENT)
            self.__pixels[s.end] = gamma_adjust_rgb(self.__pixels[s.end], self.ALIAS_GAMMA_ADJUSTMENT)

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

    @property
    def current(self):
        return self.__last_time

    @property
    def delta(self):
        return self.__last_delta

    def tick(self):

        time.sleep(max(self.SLEEP_TIME - (time.time() - self.__last_time), 0.0))

        current_time = time.time()

        self.__last_delta = current_time - self.__last_time
        self.__last_time = current_time

        self.__frame_count += 1
        display_delta_time = current_time - self.__last_display_time
        if display_delta_time >= 5:
            print('update fps', self.__frame_count / display_delta_time)
            self.__frame_count = 0
            self.__last_display_time = current_time


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

    NUM_PIXELS = 1024
    OPC_ADDRESS = 'localhost:7890'
    OPC_HEADER_SIZE = 4

    def __init__(self, parent, initial_display):
        QtCore.QThread.__init__(self, parent)
        self.__exiting = False
        self.__display = initial_display
        self.__opc_client = opc.Client(self.OPC_ADDRESS)
        self.__lepton_frame = None

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

    def run(self):

        print('update thread running')

        blen = self.NUM_PIXELS * 3
        len_hi = int(blen / 256)
        len_lo = blen % 256

        opc_buffer = np.empty([(self.NUM_PIXELS * 3) + self.OPC_HEADER_SIZE], dtype=np.uint8)
        opc_buffer[0] = 0                               # channel
        opc_buffer[1] = 0                               # command = set color
        opc_buffer[2] = len_hi  # len hi byte
        opc_buffer[3] = len_lo  # len lo byte

        pixels = opc_buffer[self.OPC_HEADER_SIZE:].reshape([self.NUM_PIXELS, 3])

        frame_time = FrameTime()
        while not self.__exiting:
            frame_time.tick()
            pixels.fill(0)
            self.__display.update(frame_time, pixels, self.__lepton_frame)
            self.__opc_client.put_message(opc_buffer.tostring())

        print('update thread exited')


class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):

    def __init__(self):

        super(self.__class__, self).__init__()

        print('main window starting')

        self.setupUi(self)

        # pixel index
        self.pixelIndexSpinBox.valueChanged.connect(self.__pixel_index_changed)
        self.pixelIndexStepSpinBox.valueChanged.connect(lambda value: self.pixelIndexSpinBox.setSingleStep(value))

        # s/r index
        self.sSpinBox.valueChanged.connect(self.__s_index_changed)
        self.rSpinBox.valueChanged.connect(self.__r_index_changed)

        # chase speed
        self.chaseSpeedSpinBox.valueChanged.connect(self.__chase_speed_changed)

        # buttons
        self.performFFCButton.clicked.connect(self.__perform_ffc)
        self.allOnPushButton.clicked.connect(self.__all_on_clicked)
        self.allOffPushButton.clicked.connect(self.__all_off_clicked)

        # displays
        self.__display_all_off = DisplayAllOff()
        self.__display_all_on = DisplayAllOn()
        self.__display_pixel = DisplayPixel(self.pixelIndexSpinBox.value())
        self.__display_r_s = DisplayRS(self.rSpinBox.value(), self.sSpinBox.value())
        self.__display_sun = DisplaySun(self.chaseSpeedSpinBox.value())

        # update thread
        self.__update_thread = UpdateThread(self, self.__display_sun)
        self.__update_thread.start()

        # lepton thread
        self.__lepton_thread = LeptonThread(self)
        self.__lepton_thread.lepton_frame_captured.connect(self.__lepton_frame_captured)
        self.__lepton_thread.start()

        # rgb / color adjust
        self.redSpinBox.valueChanged.connect(self.__color_changed)
        self.greenSpinBox.valueChanged.connect(self.__color_changed)
        self.blueSpinBox.valueChanged.connect(self.__color_changed)
        self.__color_adjust_s_index = None
        self.__color_adjust_r_index = None

        print('main window started')

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

    def __pixel_index_changed(self, index):
        self.__display_pixel.set_index(index)
        self.__update_thread.set_display(self.__display_pixel)

    def __r_index_changed(self, index):
        #self.__display_r_s.set_r_index(index)
        #self.__update_thread.set_display(self.__display_r_s)
        self.__set_color_adjust_r_index(index)

    def __s_index_changed(self, index):
        #self.__display_r_s.set_s_index(index)
        #self.__update_thread.set_display(self.__display_r_s)
        self.__set_color_adjust_s_index(index)

    def set_color_adjust_s_index(self, index):
        self.__color_adjust_r_index = None
        self.__color_adjust_s_index = index
        self.__show_color(S_COLOR_ADJUST[self.__color_adjust_s_index])

    def set_color_adjust_r_index(self, index):
        self.__color_adjust_s_index = None
        self.__color_adjust_r_index = index
        self.__show_color(R_COLOR_ADJUST[self.__color_adjust_r_index])

    def __show_color(self, rgb):
        self.redSpinBox.setValue(R_COLOR_ADJUST)

    def __color_changed(self, ignored):
        rgb = (self.redSpinBox.value, self.greenSpinBox.value, self.blueSpinBox.value)
        if self.__color_adjust_r_index:
            R_COLOR_ADJUST[self.__color_adjust_r_index] = rgb
        else:
            S_COLOR_ADJUST[self.__color_adjust_r_index] = rgb

    def __chase_speed_changed(self, speed):
        self.__display_sun.set_speed(speed)
        self.__update_thread.set_display(self.__display_sun)

    def __all_on_clicked(self):
        self.__update_thread.set_display(self.__display_all_on)

    def __all_off_clicked(self):
        self.__update_thread.set_display(self.__display_all_off)


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
