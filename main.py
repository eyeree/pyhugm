from __future__ import print_function
from __future__ import division

# python
import collections
import itertools
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


Strand = collections.namedtuple('Strand', ['begin', 'end', 'length', 'slice'])


def strands(end_point_list):
    result = []
    for end_points in end_point_list:
        if end_points[0] < end_points[1]:
            result.append(
                Strand(
                    end_points[0], end_points[1],
                    (end_points[1] - end_points[0]) + 1,
                    slice(end_points[0], end_points[1] + 1, 1)
                )
            )
        else:
            result.append(
                Strand(
                    end_points[0], end_points[1],
                    (end_points[0] - end_points[1]) + 1,
                    slice(end_points[0], end_points[1] - 1, -1)
                )
            )
    return result


S_STRANDS = strands(
    [
        (192, 210),    # 0
        (229, 211),    # 1
        (230, 246),    # 2
        (320, 336),    # 3
        (351, 337),    # 4
        (352, 366),    # 5
        (256, 268),    # 6
        (279, 269),    # 7
        (280, 286)     # 8
    ]
)

R_STRANDS = strands(
    [
        (   0,    1), #  0 (L)
        (   0,    1), #  1
        (   0,    1), #  2
        (   0,    1), #  3
        (   0,    1), #  4
        (   0,    1), #  5
        (   0,    1), #  6
        (   0,    1), #  7
        (384, 415),  # 08
        (448, 479),  # 09
        (287, 318),  # 10
        (832, 863),  # 11
        (   0,    1), # 12
        (   0,    1), # 13
        (   0,    1), # 14
        (   0,    1)  # 15 (R)
    ]
)

print('R_STRANDS\n  ', '\n  '.join([str(strand) for strand in R_STRANDS]))
print('S_STRANDS\n  ', '\n  '.join([str(strand) for strand in S_STRANDS]))


class DisplayAllOn(DisplayBase):

    def __init__(self):
        super(DisplayAllOn, self).__init__()

    def update(self, frame_time, pixels, lepton_data):
        pixels[:] = itertools.repeat((128, 128, 128), len(pixels))


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
        pixels[r.slice] = itertools.repeat((123, 31, 173), r.length)
        pixels[r.begin] = (244, 152, 66)
        pixels[r.end] = (35, 234, 21)

        s = S_STRANDS[self.__s_index]
        pixels[s.slice] = itertools.repeat((123, 31, 173), s.length)
        pixels[s.begin] = (244, 152, 66)
        pixels[s.end] = (35, 234, 21)


class DisplayChase(DisplayBase):

    MIN_CPS = 1.0
    MAX_CPS = 100.0

    def __init__(self, initial_speed):
        super(DisplayChase, self).__init__()

        self.__s_index = 0
        self.__r_index = 0
        self.__next_time = 0
        self.__skipped = 0

        self.set_speed(initial_speed)

    def set_speed(self, speed):
        self.__delta_time = 1.0 / np.interp(speed, [0, 9], [self.MIN_CPS, self.MAX_CPS])
        print('delta time', self.__delta_time)

    def update(self, frame_time, pixels, lepton_data):

        if frame_time.current >= self.__next_time:

            self.__next_time = frame_time.current + self.__delta_time

            self.__r_index += 1
            if self.__r_index == len(R_STRANDS):
                self.__r_index = 0

            self.__s_index += 1
            if self.__s_index == len(S_STRANDS):
                self.__s_index = 0

            print('skipped', self.__skipped)
            self.__skipped = 0

        else:

            self.__skipped += 1

        r = R_STRANDS[self.__r_index]
        pixels[r.slice] = itertools.repeat((123, 31, 173), r.length)
        pixels[r.begin] = (244, 152, 66)
        pixels[r.end] = (35, 234, 21)

        s = S_STRANDS[self.__s_index]
        pixels[s.slice] = itertools.repeat((123, 31, 173), s.length)
        pixels[s.begin] = (244, 152, 66)
        pixels[s.end] = (35, 234, 21)


class DisplayPixel(DisplayBase):

    def __init__(self, initial_index):
        super(DisplayPixel, self).__init__()
        self.__index = initial_index

    def set_index(self, index):
        self.__index = index

    def update(self, frame_time, pixels, lepton_data):
        if self.__index > 0:
            pixels[self.__index - 1] = (128, 0, 0)
        pixels[self.__index] = (255, 255, 255)
        if self.__index < 1023:
            pixels[self.__index + 1] = (0, 0, 128)



class FrameTime(object):

    def __init__(self):
        self.__last_time = time.clock()
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

        current_time = time.clock()
        self.__last_delta = current_time - self.__last_time
        self.__last_time = current_time

        self.__frame_count += 1
        display_delta_time = current_time - self.__last_display_time
        if display_delta_time >= 10:
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


class UpdateThread(QtCore.QThread):

    NUM_PIXELS = 1024
    OPC_ADDRESS = 'localhost:7890'
    LEPTON_DEVICE = "/dev/spidev0.1"

    image_captured = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent, initial_display):
        QtCore.QThread.__init__(self, parent)

        self.__exiting = False
        self.__display = initial_display
        self.__opc_client = opc.Client(self.OPC_ADDRESS)

        parent_conn, child_conn = multiprocessing.Pipe()
        self.__lepton_process = multiprocessing.Process(target=lepton_process, args=(child_conn, self.LEPTON_DEVICE))
        self.__lepton_process.start()
        self.__connection = parent_conn

    @QtCore.pyqtSlot()
    def stop(self):
        self.__connection.send({'stop': True})
        self.__connection.close()

    @QtCore.pyqtSlot()
    def set_display(self, display):
        self.__display = display

    def __produce_image(self, lepton_frame):
        rgb_data = np.uint8(cv2.cvtColor(lepton_frame, cv2.COLOR_GRAY2RGB))
        rows, columns, channel = rgb_data.shape
        bytesPerLine = 3 * columns
        image = QtGui.QImage(rgb_data.data, columns, rows, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.image_captured.emit(image)

    def run(self):

        print('update thread running')

        frame_time = FrameTime()

        while True:

            frame_time.tick()

            try:
                lepton_frame = self.__connection.recv()
            except IOError:
                break

            cv2.normalize(lepton_frame, lepton_frame, 0, 255, cv2.NORM_MINMAX)

            self.__produce_image(lepton_frame)

            pixels = [(0, 0, 0)] * self.NUM_PIXELS
            self.__display.update(frame_time, pixels, lepton_frame)
            self.__opc_client.put_pixels(pixels)

        print('update thread exiting')
        self.__lepton_process.join()
        print('lepton thread exited')


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
        self.__display_chase = DisplayChase(self.chaseSpeedSpinBox.value())
        self.__display_pixel = DisplayPixel(self.pixelIndexSpinBox.value())
        self.__display_r_s = DisplayRS(self.rSpinBox.value(), self.sSpinBox.value())

        # update thread
        self.__update_thread = UpdateThread(self, self.__display_all_off)
        self.__update_thread.image_captured.connect(self.__display_image)
        self.__update_thread.start()

        print('main window started')

    def closeEvent(self, event):
        print('main window stopping')
        self.__update_thread.stop()
        self.__update_thread.wait()
        print('main window stopped')

    def __perform_ffc(self):
        subprocess.call(['/home/pi/LeptonModule-master/software/flir_ffc/flir_ffc'])

    def __display_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)

    def __pixel_index_changed(self, index):
        self.__display_pixel.set_index(index)
        self.__update_thread.set_display(self.__display_pixel)

    def __r_index_changed(self, index):
        self.__display_r_s.set_r_index(index)
        self.__update_thread.set_display(self.__display_r_s)

    def __s_index_changed(self, index):
        self.__display_r_s.set_s_index(index)
        self.__update_thread.set_display(self.__display_r_s)

    def __chase_speed_changed(self, speed):
        self.__display_chase.set_speed(speed)
        self.__update_thread.set_display(self.__display_chase)

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
