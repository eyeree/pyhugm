from __future__ import print_function

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


Strand = collections.namedtuple('Strand', ['lower', 'upper', 'length'])


def transform(t):
    if t[0] < t[1]:
        return Strand(t[0], t[1], (t[1] - t[0]) + 1)
    else:
        return Strand(t[1], t[0], (t[0] - t[1]) + 1)

S = [transform(t) for t in
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
]

R = [transform(t) for t in
    [
#        (   0,    0), #  0 (L)
#        (   0,    0), #  1
#        (   0,    0), #  2
#        (   0,    0), #  3
#        (   0,    0), #  4
#        (   0,    0), #  5
#        (   0,    0), #  6
#        (   0,    0), #  7
        (384, 415),    # 8
        (448, 479)     # 9
#        (   0,    0), # 10
#        (   0,    0), # 11
#        (   0,    0), # 12
#        (   0,    0), # 13
#        (   0,    0), # 14
#        (   0,    0)  # 15 (R)
    ]
]


class DisplayAllOn(DisplayBase):

    def __init__(self):
        super(DisplayAllOn, self).__init__()

    def update(self, time_delta, pixels, lepton_data):
        pixels[:] = itertools.repeat((128, 128, 128), len(pixels))


class DisplayAllOff(DisplayBase):

    def __init__(self):
        super(DisplayAllOff, self).__init__()

    def update(self, time_delta, pixels, lepton_data):
        pass


class DisplayChase(DisplayBase):

    def __init__(self):
        super(DisplayChase, self).__init__()
        self.__s_index = 0
        self.__r_index = 0

    def update(self, time_delta, pixels, lepton_data):

        r = R[self.__r_index]
        pixels[r.lower:r.upper] = itertools.repeat((123, 31, 173), r.length)

        s = S[self.__s_index]
        pixels[s.lower:s.upper] = itertools.repeat((123, 31, 173), s.length)

        self.__r_index += 1
        if self.__r_index == len(R):
            self.__r_index = 0

        self.__s_index += 1
        if self.__s_index == len(S):
            self.__s_index = 0


class DisplayIndex(DisplayBase):

    def __init__(self, initial_index):
        super(DisplayIndex, self).__init__()
        self.__index = initial_index

    def set_index(self, index):
        self.__index = index

    def update(self, time_delta, pixels, lepton_data):
        pixels[self.__index] = (255, 255, 255)


class FrameTime(object):

    def __init__(self):
        self.__last_time = time.clock()
        self.__last_display_time = self.__last_time
        self.__frame_count = 0

    def delta(self):

        current_time = time.clock()
        delta_time = current_time - self.__last_time
        self.__last_time = current_time

        self.__frame_count += 1
        display_delta_time = current_time - self.__last_display_time
        if display_delta_time >= 10:
            print('update fps', self.__frame_count / display_delta_time)
            self.__frame_count = 0
            self.__last_display_time = current_time

        return delta_time


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

            frame_time_delta = frame_time.delta()

            try:
                lepton_frame = self.__connection.recv()
            except IOError:
                break

            cv2.normalize(lepton_frame, lepton_frame, 0, 255, cv2.NORM_MINMAX)

            self.__produce_image(lepton_frame)

            pixels = [(0, 0, 0)] * self.NUM_PIXELS
            self.__display.update(frame_time_delta, pixels, lepton_frame)
            self.__opc_client.put_pixels(pixels)

        print('update thread exiting')
        self.__lepton_process.join()
        print('lepton thread exited')


class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):

    def __init__(self):

        super(self.__class__, self).__init__()

        print('main window starting')

        self.setupUi(self)

        # spin boxes
        self.indexSpinBox.valueChanged.connect(self.__index_changed)
        self.indexSpinBox.setMinimum(0)
        self.indexSpinBox.setMaximum(UpdateThread.NUM_PIXELS - 1)
        self.indexSpinBox.setWrapping(True)
        self.stepSpinBox.valueChanged.connect(lambda value: self.indexSpinBox.setSingleStep(value))

        # buttons
        self.performFFCButton.clicked.connect(self.__perform_ffc)
        self.allOnPushButton.clicked.connect(self.__all_on_clicked)
        self.allOffPushButton.clicked.connect(self.__all_off_clicked)
        self.chasePushButton.clicked.connect(self.__chase_clicked)

        # displays
        self.__display_all_off = DisplayAllOff()
        self.__display_all_on = DisplayAllOn()
        self.__display_chase = DisplayChase()
        self.__display_index = DisplayIndex(self.indexSpinBox.value())

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

    def __index_changed(self, index):
        self.__display_index.set_index(index)
        self.__update_thread.set_display(self.__display_index)

    def __all_on_clicked(self):
        self.__update_thread.set_display(self.__display_all_on)

    def __all_off_clicked(self):
        self.__update_thread.set_display(self.__display_all_off)

    def __chase_clicked(self):
        self.__update_thread.set_display(self.__display_chase)


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
