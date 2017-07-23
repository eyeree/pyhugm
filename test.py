from __future__ import division
from __future__ import print_function

import time

for i in range(3):
    start_time = time.time()
    start_clock = time.clock()
    time.sleep(2)
    stop_time = time.time()
    stop_clock = time.clock()
    print('clock', start_clock, stop_clock, stop_clock - start_clock)
    print('time', start_time, stop_time, stop_time - start_time)








