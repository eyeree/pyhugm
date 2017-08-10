from __future__ import division

for d in range(0,100, 5):
    print '{0:3}| {1}'.format(d, ' '.join([ '{0:6}'.format(((d - p) + 100) / 2) for p in range(0, 100, 5) ]) )

