class Test(object):

    def __init__(self):
        pass

    def x(self, a, b = 10):
        print a, b
        return 'foo'


t = Test()

print t.x(1)
