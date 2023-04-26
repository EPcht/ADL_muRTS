from test2 import test2

class test1:
    def __init__(self):
        self.x = 10
        self.test = test2()

    def start(self):
        print(self.x)
        self.test.do(self)
        print(self.x)

test = test1()
test.start()