class Man:
    def __init__(self, name):
        self.name = name

    def hello(self):
        print('Hello ' + self.name)


m = Man('asd')

m.hello()
