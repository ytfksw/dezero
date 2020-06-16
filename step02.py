from step01 import Variable


class Function:
    def __call__(self, input_var):
        y = self.forward(input_var.data)
        return Variable(y)

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == "__main__":
    import numpy as np
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
