import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func


class Function:
    def __call__(self, input_var):
        x = input_var.data
        y = self.forward(x)
        output_var = Variable(y)
        output_var.set_creator(self)
        self.input_var = input_var
        self.output_var = output_var
        return output_var


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input_var.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input_var.data
        gx = np.exp(x) * gy
        return gx


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input_var == b
    assert y.creator.input_var.creator == B
    assert y.creator.input_var.creator.input_var == a
    assert y.creator.input_var.creator.input_var.creator == A
    assert y.creator.input_var.creator.input_var.creator.input_var == x
