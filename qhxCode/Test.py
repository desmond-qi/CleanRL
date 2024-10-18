class A:
    def __init__(self, a: float = 0, b: float = 0):
        self.a = a
        self.b = b
        print(f"A initialized with a = {self.a}")

class B(A):
    def __init__(self, a: float = 0, b: float = 0):  # 给a赋默认值
        super().__init__(a, b)  # 将a传递给A

class C(B):
    metadata = {
        "b": [0, 1, 2, 3],
    }
    def __init__(self, a):  # 可以在C中也给a赋默认值
        super().__init__()  # 调用B的__init__，并传递a

# 创建C类实例并传入a=1
c = C(a=2, b=1)  # 输出: A initialized with a = 1
