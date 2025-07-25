
class B:
    def __init_subclass__(cls):
        super().__init_subclass__()
        set(cls, 'b', 2)


class A(B):
    x = 1

