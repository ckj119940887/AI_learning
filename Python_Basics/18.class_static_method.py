class MathUtil:
    # 可以不用实例化对象直接使用该函数
    @staticmethod
    def add(a, b):
        return a + b


print(MathUtil.add(1, 2))