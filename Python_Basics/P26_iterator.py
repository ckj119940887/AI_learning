# 迭代器只能从前向后遍历
for item in (1,2,3):
    print(item)

# 判断对象是否是可迭代的
from collections.abc import Iterable, Iterator

print(isinstance([], Iterable))

# 迭代器有两个方法 iter() next()
it = iter([1,2,3])
print(next(it))

# 自定义迭代器
class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        else:
            self.index -= 1
            return self.data[self.index]

rev = Reverse([1,2,3])
print(isinstance(rev, Iterable))
print(isinstance(rev, Iterator))
print(next(rev))
print(next(rev))
