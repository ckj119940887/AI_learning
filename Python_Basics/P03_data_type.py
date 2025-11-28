# 不可变类型：数值，字符串，元组
# 可变类型：列表，集合，字典

# type 不会认为子类是一种父类型
# isinstance 认为子类是一种父类型

n = 1
b = True

print(type(n))
print(type(b))
# False, 不认为子类是一种父类型
print(type(n) == type(b))

print(isinstance(n, int))
# bool inherit from int
print(isinstance(b, int))

# id() 获取内存地址
num1 = 300
num2 = 300
print(id(num1) == id(num2))

# 浮点数
f1 = 0.10009
f2 = 0.2
f3 = f1 + f2
print(type(f3))
print(f3)

# 处理精度问题
from decimal import Decimal
t1 = Decimal(0.1)
t2 = Decimal(0.2)
print(t1 + t2)

# bool
num1 = True
print(num1 == 1)
# 'is' is used to check whether the address is same
print(num1 is True)

# string
# ' '
# " "
# """ """

r1 = 4
r2 = 1
# here is not int data
r3 = r1 / r2
print(r3)
print(type(r3))