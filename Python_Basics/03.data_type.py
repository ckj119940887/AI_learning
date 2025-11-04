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