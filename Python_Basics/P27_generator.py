# 创建生成器对象，使用元组推导式
generator = (x for x in range(10))
print(generator)
for x in generator:
    print(x)

# 创建生成器对象，使用函数
# 每用一次next,执行一次yield,会自动记录上次执行到到哪
def fib():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a+b

f = fib()
print(next(f))
