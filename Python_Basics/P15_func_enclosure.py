# 闭包
# 1.外部函数定义一个内部函数
# 2.内部函数使用外部函数的变量
# 3.外部函数将内部函数作为返回值

# Function B can Function A's variable
def func_a():
    num1 = 10
    def func_b():
        print(num1)
    return func_b # no parenthesis

fb = func_a()
print(fb)
fb()

print("test closure")
# 拿到内部函数所引用的外部变量，通过__closure__
for e in fb.__closure__:
    print(e.cell_contents)
