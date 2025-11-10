def cal(num1, num2, op):
    return op(num1, num2)

print(cal(1,2, lambda num1, num2: num1 + num2))
