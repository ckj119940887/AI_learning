def cal(num1, num2, op):
    return op(num1, num2)

print(cal(1,2, lambda num1, num2: num1 + num2))

def my_map(list, func):
    for i, item in enumerate(list):
        list[i] = func(item)
    return list

list1 = list(range(10))
print(my_map(list1, lambda x: x ** 2))