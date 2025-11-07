num1 = 10
num2 = 11
numbers = [10, 12, 14]
print(num1 in numbers)
print(num1 not in numbers)

print(num1 is num2)
print(num1 is not num2)

a = [1, 2, 3]
b = a
print(b == a)
print(a is b)

c = a[:]
print(c is a)
print(c == a)