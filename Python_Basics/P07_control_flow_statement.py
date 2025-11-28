from nis import match
from random import randint
a = randint(0, 100)

if a > 50:
    print('a > 50')
elif a < 50 and a >10:
    print('10 < a < 50')
else:
    print('a < 10')

# match case
test = randint(0, 3)
match test:
    case 0|1|2: print("a is less than 3")
    case _: print("a is 3")

maxNum = test if test > 0 else 0
print(maxNum)

# while loop
print("test while:")
while test < 3:
    print(test)
    test = test + 1
else:
    print("test is larger than 3")

# for loop
print("test for loop")
l = [1, 2, 3]
for i in l:
    print(i, end=' ')
print()
for i in range(len(l)):
    print(l[i])

# range
# range([start,] stop [,step])
for i in range(10, 0, -1):
    print(i)

# break, continue, pass
for i in range(1, 10):
    if(i % 2 == 0):
        continue
    else:
        print(i, end = " ")
print()
for i in range(1, 10):
    if(i % 2 == 0):
        break
    else:
        print(i, end = " ")
print()
for i in range(1, 10):
    if(i % 2 == 0):
        pass
    else:
        print(i, end = " ")
