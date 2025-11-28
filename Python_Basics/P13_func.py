def add(a,b):
    return a+b
print(add(1,2))

# passing immutable object
def plusOne(a):
    print(f"a before modification: {id(a)}")
    a = 20
    print(f"a after modification: {id(a)}")

b = 10
plusOne(b)
print(f"b after function: {id(b)}")

# passing mutable object
def changeList(l):
    print(f"l before modification: {id(l)}")
    l[0] = 0
    print(f"l after modification: {id(l)}")

targetList = [1,2,3]
changeList(targetList)
print(f"targetList after function: {id(targetList)}")
print(targetList)

# variable length parameter, * represent tuple, ** represent dict
def func1(num, *args):
    print(f"args is : {args}")

func1(1,1,2,3,4)

def func2(num, **args):
    print(f"args is : {args}")

# func2(1, 2, 3, 4), error, because ** represent dict, not tuple
func2(1, a=1, b=2, c=3)

# deference-package parameter
def funcTemp(a, b, c):
    print(f"a: {a}, b: {b}, c: {c}")

funcTemp(*(1, 2, 3)) # passing a tuple
funcTemp(**{"":1, "b":2, "c":3}) # passing a dict