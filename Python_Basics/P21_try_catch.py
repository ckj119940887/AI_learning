try:
    result = 3 / 0
    print(result)
except ZeroDivisionError as e1:
    print("Division by zero")
except (IOError, ValueError) as e2:
    print("other error")
except:
    print("something else")
else: # no exception will execute this statement
    print("else")
finally: # always execute this statement no matter whether there is exeception
    print("finally")

def add(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return x + y
    else:
        raise TypeError("x and y is not int")

try:
    print(add(1, 2.0))
except TypeError as e1:
    print(e1)

# self defined exception
class MyException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

try:
    raise MyException(3)
except MyException as e1:
    print(f"my exception is {e1}")