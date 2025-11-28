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
for e in fb.__closure__:
    print(e.cell_contents)
