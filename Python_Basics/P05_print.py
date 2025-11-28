num1 = 1
num2 = 2
print("int1 = %d, int2 = %d" % (num1, num2))

# use format
print("int1 = {}, int2 = {}".format(num1, num2))
# specify the position index
print("int1 = {1}, int2 = {0}".format(num1, num2))
# specify arg name
print("int1 = {n0}, int2 = {n1}".format(n0 = num1, n1 = num2))

# use f
print(f"int1 = {num1}, int2 = {num2}")