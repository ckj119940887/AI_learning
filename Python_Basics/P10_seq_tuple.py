tup1 = ("a", "b", "c")
# tuple only one element
tup2 = ("a",)

# cannot get the tuple directly
tup3_gen = (i for i in range(3))
tup4 = tuple(tup3_gen)
print(tup4)

print(tup4[0:-1])