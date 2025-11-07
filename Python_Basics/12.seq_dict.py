# dict, key is immutable, value is mutable
empty_dict = dict()

# create dict
dict1 = {"a":"A", "b":"B", "c":"C"}
dict2 = dict(a="A", b="B", c="C")
dict3 = dict([("a", "A"), ("b", "B"), ("c", "C")])
print(dict1)
print(dict2)
print(dict3)
dict4 = dict([(i, i*2) for i in range(3)])
print(dict4)

# access element
print(dict1["a"])
print(dict1.get("b"))

# add/modify element
dict1["d"] = "D"
print(dict1)

# check key, cannot check value
print("b" in dict1)

# iteratoin
for key in dict1.keys():
    print(key)
for value in dict1.values():
    print(value)
for key, value in dict1.items():
    print(key, value)

# delete / pop
del dict1["a"]
dict2.pop("b")
print(dict2.pop("c"))

# delete the whole dict
# del dict1

# clear
dict2.clear()
