# slice, [start:stop:step]
l1 = [1,2,3,4,5]
print(l1[0:len(l1)])
print(l1[0:])
print(l1[:])
print(l1[:2])
print(l1[::2])
print(l1[::-1])

l2 = [6,7,8]
print(l1+l2)

l2.insert(0, -1)
l2.append(-2)
print(l2)

l2[1:3] = ["a", "b"]
print(l2)

# using enumerate to access list
for i,val in enumerate(l2):
    print(i, val)

# remove, del, pop
del l2[0]
# use value to delete element
l2.remove("a")
# use index to delete element
l2.pop(3)

# list expression
list_temp_1 = [i*2 for i in range(4)]
print(list_temp_1)

list_temp_2 = [i*2 for i in range(4) if i % 2 == 0]
print(list_temp_2)

list_temp_3 = ["a", "b"]
list_temp_4 = [(i,j) for i in list_temp_2 for j in list_temp_3]
print(list_temp_4)

# zip
zipped = zip(list_temp_2, list_temp_3)
print(type(zipped))
for i, j in zipped:
    print(i, j)

# list_zip is empty because zipped is an iterator, it cannot be re-used
list_zip = list(zipped)
print(list_zip)

list_zip_2 = list(zip(list_temp_2, list_temp_3))
print(list_zip_2)