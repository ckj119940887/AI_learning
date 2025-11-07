# create set
s1 = {1,2,3}
s2 = set([1,2,3])
print(s1)
print(s2)

# empty set, {} represent an empty dict
empty = set()

# add
empty.add(1)
empty.add(2)

# remove, discard
empty.remove(1)
empty.discard(1)
print(empty)

# update, must be iterable object
empty.update([11,12,13])
print(empty)

# union, |
unioned = empty.union(set([1,2,3]))
print(unioned)

# clear, clear whole set
# empty.clear()

# difference (-), difference_update
diff = empty.difference(set([1,2,3]))
empty.difference_update(diff)
print(empty)

# intersection (&), intersection_update