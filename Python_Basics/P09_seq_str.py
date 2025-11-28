str1 = "abcde"
print(str1[1])
print(str1[1:2])

print(str1 + str1)
print("a"*10)

# check character
print("a" in str1)

# keep original string
str2 = r"ab\t\tde"
str3 = "ab\t\tde"
print(str2)
print(str3)

# replace(old, new, [,maxTimes])
replaced = str1.replace("a", "b")
print(replaced)

# split(x, [,n])
str4 = "a,b,c,d,e,f"
splited = str4.split(",")
print(splited)

# x.join(seq)
l = ['a', 'b', 'c', 'd', 'e', 'f']
joined = "-".join(l)
print(joined)

# find(x[,start][,end])
print(joined.count("a"))

# index
print(joined.index("b"))
