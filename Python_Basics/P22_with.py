# with将try-except-finally打包起来了
# 执行with时，自动调用__enter__和__exit__方法，最后会自动调用close方法

# try:
#     file = open("file.txt", "w")
#     file.write(a)
#     file.close() # this statement is not executed
# except:
#     print("file status: ", file.closed)

try:
    with open("file.txt") as f:
        f.write(a)
finally:
    print("f status: ", f.closed)

