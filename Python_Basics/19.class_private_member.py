class Person1:
    def __init__(self, name, age):
        # public member
        self.name = name
        self.age = age

p1 = Person1("kejun", 100)
print(p1.name)

class Person2:
    # 私有成员以__开头且结尾不能多于一个_
    # python编译器会自动将私有成员的名字转换为 _类名__x
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

p2 = Person2("chen", 100)
# print(p2.name)
print(p2._Person2__name)

class Person3:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def eat1(self):
        print(self.__name)

    # 调用时可以省去括号
    # 可以用来限制读权限
    @property
    def eat2(self):
        if(self.__name == "kejun"):
            print("chen")
        else:
            print(self.__name)

    # 设置写权限
    @eat2.setter
    def eat2(self, eat):
        self.__name = eat

p3 = Person3("chen", 1099)
p3.eat1()
p3.eat2
p3.eat2 = "kejun"
p3.eat2