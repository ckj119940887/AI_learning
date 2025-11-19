class Student:
    """this is class for student"""

    # 类属性，所有实例共享的属性
    school = "ksu"

    # 类方法
    @classmethod
    def schoolName(cls):
        print(cls.school)

    # default method
    def __init__(self):
        self.name = ""
        self.age = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def play(self):
        print(f"{self.name} is playing games")

    def study(self):
        print(f"{self.name} is studying")

    def palyAndStudy(self):
        # Student.play(self)
        # Student.study(self)
        self.play()
        self.study()

stu = Student("ckj", 20)
stu.play()
stu.study()

# add new attribute
stu.color = "red"
print(stu.color)

Student.schoolName()