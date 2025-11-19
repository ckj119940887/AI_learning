class Person:
    home = "earth"

    def __init__(self, name):
        self.__name = name

    def eat(self):
        print("eating")

class Yellow(Person):
    color = "yellow"

class White(Person):
    color = "white"

class Black(Person):
    color = "black"

class Student(Person):
    def __init__(self, name, grade):
        super().__init__(name)
        self.__grade = grade

    def study(self):
        print(self.__name)
        print(self.__grade)

    def play_game(self):
        print(self.__name)

class ChineseStudent(Student, Yellow):
    def __init__(self, name, grade):
        Student.__init__(self, name, grade)
        Yellow.__init__(self, name)

    def xuexi(self):
        Student.study(self)

    def play_game(self):
        print("ChineseStudent: " + self.__name)