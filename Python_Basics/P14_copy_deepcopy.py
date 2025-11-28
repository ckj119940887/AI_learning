# 在change_list中我们传入了一个list，但是我们并不想修改传入的list中的元素
# 在如下例子中已经修改了original_list中的元素
import copy


def changeList(m_list):
    m_list[0] = "a"

original_list_1 = [1, 2, 3]
changeList(original_list_1)
print(original_list_1)

# 使用copy方法，这是浅拷贝，元素还是被修改了
original_list_2 = [1, 2, 3]
copy_list = original_list_2.copy()
changeList(copy_list)
print(original_list_2) # 这里确实没有改变original_list_2

def changeList2(m_list):
    """show the depp copy"""
    m_list[3].append("b")
original_list_3 = [1, 2, 3, [100, 200, 300]]
copy_list = original_list_3.copy()
changeList2(copy_list)
print(original_list_3) # 但是子列表中的元素确实被改了，子列表也是可变对象，存的是它的引用，子列表被修改了，但是其引用并没有改

# 浅拷贝：只复制最外层容器，新容器里的元素仍然指向原来的对象引用。
# 深拷贝：递归复制所有可变子对象，使新容器与原容器在任何层级都互不影响。
original_list_4 = [1, 2, 3, [100, 200, 300]]
deepcopy_list = copy.deepcopy(original_list_4)
changeList2(deepcopy_list)
print(original_list_4)

# 当函数retrun空或没有return，返回的是None
# 当函数返回多个值，会自动将多个值放到tuple中