# 包和普通文件夹的唯一区别就是有没有__init__.py
# __init__.py 使用 __all__ = ['module1', 'module2']
# 使用__all__定义了所有使用 from xxx import * 可以导入的包

# 全局导入
# import 包名.模块名 [as]

# 局部导入
# from 顶层包 import 底层包
# from 顶层包.底层包.模块名 import 成员