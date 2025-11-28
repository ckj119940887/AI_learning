# 每个py文件都是一个模块
# 全局导入
import P13_func
print(P13_func.add(1,2))

import P13_func as p13
print(p13.add(1,2))

# 局部导入
from P13_func import add as plus, plusOne as plus1
print(plus(1,2))

# 特殊情况
# 导入所有不以下划线_开头的成员
# 用__all__可以限制上述情况外的成员, __all__ = ["add", "plusOne"]
from P13_func import *
