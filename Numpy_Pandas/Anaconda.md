# 去Anacodna官网下载安装包
```
往往安装的是miniconda
```

# pycharm中添加本地interpreter
```
在pycharm右下角添加基于conda的解释器
conda的路径位于/home/kejun/software/miniconda3/bin/conda
```

# pycharm 安装新的依赖
```
在pycharm右下角选择Interpreter Setting来安装新的依赖
在python-->Interpreter-->选择"+"
```

# 命令
```
# 创建环境
conda create -n env_name python=3.12

# 删除环境
conda remove -n env_name --all

# 查看所有conda 环境
conda env list

# 退出环境
conda deactivate

# 切换环境
conda activate env_name

# 安装依赖
conda install dep_name[=version]

# 更新依赖
conda update dep_name

# 卸载依赖
conda remove dep_name
```

# Terms of Service 接受问题
```
如果提到Terms of Service的问题，请执行如下命令：
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

```