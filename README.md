# XJTU-Graphic2023
## 主要完成的是渲染部分
除了“多线程渲染”部分，其余渲染部分均验收满分。

关于课程具体细节，详见`README_Course.md`。

## 环境

`Windows: Visual Studio 2022` + `VsCode`

## 代码使用方法

> cmake -S .. -B .
> cmake --build . --config Debug --target dandelion --parallel 8

debug模式启动：
> .\Debug\dandelion.exe

release模式启动:
> cmake --build . --config Release --target dandelion --parallel 8
> .\Release\dandelion.exe
