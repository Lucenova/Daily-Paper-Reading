### Tmux

命令行的典型使用方式是，打开一个终端窗口（terminal window，以下简称"窗口"），在里面输入命令。**用户与计算机的这种临时的交互，称为一次"会话"（session）** 。

会话的一个重要特点是，窗口与其中启动的进程是[连在一起](https://www.ruanyifeng.com/blog/2016/02/linux-daemon.html)的。打开窗口，会话开始；关闭窗口，会话结束，会话内部的进程也会随之终止，不管有没有运行完。

#### Tmux的基本用法

我常用的场景是使用Mac SSH到Linux服务器上，然后在终端上使用Tmux。

需要记住的Tmux的命令是：

```bash
# 新建一个指定名称的会话
tmux new -s <session-name>
# 将当前窗口与会话分离
tmux detach
# 查看当前所有的会话
tmux ls
# 重新接入某个已经存在的会话
tmux attach -t <session-name>
# 杀死某个会话
tmux kill-session -t <session-name>
# 切换会话
tmux switch -t <session-name>
```

需要记住的mac进入tmux prefix的快捷键是：

control + b

