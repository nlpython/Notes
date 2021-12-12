# 													Hadoop

## 1. HDFS

**优点**:

- 处理超大文件
- 支持流式数据访问
- 低成本运行

**缺点**:

- 不适合处理低延迟的数据访问
- 不适合处理大量的小文件
- 不适合多用户写入及任意修改文件

![img](https://img2018.cnblogs.com/blog/1271254/201909/1271254-20190925195925515-846526737.png)

### 1.1 NameNode

**名称节点**是HDFS的管理者, 它主要有三方面职责.

- 负责管理维护hdfs的命名空间(namespace), 维护其中的两个重要文件--edits, fsimage.
- 管理DataNode上的数据块(block), 维持副本数量.
- 接受客户端的请求, 比如文件的上传, 下载, 创建目录.

#### 1.1.1 维护namespace

管理namespace信息的文件有两个, 分别是命名空间镜像文件(fsimage)和操作日志文件(edits).

fsimage包含Hadoop文件系统中的所有目录和文件的序列化信息. 对于文件, 包含有修改时间, 访问时间, 块大小和组成一个文件的数据块信息等.对于目录, 包含修改时间, 访问控制权限等.

edits主要对HDFS上的各种更新操作进行记录.

#### 1.1.2 管理DataNode上的数据块

在HDFS中, 一个文件被分为一个或多个数据块, 这些数据块存储在DataNode中, NameNode负责管理数据块的所有元数据信息, 主要包括"文件名->数据块"的映射, "数据块->DataNode"映射列表.



### 1.2 DataNode

数据节点负责存储数据, 一个数据块会在多个DataNode中进行冗余备份, 一个数据块在一个DataNode上最多只有一个备份, DataNode上存储了数据块ID和数据块的内容, 以及他们的映射关系.

DataNode定时和NameNode进行心跳通信, 接受NameNode的指令.

DataNode之间还会互相通信, 执行数据块复制任务. 在客户端执行写操作时, DataNode之间需要相互配合, 保证写操作的一致性. DataNode功能如下:

- 保存数据块
- 运行DataNode线程
- 定期向NameNode发送心跳信息保持联系, 如果NameNode10分钟没有收到DataNode的心跳信息, 则认为其宕机(失去联系), 并将其上的数据块复制到其他DataNode.



### 1.3 SecondaryNameNode

HDFS定义了一个第二名称节点, 主要职责是定期把NameNode的fsimage和edits下载到本地, 并将他们加载到内存进行合并, 最后将合并后的新的fsimage上传回NameNode, 这个过程称为检查点(checkpoint). 处于可靠性考虑, SecondaryNameNode与NameNode通常运行在不同的机器上, 且他们内存一样大.

参数dfs.namenode.checkpoint.period指定连个连续检查点的时间间隔, 默认为一小时.

参数dfs.namenode.checkpoint.txns定义了NameNode上的新增事务的数量, 默认设置为1,000,000.

当时间间隔达到设定值或事务数量达到设定值, 都会启动检查点进程.



定期合并fsimage和edits文件, 使edits大小始终保持在限制范围内, 这样减少了重新启动NameNode时合并fsimage和edits耗费的时间, 从而减少NameNode的启动时间, 也有冷备份的作用.

![img](https://img-blog.csdnimg.cn/2021030718543961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1MDY5Mjc5,size_16,color_FFFFFF,t_70)



### 1.2. HDFS工作机制

#### 2.1 副本冗余存储策略

HDFS默认保存3个副本.

- 副本一: 放置在上传文件的数据节点; 如果是在集群外提交, 则随机挑选一台磁盘不太满, cpu不太忙的节点.
- 副本二: 放置在与第一个副本不同的机架的节点上.
- 副本三: 放置在与第二个副本相同机架的其他节点上.



#### 2.2 HDFS Shell命令

'$'表示普通用户, '#'表示root用户.

##### 2.2.1 创建文件夹

```
hdfs dfs -mkdir -p <path>
# -p表示如果父目录不存在, 先自动创建父目录.
hdfs dfs -mkdir /mydir
hdfs dfs -mkdir -p /mydir/dir1/dir2
```

##### 2.2.2 列出指定的文件和目录

```
hdfs dfs -ls [-d][-h][-R] <path>
# [-d]: 返回path
# [-h]: 人性化显示文件大小
# [-R]: 级联显示paths下的文件.

hdfs dfs -ls /mydir
```

##### 2.2.3 新建文件

```
hdfs dfs -touchz <path>

hdfs dfs -touchz /mydir/input.txt
```

##### 2.2.4 上传文件

```
hdfs dfs -put[copyFromLocal] [-f][-p] <localsrc> <dst>
# <localsrc>表示本地文件路径, <dst>表示保存在HDFS上的路径.

hdfs dfs -put data.txt /mydir/data.txt
```

##### 2.2.5 移动文件

与上传文件不同的是, 移动后本地文件将会被删除.

```
hdfs -moveFromLocal <localsrc> <dst>
```

##### 2.2.6 下载文件

```
hdfs dfs -get[copyToLocal] [-p] <src> <localsrc>

hdfs dfs -get /mydir/data.txt /local_data.txt
```

##### 2.2.7 查看文件

```
hdfs dfs -cat/text[-ignoreCrc] <src>
hdfs dfs -tail [-f] <file>

hdfs dfs -cat /mydir/data.txt
```

##### 2.2.8  追写文件

```
hdfs dfs -appendToFile <localsrc> <dst>

# 该命令将localsrc指向的本地文件内容写入目标文件dst. 如果localsrc是"- ", 表示数据来自键盘, "Ctrl+C"结束输入.

hdfs dfs -appendToFile data.txt /mydir/data.txt
```

##### 2.2.9 删除文件

```
hdfs dfs -rm [-f][-r] <src>
# -f: 如果要删除的文件不存在, 不显示错误消息
# -r: 级联删除目录下的所有文件和子目录文件
hdfs dfs -rm /mydemo/data.txt
```

##### 2.2.10 显示占用的磁盘空间大小

```
hdfs dfs -du [-s][-h] <path>
# [-s]: 显示指定目录下文件总大小
# [-h]: 人性化显示大小
```

##### 2.2.11 文件复制

```
hdfs dfs -cp <src> <dst>

hdfs dfs -cp /mydir/data.txt /mydir/data1.txt
```

##### 2.2.12 文件移动(改名)

```
hdfs dfs -mv <src> <dst>

hdfs dfs -mv /mydir/data1.txt /mydir/data2.txt
```

**P66**

##### 2.2.13 管理命令

```
# 报告文件系统的基本信息和统计信息
hdfs dfsadmin -report
# 查看拓扑
hdfs dfsadmin -printTopology
```



### 1.3 HDFS的高级功能

#### 3.1 安全模式

安全模式是HDFS所处的一种特殊状态. 处于这种状态时, HDFS只接受读数据请求, 不能对文件进行写, 删除等操作. 在NameNode主节点启动时, HDFS首先进入安全模式, DataNode会向NameNode上传他们数据块的列表, 让NameNode得到数据块的位置信息, 并对每个文件对应的数据块副本进行统计. 当最小副本条件满足时, 即数据块都达到最小副本数, HDFS自动离开安全模式.

#### 3.2 回收站

#### 3.3 快照

快照是HDFS2.x版本增加的基于时间点的数据的备份(复制).

#### 3.4 配额

```
hdfs dfsadmin -setQuota 5 /user/dataset
# setQuota命令设置HDFS中某个目录下文件数量和目录数量之和的最大值.
hdfs dfsadmin -setSpaceQuota 13417728 /user/dataset
# setSpaceQuota设置最大存储空间
```

#### 3.5 高可用性

#### 3.6 联邦



## 2. YARN

### 2.1 MapReduce

![img](https://images2015.cnblogs.com/blog/615800/201604/615800-20160419221431929-23331495.jpg)

如果为MapReduce架构图：它由Clint（客户端），JobTracker和TaskTracker组成，其中，JobTracker和TaskTracker是MapReduce1最重要的组成部分。JobTracker的职责主要是负责资源管理和所有作业的控制，TaskTracker的职责主要负责接受来自JobTracker的命令并执行。

### 2.2 YARN

**MapReduce2 = MapReduce1 + YARN.**

#### 2.2.1 Countainer

YARN中的资源包括内存，GPU，磁盘IO等. Container是YARN中的资源抽象, 它封装了某个节点上的多维资源. YARN会为每个任务分配Container.

#### 2.2.2 ResourceManager

负责整个系统的资源分配和管理, 是一个全局资源管理器, 主要由两个组件构成: 调度器(Scheduler)和应用程度管理器(ApplicationManager). 调度器根据资源情况为应用程序分配封装在Container中的资源.

#### 2.2.3 NodeManager

NodeManager是每个节点上的资源和任务管理器. 它定时向ResourceManager汇报本节点上的资源使用情况和各个Container的运行状态, 接收并处理来自ApplicationManager的Container启动/停止等请求.

#### 2.2.4 ApplicationMaster

ApplicationMaster是一个详细的框架库, 它结合从ResourceManager获得的资源与NodeManager协同工作, 来运行和监控任务.





## 3. MapReducecc















































