---
title: hexo_init_报错
mathjax: true
categories:
  - 日常
tags:
  - debug
date: 2024-01-25 23:34:21
---

# hexo init 报错
```shell
$ hexo init -g --clone=git@github.com:hexojs/hexo-starter.git
INFO  Cloning hexo-starter https://github.com/hexojs/hexo-starter.git
fatal: unable to access 'https://github.com/hexojs/hexo-starter.git/': Failed to connect to github.com port 443 after 21101 ms: Couldn't connect to server
WARN  git clone failed. Copying data instead
INFO  Install dependencies
WARN  Failed to install dependencies. Please run 'npm install' in "D:\Program Files\GitHub\myblog" folder.
```

# 解决方案
- 这是两个报错，需要分别解决：
    1. 对于 `unable to access` 的报错，大概率是因为 GitHub 无法 ping 通，尝试了网上说的很多方法（包括连接梯子依然不可行），最后修改了 windows 系统的 hosts 文件才成功，具体来说，需要查找 GitHub 的 IP 地址，然后在 hosts 文件中添加一行 `IP github.com`，后续即可成功。
        - 在 wins10 系统下 hosts 文件在 `C:\WINDOWS\system32\drivers\etc` 目录中。
    2. 对于第二个报错，网上推荐先用 `npm` 安装 `cnpm` ，接着运行 `cnpm install` 来代替 `npm install`，这样会更快，但是整个过程仍然报错（整个过程指的是用 `npm install cnpm` 或者 `npm install`）：
        ```shell
        $ npm install -g cnpm --registry=https://registry.npm.taobao.org
        npm ERR! code CERT_HAS_EXPIRED
        npm ERR! errno CERT_HAS_EXPIRED
        npm ERR! request to https://registry.npm.taobao.org/cnpm failed, reason: certificate has expired

        npm ERR! A complete log of this run can be found in
        ```
        最后一行 error 不用在意，关键在于前三行，说的是认证过期，参考内容 3 给出了解决方案来取消 ssl 验证：
        ```shell
        npm config set strict-ssl false
        ```

# 参考内容
1. [hexo init 403超时报错](https://www.jianshu.com/p/f52a20db0f85)
2. [解决ping github.com超时](https://juejin.cn/post/6844904194852257805)
2. [cnpm install或者npm install遇到的问题Error: certificate has expired](https://blog.csdn.net/qq_42761482/article/details/121018086)