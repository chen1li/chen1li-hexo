---
title: MUSIC_算法
mathjax: true
categories:
  - 算法
tags:
  - DoA
date: 2024-01-25 23:35:30
---

# 2 MUSIC 算法
## 2.1 算法原理
&emsp;&emsp;根据信号模型，可以得到接收信号的协方差矩阵，理想协方差矩阵如下：
$$
\begin{aligned}
    \mathbf{R} &= \mathrm{E}\{\mathbf{x}(t)\mathbf{x}^{H}(t)\} \\
    &= \mathbf{A}\mathrm{E}\{\mathbf{s}(t)\mathbf{s}^{H}(t)\}\mathbf{A}^H + \mathrm{E}\{\mathbf{n}(t)\mathbf{n}^{H}(t)\} \\
    &= \mathbf{A}\mathbf{R}_s\mathbf{A}^H + \sigma_n^2\mathbf{I}
\end{aligned}
$$
其中 $\mathbf{R}\_s$ 和 $\sigma_n^2\mathbf{I}$ 分别代表信号协方差矩阵和噪声协方差矩阵，$\sigma_n^2$ 为噪声方差， $\mathbf{I}$ 为单位矩阵。需要注意到的是，由于信号源之间是独立的，$\mathbf{R}_s$ 是对角非奇异矩阵；$\mathbf{R}$ 是非奇异的，同时 $\mathbf{R} = \mathbf{R}^H$，因此 $\mathbf{R}$ 是 Hermitian 矩阵且是正定矩阵。
&emsp;&emsp;但因为采样数 $T$ 有限，理想协方差矩阵无法直接获得，此时通常用样本的协方差矩阵 $\hat{\mathbf{R}}$ 来代替：
$$
\begin{equation*}
    \mathbf{R} \triangleq \hat{\mathbf{R}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{x}(t)\mathbf{x}^{H}(t) = \frac{1}{T} \mathbf{X}\mathbf{X}^H
\end{equation*}
$$
&emsp;&emsp;对 $\mathbf{R}$ 进行特征值分解，可得到：
$$
\begin{equation*}
\begin{aligned}
    \mathbf{R} &= \mathbf{U} \Sigma \mathbf{U}^H = \sum\limits_{i=1}^M \lambda_i \mathbf{u}_i \mathbf{u}_i^H \\
    &=
    \begin{bmatrix}
        \mathbf{U}_s & \mathbf{U}_n
    \end{bmatrix}
    \begin{bmatrix}
        \Sigma_s & \mathbf{O} \\
        \mathbf{O} & \Sigma_n
    \end{bmatrix}
    \begin{bmatrix}
        \mathbf{U}_s^H \\
        \mathbf{U}_n^H
    \end{bmatrix} \\
    &= \mathbf{U}_s \Sigma_s \mathbf{U}_s^H + \mathbf{U}_n \Sigma_n \mathbf{U}_n^H
\end{aligned}
\end{equation*}
$$
其中 $\lambda_i$ 和 $\mathbf{u}_i$ 分别为特征值和对应的特征向量，$\mathbf{O}$ 为全零矩阵，$\Sigma$ 为由 $M$ 个特征值 $\lambda_1, \cdots, \lambda_M$ 组成的对角矩阵：
$$
\begin{equation*}
    \Sigma = \operatorname{diag} \{\lambda_1, \cdots, \lambda_M\}
\end{equation*}
$$
且假设：
$$
\begin{equation*}
    \lambda_1 >  \cdots > \lambda_K > \lambda_{K+1} = \cdots = \lambda_M = \sigma_n^2
\end{equation*}
$$
$\Sigma_s$ 为由 $K$ 个较大特征值 $\lambda_1, \cdots, \lambda_K$ 组成的对角矩阵， $\Sigma_n$ 为由 $M-K$ 个较小特征值 $\lambda_{K+1}, \cdots, \lambda_M$ 组成的对角矩阵：
$$
\begin{equation*}
    \Sigma_s = \operatorname{diag} \{\lambda_1, \cdots, \lambda_K\},
    \Sigma_n = \operatorname{diag} \{\lambda_{K+1}, \cdots, \lambda_M\}
\end{equation*}
$$
$\mathbf{U}\_s$ 表示由 $K$ 个较大特征值 $\lambda_1, \cdots, \lambda_K$ 对应的特征矢量张成的信号子空间，$\mathbf{U}_n$ 表示由 $M-K$ 个较小特征值 $\lambda_{K+1}, \cdots, \lambda_M$ 对应的特征矢量张成的噪声子空间。注意，$\mathbf{U}$ 为酉矩阵。
&emsp;&emsp;将 $\mathbf{R} = \mathbf{A}\mathbf{R}_s\mathbf{A}^H + \sigma_n^2\mathbf{I}$ 左右两边乘以 $\mathbf{U}_n$ 可得：
$$
\begin{equation*}
    \mathbf{R}\mathbf{U}_n = \mathbf{A}\mathbf{R}_s\mathbf{A}^H \mathbf{U}_n+ \sigma_n^2\mathbf{U}_n
\end{equation*}
$$
&emsp;&emsp;同时将 $\mathbf{R} =\mathbf{U}_s \Sigma_s \mathbf{U}_s^H + \mathbf{U}_n \Sigma_n \mathbf{U}_n^H$ 左右两边乘以 $\mathbf{U}_n$ 可得：
$$
\begin{equation*}
    \mathbf{R}\mathbf{U}_n = \mathbf{U}_s \Sigma_s \mathbf{U}_s^H\mathbf{U}_n + \mathbf{U}_n \Sigma_n \mathbf{U}_n^H\mathbf{U}_n = \mathbf{O} + \sigma_n^2 \mathbf{U}_n = \sigma_n^2 \mathbf{U}_n
\end{equation*}
$$
&emsp;&emsp;联立上面两式子可得：
$$
\begin{equation*}
    \mathbf{A}\mathbf{R}_s\mathbf{A}^H \mathbf{U}_n = \mathbf{O}
\end{equation*}
$$
&emsp;&emsp;由于 $\mathbf{R}_s$ 和 $\mathbf{A}^H \mathbf{A}$ 均满秩，所以均可逆，因此上式两边同时乘以 $\mathbf{R}_s^{-1} (\mathbf{A}^H\mathbf{A})^{-1} \mathbf{A}^H$ 可得：
$$
\begin{equation*}
    \mathbf{R}_s^{-1} (\mathbf{A}^H\mathbf{A})^{-1} \mathbf{A}^H\mathbf{A}\mathbf{R}_s\mathbf{A}^H \mathbf{U}_n 
    = \mathbf{R}_s^{-1} (\mathbf{A}^H\mathbf{A})^{-1} \mathbf{A}^H\mathbf{O}
\end{equation*}
$$
化简得：
$$
\begin{equation*}
    \mathbf{A}^H \mathbf{U}_n = \mathbf{O}
\end{equation*}
$$
或者可以写成如下形式：
$$
\begin{equation*}
    \mathbf{A}^H\mathbf{u}_i = \mathbf{0}, i = K+1, \cdots, M
\end{equation*}
$$
其中 $\mathbf{0}$ 为全零向量。上式表明：噪声特征值所对应的特征向量 $\mathbf{u}_i$ 与方向矢量矩阵 $\mathbf{A}$ 的列向量正交，而 $\mathbf{A}$ 的各列与信号源的方向相对应的，因此可以利用噪声子空间求解信号源方向。

## 2.2 算法步骤
&emsp;&emsp;MUSIC 算法步骤如下（输入为阵列接收矩阵 $\mathbf{X}$）：
1. 计算协方差矩阵 $\mathbf{R} = \frac{1}{T} \mathbf{X}\mathbf{X}^H$。
2. 对 $\mathbf{R}$ 进行特征值分解，并对特征值进行排序，然后取得 $M-K$ 个较小特征值对应的特征向量来组成噪声子空间 $\mathbf{U}_n$。
3. 以下式遍历 $\theta \in [-90^{\circ}, 90^{\circ}]$：
    $$
    \begin{equation*}
        P(\theta) = \frac{1}{\mathbf{a}^H(\theta)\mathbf{U}_n\mathbf{U}_n^H\mathbf{a}(\theta)}
    \end{equation*}
    $$
    此时得到一组 $P(\theta)$，$K$ 个最大值对应的 $\theta$ 就是需要返回的结果。
## 2.3 代码实现

    ```matlab
    % music.m
    clear;
    clc;
    close all;

    % 参数设定
    c = 3e8;                                              % 光速
    fc = 500e6;                                           % 载波频率
    lambda = c/fc;                                        % 波长
    d = lambda/2;                                         % 阵元间距，可设 2*d = lambda
    twpi = 2.0*pi;                                        % 2pi
    derad = pi/180;                                       % 角度转弧度
    theta = [-20, 30]*derad;                              % 待估计角度
    idx = 0:1:7; idx = idx';                              % 阵元位置索引

    M = length(idx);                                      % 阵元数
    K = length(theta);                                    % 信源数
    T = 512;                                              % 快拍数
    SNR = 0;                                              % 信噪比

    %% 信号模型建立
    S = randn(K,T) + 1j*randn(K,T);                       % 复信号矩阵S，维度为K*T
    % A = exp(-1j*twpi*d*idx*sin(theta)/lambda);           % 方向矢量矩阵A，维度为M*K
    A = exp(-1j*pi*idx*sin(theta));                       % 2d = lambda，直接忽略不写
    X = A*S;                                              % 接收矩阵X，维度为M*T
    X = awgn(X,SNR,'measured');                           % 添加噪声

    %% MUSIC 算法
    % 计算协方差矩阵
    R = X*X'/T;
    % 特征值分解并取得噪声子空间
    [U,D] = eig(R);                                       % 特征值分解
    [D,I] = sort(diag(D));                                % 将特征值排序从小到大
    U = fliplr(U(:, I));                                  % 对应特征矢量排序，fliplr 之后，较大特征值对应的特征矢量在前面
    Un = U(:, K+1:M);                                     % 噪声子空间
    Gn = Un*Un';
    % 空间谱搜索
    searchGrids = 0.1;                                    % 搜索间隔
    ang = (-90:searchGrids:90)*derad;
    Pmusic = zeros(1, length(ang));                       % 空间谱
    for i = 1:length(ang)
        a = exp(-1j*pi*idx*sin(ang(i)));
        Pmusic(:, i) = 1/(a'*Gn*a);
    end
    % 归一化处理，单位化为 db
    Pmusic = abs(Pmusic);
    Pmusic = 10*log10(Pmusic/max(Pmusic));
    % 作图
    figure;
    plot(ang/derad, Pmusic, '-', 'LineWidth', 1.5);
    set(gca, 'XTick',(-90:30:90));
    xlabel('\theta(\circ)', 'FontSize', 12, 'FontName', '微软雅黑');
    ylabel('空间谱(dB)', 'FontSize', 12, 'FontName', '微软雅黑');
    % 找出空间谱Pmusic中的峰值并得到其对应角度
    [pks, locs] = findpeaks(Pmusic);                      % 找极大值
    [pks, id] = sort(pks);
    locs = locs(id(end-K+1:end))-1;
    Theta = locs*searchGrids - 90;
    Theta = sort(Theta);
    disp('估计结果：');
    disp(Theta);
    ```
## 2.4 参考内容
1. [【博客园】子空间分析方法](https://www.cnblogs.com/xingshansi/p/7554200.html)
2. [【博客园】空间谱专题10：MUSIC算法](https://www.cnblogs.com/xingshansi/p/7553746.html)
3. [【CSDN】子空间方法——MUSIC算法](https://blog.csdn.net/qq_36583373/article/details/109333087)
4. [【CSDN】Traditional Comm笔记【5】：基于MUSIC Algorithm的DoA/AoA估计以及MATLAB实现](https://blog.csdn.net/S2849366069/article/details/120956470)
5. [【知乎】DoA 估计：多重信号分类 MUSIC 算法（附 MATLAB 代码）](https://zhuanlan.zhihu.com/p/613304918)