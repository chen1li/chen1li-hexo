---
title: unitary_MUSIC_算法
mathjax: true
categories:
  - 论文
tags:
  - DoA
date: 2024-01-26 21:29:15
---

# unitary MUSIC 算法
&emsp;&emsp;论文 *A Unitary Transformation Method for Angle-of-Arrival Estimation* 中提出了 unitary MUSIC 的算法，直译就是酉 MUSIC 算法，即酉变换 MUSIC 算法。该算法的目的是简化计算复杂度，将传统 MUSIC 算法中的复数 SVD 和复数网格搜索计算转化为实数计算。在学习 unitary MUSIC 之前需要理解 Hermitian 矩阵及 Persymmetric 矩阵的概念及性质：

1. Hermitian 矩阵指的是满足 $\mathbf{A}^H = \mathbf{A}$ 的矩阵 $\mathbf{A}$；
2. Persymmetric 矩阵指的是满足 $\mathbf{A}\mathbf{J} = \mathbf{J}\mathbf{A}^T$ 的矩阵 $\mathbf{A}$，其中 $\mathbf{J}$ 其对角线从左下至右上，很多地方又称 $\mathbf{J}$ 为选择矩阵。
3. 假若矩阵 $\mathbf{A}$ 既为 Hermitian 矩阵又为 Persymmetric 矩阵，则满足：
$$\mathbf{J}\mathbf{A}^*\mathbf{J}=\mathbf{A}$$
其中 $\mathbf{A}^*$ 为 $\mathbf{A}$ 的共轭。

&emsp;&emsp;在接下来的讨论中，$\mathbf{I}$ 和 $\mathbf{J}$ 分别用作表示单位矩阵和选择矩阵，下文中将会出现这两种矩阵的运算，例如 $\mathbf{A}\mathbf{I}$ 或 $\mathbf{J}\mathbf{B}$，设 $\mathbf{A}$ 和 $\mathbf{B}$ 均为方阵，如果没有特别强调，则说明 $\mathbf{I}$ 和 $\mathbf{J}$ 分别和 $\mathbf{A}$ 和 $\mathbf{B}$ 同维度。
## 算法原理
&emsp;&emsp;前面讨论的子空间算法中，复协方差矩阵的特征值分解是至关重要的一步，然而该步的计算量很高。为了降低计算量，unitary MUSIC 算法考虑利用一个酉矩阵将原先的复协方差矩阵 $\mathbf{R}$ 转换成实协方差矩阵，同时传统算法中的复空间搜索向量 $\mathbf{a}(\theta)$ 也用实向量来代替。
&emsp;&emsp;unitary MUSIC 算法的提出基于一个性质，即若不相关的窄带远场信号源射入均匀线阵中，其协方差矩阵不仅是 Hermitian，且 Persymmetric。通常估计的协方差矩阵 $\mathbf{R}\triangleq \hat{\mathbf{R}}$ 仅仅只是 Hermitian 矩阵但不满足 Persymmetric 性质，需要先获得一个满足 Persymmetric 性质的估计协方差矩阵：
$$
\begin{equation*}
\mathbf{R}\triangleq \frac{1}{2}(\hat{\mathbf{R}}+ \mathbf{J}\hat{\mathbf{R}}^*\mathbf{J})
\end{equation*}
$$
&emsp;&emsp;假设阵元数 $M$ 为偶数，unitary MUSIC 算法引入了一个酉矩阵 $\mathbf{Q}\in\mathbb{C}^{M\times M}$：
$$
\mathbf{Q} = \frac{1}{\sqrt{2}}
\begin{bmatrix}
    \mathbf{I} & \mathbf{J} \\
    \mathrm{j}\mathbf{J} & -\mathrm{j}\mathbf{I}
\end{bmatrix}
$$
其中 $\mathbf{I}$ 和 $\mathbf{J}$ 分别为单位矩阵和选择矩阵，且该两个矩阵维度均为 $\frac{M}{2}\times \frac{M}{2}$。易得 $\mathbf{Q}$ 为酉矩阵，即 $\mathbf{Q}^{-1} = \mathbf{Q}^H$，同时满足：
$$\mathbf{Q}^*\mathbf{J} = \mathbf{Q}$$
&emsp;&emsp;至此到了本算法的关键，它在于证明**由于 $\mathbf{R}$ 是 Hermitian 且 Persymmetric 矩阵，$\mathbf{Q}\mathbf{R}\mathbf{Q}^H$ 是实对称矩阵**：
>因为 $\mathbf{R}$ 为 Hermitian，易得 $\mathbf{Q}\mathbf{R}\mathbf{Q}^H$ 为 Hermitian；因此只需要证明 $\mathbf{Q}\mathbf{R}\mathbf{Q}^H$ 是实矩阵，即证明 $(\mathbf{Q}\mathbf{R}\mathbf{Q}^H)^* = \mathbf{Q}\mathbf{R}\mathbf{Q}^H$：
$$
\begin{equation*}
    \begin{aligned}
    &(\mathbf{Q}\mathbf{R}\mathbf{Q}^H)^* \\
    = &\mathbf{Q}^*\mathbf{R}^*\mathbf{Q}^T \\
    = &(\mathbf{Q}^*\mathbf{J})(\mathbf{J}\mathbf{R}^*\mathbf{J})(\mathbf{J}\mathbf{Q}^T)\\
    = &\mathbf{Q}\mathbf{R}\mathbf{Q}^H
    \end{aligned}
\end{equation*}
$$
由此得证。

&emsp;&emsp;综上所述，unitary MUSIC 算法引入酉矩阵 $\mathbf{Q}$ 并令 $\mathbf{R}\triangleq\mathbf{Q}\mathbf{R}\mathbf{Q}^H$，使得酉变换后的协方差矩阵变为实对称矩阵，接着对其特征值分解即可进行后续的搜索步骤。而 ULA 的搜索方向矢量为：
$$
\mathbf{a}(\theta) = \left[1, e^{-\mathrm{j}2\pi d\sin\theta/\lambda},\cdots,e^{-\mathrm{j}(M-1)2\pi d\sin\theta/\lambda}\right]^T
$$
则 unitary MUSIC 算法的搜索方向矢量为 $\mathbf{a}(\theta)\triangleq\mathbf{Q}\mathbf{a}(\theta)$。
&emsp;&emsp;为了进一步降低算法计算复杂度，Unitary MUSIC 算法考虑将搜索方向矢量也用实变量代替，做法如下：
$$
\begin{aligned}
    \mathbf{a}(\theta) &\triangleq e^{j\frac{M-1}{2}2\pi d\sin\theta/\lambda}\mathbf{Q}\mathbf{a}(\theta) \\
    &= \mathbf{Q}\left[e^{\mathrm{j}\frac{M-1}{2}2\pi d\sin\theta/\lambda},\cdots, e^{\mathrm{j}\frac{1}{2}2\pi d\sin\theta/\lambda},e^{-\mathrm{j}\frac{1}{2}2\pi d\sin\theta/\lambda},\cdots,e^{-\mathrm{j}\frac{M-1}{2}2\pi d\sin\theta/\lambda}\right]^T
\end{aligned}
$$
不难看出原方向矢量对应的阵列索引位置为 $\{0,1,\cdots,M-1\}$，而更新后方向矢量对应的阵列索引位置为 $\{-\frac{M-1}{2},-\frac{M-3}{2},\cdots,-\frac{1}{2},\frac{1}{2},\cdots,\frac{M-3}{2},\frac{M-1}{2}\}$。此时 unitary MUSIC 算法的搜索方向矢量更新为：
$$
    \begin{aligned}
        \overline{\mathbf{a}}(\theta)&= e^{\mathrm{j}\frac{M-1}{2}2\pi d\sin\theta/\lambda}\mathbf{Q}\mathbf{a}(\theta) \\
        &= \sqrt{2}
        \begin{bmatrix}
            \cos\left(\frac{M-1}{2}2\pi d\sin\theta/\lambda\right) \\
            \vdots \\
            \cos\left(\frac{1}{2}2\pi d\sin\theta/\lambda\right) \\
            \sin\left(-\frac{1}{2}2\pi d\sin\theta/\lambda\right) \\
            \vdots \\
            \sin\left(-\frac{M-1}{2}2\pi d\sin\theta/\lambda\right)
        \end{bmatrix}
    \end{aligned}
$$
&emsp;&emsp;当 $M$ 为奇数时，酉矩阵 $\mathbf{Q}$ 的形式为：
$$
\mathbf{Q} = \frac{1}{\sqrt{2}}
\begin{bmatrix}
    \mathbf{I} & \mathbf{O} & \mathbf{J} \\
    \mathbf{O}^T & \sqrt{2} & \mathbf{O}^T \\
    \mathrm{j}\mathbf{J} & \mathbf{O} & -\mathrm{j}\mathbf{I}
\end{bmatrix}
$$
其中 $\mathbf{O}$ 为零矩阵。
## 进一步理解
&emsp;&emsp;在上一小节中，我整理了论文对于 unitary MUSIC 算法的解释及证明，其核心思想在于酉矩阵 $\mathbf{Q}$ 的提出。在本小节中，我将进一步对 $\mathbf{Q}$ 的作用谈谈自己的理解。
&emsp;&emsp;在论文中，unitary 算法一直强调 $\mathbf{R}$ 的 Hermitian 及 Persymmetric 性质，因为假若 $\mathbf{R}$ 不满足这两个性质，$\mathbf{Q}\mathbf{R}\mathbf{Q}^H$ 将不是实矩阵，但是利用 $\mathrm{Re}(\mathbf{Q}\mathbf{R}\mathbf{Q}^H)$ 来进行后续的估计其实是可以估计角度的。假设 $M=4$ 和 $\varphi = 2\pi d \sin\theta/\lambda$，展开 $\mathbf{Q}\mathbf{a}(\theta)$：
$$
\begin{aligned}
\mathbf{Q}\mathbf{a}(\theta) &= \sqrt{2} \begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
0 & \mathrm{j} & -\mathrm{j} & 0 \\
\mathrm{j} & 0 & 0 & -\mathrm{j}
\end{bmatrix}
\begin{bmatrix}
1\\
e^{-\mathrm{j}\varphi}\\
e^{-\mathrm{j}2\varphi}\\
e^{-\mathrm{j}3\varphi}
\end{bmatrix} \\
&=\sqrt{2}\begin{bmatrix}
1 + e^{-\mathrm{j}3\varphi}\\
e^{-\mathrm{j}\varphi} + e^{-\mathrm{j}2\varphi}\\
\mathrm{j}(e^{-\mathrm{j}\varphi} - e^{-\mathrm{j}2\varphi})\\
\mathrm{j}(1 - e^{-\mathrm{j}3\varphi})
\end{bmatrix}\\
&=\sqrt{2}e^{-\mathrm{j}\frac{3}{2}\varphi}\begin{bmatrix}
e^{\mathrm{j}\frac{3}{2}\varphi} + e^{-\mathrm{j}\frac{3}{2}\varphi}\\
e^{\mathrm{j}\frac{1}{2}\varphi} + e^{-\mathrm{j}\frac{1}{2}\varphi}\\
\mathrm{j}(e^{\mathrm{j}\frac{1}{2}\varphi} - e^{-\mathrm{j}\frac{1}{2}\varphi})\\
\mathrm{j}(e^{\mathrm{j}\frac{3}{2}\varphi} - e^{-\mathrm{j}\frac{3}{2}\varphi})
\end{bmatrix}\\
&=\sqrt{2}e^{-\mathrm{j}\frac{3}{2}\varphi}\begin{bmatrix}
\cos(\frac{3}{2}\varphi) \\
\cos(\frac{1}{2}\varphi) \\
\sin(-\frac{1}{2}\varphi) \\
\sin(-\frac{3}{2}\varphi)
\end{bmatrix}\\
&=e^{-\mathrm{j}\frac{3}{2}\varphi}\overline{\mathbf{a}}(\theta)
\end{aligned}
$$
&emsp;&emsp;至此可以得到 $\mathbf{Q}\mathbf{a}(\theta)=e^{-\mathrm{j}\frac{M-1}{2}\varphi}\overline{\mathbf{a}}(\theta)$，进一步我们可以得到：
$$\mathbf{Q}\mathbf{A}=e^{-\mathrm{j}\frac{M-1}{2}\varphi}\overline{\mathbf{A}}$$
其中 $\overline{\mathbf{A}}\in\mathbb{R}^{M\times K}$ 是由 $K$ 个形如 $\overline{\mathbf{a}}(\theta)$ 的实向量组成的矩阵。最后一步，我们可以得到：
$$
\begin{aligned}
\mathbf{Q}\mathbf{R}\mathbf{Q}^H&=\left(e^{-\mathrm{j}\frac{M-1}{2}\varphi}\overline{\mathbf{A}}\right)\mathbf{R}_s\left(e^{\mathrm{j}\frac{M-1}{2}\varphi}\overline{\mathbf{A}}^T\right)\\
&=\overline{\mathbf{A}}\mathbf{R}_s\overline{\mathbf{A}}^T
\end{aligned}
$$
因此有 $\mathrm{Re}(\mathbf{Q}\mathbf{R}\mathbf{Q}^H) = \overline{\mathbf{A}}\mathrm{Re}(\mathbf{R}_s)\overline{\mathbf{A}}^T$，不难看出即使 $\mathbf{R}$ 不满足 Hermitian 及 Persymmetric 性质，仍然不会破坏正交性。
&emsp;&emsp;总的来说，$\mathbf{Q}$ 的作用就是使得方向矢量转为实向量，如此便可以利用协方差矩阵的实部进行后续的计算。
## 算法步骤
&emsp;&emsp;unitary MUSIC 算法步骤如下（输入为阵列接收矩阵 $\mathbf{X}$）：
1. 计算协方差矩阵 $\mathbf{R} = \frac{1}{T} \mathbf{X}\mathbf{X}^H$ 和酉矩阵 $\mathbf{Q}$，接着以 $\mathbf{R}\triangleq\mathbf{Q}\mathbf{R}\mathbf{Q}^H$ 更新协方差矩阵。
2. 对 $\mathbf{R}$ 进行特征值分解，并对特征值进行排序，然后取得 $M-K$ 个较小特征值对应的特征向量来组成噪声子空间 $\mathbf{U}_n$。
3. 以下式遍历 $\theta \in [-90^{\circ}, 90^{\circ}]$：
    $$
    \begin{equation*}
        P(\theta) = \frac{1}{\overline{\mathbf{a}}^H(\theta)\mathbf{U}_n\mathbf{U}_n^T\overline{\mathbf{a}}(\theta)}
    \end{equation*}
    $$
    此时得到一组 $P(\theta)$，$K$ 个最大值对应的 $\theta$ 就是需要返回的结果。
## 代码实现

    ```matlab
    clear; close all; clc;

    %% Parameters
    lambda     = 3e8/1e9;         % wavelength, c/f
    d          = lambda/4;        % distance between sensors
    theta      = [10,20];         % true DoAs, 1 times K vector
    theta      = sort(theta);
    M          = 16;              % # of sensors
    T          = 500;             % # of snapshots
    K          = length(theta);   % # of signals
    noise_flag = 1;
    SNR        = 0;               % signal-to-noise ratio
    grid       = 0.1;             % search grid

    %% Signals
    R = generateSignal(M,K,T,theta,lambda,d,noise_flag,SNR);

    %% DoA 
    % unitary-MUSIC
    [theta_unitary_music,P_unitary_music] = unitaryMUSIC(R,M,K,lambda,d,grid);

    %% plot
    figure;
    hold on;
    ang_list = -90:grid:90;
    plot(ang_list, P_unitary_music);
    hold off;

    function [R,X,A,S] = generateSignal(M,K,T,theta,lambda,d,noise_flag,SNR)
        S = exp(1j*2*pi*randn(K,T)); % signal matrix
        A = exp(-1j*(0:M-1)'*2*pi*d/lambda*sind(theta)); % steering vector matrix
        N = noise_flag.*sqrt(10.^(-SNR/10))*(1/sqrt(2))*(randn(M,T)+1j*randn(M,T)); % noise matrix
        X = A*S+N; % received matrix
        R = X*X'/T; % covariance matrix
    end

    function [theta,P] = unitaryMUSIC(R,M,K,lambda,d,grid)
        M_half = floor(M/2);
        O = zeros(M_half,1);
        I = eye(M_half);
        J = fliplr(I);
        if mod(M,2) == 0
            Q = [I,J;1j*J,-1j*I]./sqrt(2);
        else
            Q = [I,O,J;O',sqrt(2),O';1j*J,O,-1j*I]./sqrt(2);
        end
        R = real(Q*R*Q');
        [U,~] = svd(R);
        Un = U(:, K+1:M);

        a_list = exp(-1j*(0:M-1).'*2*pi*d/lambda*sind(-90:grid:90));
        a_list = a_list.*exp(1j*(M-1)*ones(M,1)*pi*d/lambda*sind(-90:grid:90));
        a_list = real(Q*a_list);
        P = arrayfun(@(i) 1/norm(Un'*a_list(:,i),'fro'),1:size(a_list,2)); % spectral spectrum grid search
        P = 10*log10(P./max(P));

        [~, idx] = findpeaks(P,'NPeaks',K,'SortStr','descend'); % find K peaks
        theta = sort((idx-1)*grid-90);
    end
    ```

## 参考内容
1. Huarng K C, Yeh C C. A unitary transformation method for angle-of-arrival estimation[J]. IEEE Transactions on Signal Processing, 1991, 39(4): 975-977.
2. [【wikipedia】Persymmetric matrix](https://en.wikipedia.org/wiki/Persymmetric_matrix)
3. [【wikipedia】Hermitian matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)