---
title: latex速查
date: 2017-11-17 11:09:26
tags: latex
categories: tool
---

latex速查大全

<!-- more -->

## 上下标

$a^1$ $a^{(2)}$ $a_1$ $a_{(2)}$
`$a^1$ $a^{(2)}$ $a_1$ $a_{(2)}$`

$_ZX$
`$_ZX$`

## 分式

$\frac{1}{2}$
`\frac{1}{2}`

## 根号

$\sqrt{x}$ $\sqrt[n]{x}$
`$\sqrt{x}$ $\sqrt[n]{x}$`

## 求和、连乘

$\sum_{k=1}^n$ $\sum{n \atop k=1}$
`$\sum_{k=1}^n$ $\sum{n \atop k=1}$`

$\prod_{k=1}^n$ $\prod{n \atop k=1}$
`$\prod_{k=1}^n$ $\prod{n \atop k=1}$`

## 微积分

$\mathrm{d}x$ $\partial{x}$
`$\mathrm{d}x$ $\partial{x}$`

$\int_{k=1}^n$ $\int{n \atop k=1}$
`$\int_{k=1}^n$ $\int{n \atop k=1}$`

## 各种线

$\overline{x+y}$ $\underline{x+y}$
`$\overline{x+y}$ $\underline{x+y}$`

$\overbrace{x+y}$ $\underbrace{x+y}$
`$\overbrace{x+y}$ $\underbrace{x+y}$`

$\underbrace{ a + \overbrace{b+\cdots+b}^{10} }_{20}$
`$\underbrace{ a + \overbrace{b+\cdots+b}^{10} }_{20}$`

## 堆叠符号

$\stackrel{\mathrm{def}}{=}$
`$\stackrel{\mathrm{def}}{=}$`

这种方式得到的上下符号字号不同，要得到平等地位的结构，使用`{上公式 \atop 下公式}`

## 括号定界符

使用`\left(` `\right)`来放大括号

需要配对使用，若只有一端需要，另一端用`.`替代，如`\left.`

## 矩阵

$$
\left[
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{matrix}
\right]
$$

```
\left[
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{matrix}
\right]
$$
```

## 极限

$\lim_{k\to\infty}k^{-1}=0$
`\lim_{k\to\infty}k^{-1}=0`

## 取整

| Symbol    | Latex     |
| --------- | --------- |
| $\lfloor$ | `\lfloor` |
| $\rfloor$ | `\rfloor` |
| $\lceil$  | `\lceil`  |
| $\rceil$  | `\rceil`  |

## 对数

$log_ab$ $\ln a$ $\lg10$
`$\overline{x+y}$ $\underline{x+y}$`

## 组合数

${n\choose m}$
`{n\choose m}`

## 推出

| Symbol                | Latex                 | Symbol                | Latex                 |
| --------------------- | --------------------- | --------------------- | --------------------- |
| $\leftarrow$          | `\leftarrow`          | $\Leftarrow$          | `\Leftarrow`          |
| $\rightarrow$         | `\rightarrow`         | $\Rightarrow$         | `\Rightarrow`         |
| $\leftrightarrow$     | `\leftrightarrow`     | $\Leftrightarrow$     | `\Leftrightarrow`     |
| $\uparrow$            | `\uparrow`            | $\Uparrow$            | `\Uparrow`            |
| $\downarrow$          | `\downarrow`          | $\Downarrow$          | `\Downarrow`          |
| $\updownarrow$        | `\updownarrow`        | $\Updownarrow$        | `\Updownarrow`        |
| $\longleftarrow$      | `\longleftarrow`      | $\Longleftarrow$      | `\Longleftarrow`      |
| $\longrightarrow$     | `\longrightarrow`     | $\Longrightarrow$     | `\Longrightarrow`     |
| $\longleftrightarrow$ | `\longleftrightarrow` | $\Longleftrightarrow$ | `\Longleftrightarrow` |
| $\iff$                | `\iff`                |

## 希腊字母

| 小写字母      | Latex         | 大写字母  | Latex     |
| ------------- | ------------- | --------- | --------- |
| $\alpha$      | `\alpha`      |           |           |
| $\beta$       | `\beta`       |           |           |
| $\gamma$      | `\gamma`      | $\Gamma$  | `\Gamma`  |
| $\delta$      | `\delta`      | $Delta$   | `\Delta`  |
| $\epsilon$    | `\epsilon`    |           |           |
| $\varepsilon$ | `\varepsilon` |           |           |
| $\zeta$       | `\zeta`       |           |           |
| $\eta$        | `\eta`        |           |           |
| $\theta$      | `\theta`      | $\Theta$  | `\Theta`  |
| $\kappa$      | `\kappa`      |           |           |
| $\lambda$     | `\lambda`     | $\Lambda$ | `\Lambda` |
| $\mu$         | `\mu`         |           |           |
| $\nu$         | `\nu`         |           |           |
| $\xi$         | `\xi`         | $\Xi$     | `\Xi`     |
| $\pi$         | `\pi`         | $\Pi$     | `\Pi`     |
| $\rho$        | `\rho`        |           |           |
| $\sigma$      | `\sigma`      | $\Sigma$  | `\Sigma`  |
| $\tau$        | `\tau`        |           |           |
| $\upsilon$    | `\upsilon`    |           |           |
| $\phi$        | `\phi`        | $\Phi$    | `\Phi`    |
| $\varphi$     | `\varphi`     |           |           |
| $\chi$        | `\chi`        |           |           |
| $\psi$        | `\psi`        | $\Psi$    | `\Psi`    |
| $\omega$      | `\omega`      | $\Omega$  | `\Omega`  |

## 其他物理量

| Symbol      | Latex       |
| ----------- | ----------- |
| $\vec{a}$   | `\vec{a}`   |
| $\dagger$   | `\dagger`   |
| $\ast$      | `\ast`      |
| $\bot$      | `\bot`      |
| $\dot{x}$   | `\dot{x}`   |
| $\ddot{x}$  | `\ddot{x}`  |
| $\bar{x}$   | `\bar{x}`   |
| $\ell$      | `\ell`      |
| $\hbar$     | `\hbar`     |
| $^{\circ}C$ | `^{\circ}C` |
| $\langle$   | `\langle`   |
| $\rangle$   | `\rangle`   |

## 集合运算

| Symbol        | Latex         |
| ------------- | ------------- |
| $\mid$ | `\mid` |
| $\in$         | `\in`         |
| $\notin$      | `\notin`      |
| $\ni$         | `\ni`         |
| $\subset$ | `\subset` |
| $\subsetneqq$ | `\subsetneqq` |
| $\supset$ | `\supset` |
| $\supsetneqq$ | `\supsetneqq` |
| $\not\subset$ | `\not\subset` |
| $\cap$ | `\cap` |
| $\cup$ | `\cup` |
| $\overline A$ | `\overline A` |
| $\setminus$ | `\setminus` |
| $\mathbb R$ | `\mathbb R` |
| $\emptyset$ | `\emptyset` |

## 运算符

| Symbol        | Latex         |
| ------------- | ------------- |
| $\pm$         | `\pm`         |
| $\nabla$      | `\nabla`      |
| $\mp$         | `\mp`         |
| $\times$      | `\times`      |
| $\div$        | `\div`        |
| $\oplus$      | `\oplus`      |
| $\otimes$     | `\otimes`     |
| $\bullet$     | `\bullet`     |
| $\le$         | `\le`         |
| $\ge$         | `\ge`         |
| $\ll$         | `\ll`         |
| $\gg$         | `\gg`         |
| $\ne$         | `\ne`         |
| $\propto$     | `\propto`     |
| $\approx$     | `\approx`     |
| $\sim$        | `\sim`        |
| $\simeq$      | `\simeq`      |
| $\cong$       | `\cong`       |
| $\equiv$      | `\equiv`      |
| $a\mod b$     | `a\mod b`     |
| $a\pmod {bc}$ | `a\pmod {bc}` |
| $\cdot$       | `\cdot`       |

## 三角

| Symbol  | Latex   |
| ------- | ------- |
| $\sin$  | `\sin`  |
| $\cos$  | `\cos`  |
| $\tan$  | `\tan`  |
| $\cot$  | `\cot`  |
| $\circ$ | `\circ` |

## 其他符号

| Symbol          | Latex           |
| --------------- | --------------- |
| $\infty$        | `\infty`        |
| $\forall$       | `\forall`       |
| $\exists$       | `\exists`       |
| $\Box$          | `\Box`          |
| $\xcancel{123}$ | `\xcancel{123}` |
