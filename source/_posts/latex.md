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

| 符号        | Latex       |
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

## 运算符

| 符号      | Latex     |
| --------- | --------- |
| $\pm$     | `\pm`     |
| $\nabla$  | `\nabla`  |
| $\mp$     | `\mp`     |
| $\times$  | `\times`  |
| $\div$    | `\div`    |
| $\oplus$  | `\oplus`  |
| $\otimes$ | `\otimes` |
| $\bullet$ | `\bullet` |
| $\le$     | `\le`     |
| $\ge$     | `\ge`     |
| $\ll$     | `\ll`     |
| $\gg$     | `\gg`     |
| $\ne$     | `\ne`     |
| $\propto$  | `\propto`  |
| $\approx$  | `\approx`  |
| $\sim$  | `\sim`  |
| $\simeq$  | `\simeq`  |
| $\in$  | `\in`  |
| $\ni$  | `\ni`  |
| $\equiv$  | `\equiv`  |
| $\cdot$  | `\cdot`  |

## 其他符号

| 符号      | Latex     |
| --------- | --------- |
| $\infty$  | `\infty`  |
| $\forall$  | `\forall`  |
| $\exists$  | `\exists`  |
| $\rightarrow$  | `\rightarrow`  |
| $\Rightarrow$  | `\Rightarrow`  |
| $\uparrow$  | `\uparrow`  |
| $\downarrow$  | `\downarrow`  |
| $\Box$  | `\Box`  |
