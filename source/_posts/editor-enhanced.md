---
title: 各种编辑器的加强功能
date: 2017-09-14 15:58:21
tags: tool
categories: tool
---

# 浏览器插件

<!-- more -->

## Google翻译(中英文)

## [Markdown Preview Plus](https://chrome.google.com/webstore/detail/markdown-preview-plus/febilkbfcbhebfnokafefeacimjdckgl?utm_source=chrome-app-launcher-info-dialog)

## [Github Tree](https://github.com/buunguyen/octotree/blob/master/dist/chrome.crx)

点击view raw下载，然后安装

# Vim

[reference](http://www.cnblogs.com/wangj08/archive/2013/03/13/2957309.html)

`vim ~/.vimrc`

```
syntax on
set number
set tabstop=4
set expandtab
set hlsearch
```

# Sublime3

## 激活

3126激活码

```
—– BEGIN LICENSE —–
Michael Barnes
Single User License
EA7E-821385
8A353C41 872A0D5C DF9B2950 AFF6F667
C458EA6D 8EA3C286 98D1D650 131A97AB
AA919AEC EF20E143 B361B1E7 4C8B7F04
B085E65E 2F5F5360 8489D422 FB8FC1AA
93F6323C FD7F7544 3F39C318 D95E6480
FCCC7561 8A4A1741 68FA4223 ADCEDE07
200C25BE DBBC4855 C4CFB774 C5EC138C
0FEC1CEF D9DCECEC D3A5DAD1 01316C36
—— END LICENSE ——
```

3143激活码

```
—– BEGIN LICENSE —–  
TwitterInc  
200 User License  
EA7E-890007  
1D77F72E 390CDD93 4DCBA022 FAF60790  
61AA12C0 A37081C5 D0316412 4584D136  
94D7F7D4 95BC8C1C 527DA828 560BB037  
D1EDDD8C AE7B379F 50C9D69D B35179EF  
2FE898C4 8E4277A8 555CE714 E1FB0E43  
D5D52613 C3D12E98 BC49967F 7652EED2  
9D2D2E61 67610860 6D338B72 5CF95C69  
E36B85CC 84991F19 7575D828 470A92AB  
—— END LICENSE —— 
```

## 设置

`Prefences`->`Settings`

```
    "font_size": 14,
    "default_encoding":"UTF-8", //unix模式
    "default_line_ending":"unix",

    "translate_tabs_to_spaces": false, //
```

## 在文件夹右键和空格右键处添加用Sublime打开

`Win + R` -> `regedit`打开注册表

### 在资源管理器的当前目录打开

定位到如下：`HKEY_CLASSES_ROOT\Directory\background\shell`

右键点击`shell`，`新建`->`项`，命名为`Sublime`

右键点击`Sublime`，`新建`->`项`，命名为`command`

点击`command`，右侧会出现默认的一条数据，双击`名称`一栏的`(默认)`，修改数值数据为`"C:\Program Files\Sublime Text 3\sublime_text.exe" "-a" "%v"`

### 点击文件夹用Sublime打开

定位到如下：`HKEY_CLASSES_ROOT\Directory\shell`

重复以上内容。

## Markdown Editing

[Reference](http://blog.csdn.net/hfut_jf/article/details/52853868)

`Ctrl+Shift+P`->`Prefences: MarkdownEditing Settings User`

```
{
    "font_size": 13,
    "ignored_packages":
    [
        "Vintage"
    ],

    /*
        Enable or not mathjax support.
    */
    "enable_mathjax": true,

    /*
        Enable or not highlight.js support for syntax highlighting.
    */
    "enable_highlight": true,

    //"color_scheme": "Packages/MarkdownEditing/MarkdownEditor-Dark.tmTheme",
    "color_scheme": "Packages/MarkdownEditing/MarkdownEditor-Yellow.tmTheme",
    //"color_scheme": "Packages/MarkdownEditing/MarkdownEditor.tmTheme",

    // Layout
    "draw_centered": false,
    "wrap_width": 0,

    // Line
    "line_numbers": true,
    "highlight_line": true,
}
```

附：[MarkdownEditing GitHub](https://github.com/SublimeText-Markdown/MarkdownEditing)

## MarkdownPreview

[test.md](https://github.com/revolunet/sublimetext-markdown-preview/blob/master/tests/test.md)

## pylint

[Sublime配置pylinter实现查错,格式化,代码自动规范,对错误显示图标(python语法检查)](https://www.zhaokeli.com/Article/6353.html)

```
{
    // When versbose is 'true', various messages will be written to the console.
    // values: true or false
    "verbose": false,
    // The full path to the Python executable you want to
    // run Pylint with or simply use 'python'.
    "python_bin": "python",
    // The following paths will be added Pylint's Python path
    "python_path": [
        "c:/python27/python.exe"
                   ],
    // Optionally set the working directory
    "working_dir": null,
    // Full path to the lint.py module in the pylint package
    "pylint_path": "c:/python27/Lib/site-packages/pylint/lint.py",
    // Optional full path to a Pylint configuration file
    "pylint_rc": null,
    // Set to true to automtically run Pylint on save
    "run_on_save": true,
    // Set to true to use graphical error icons
    "use_icons": true,
    "disable_outline": false,
    // Status messages stay as long as cursor is on an error line
    "message_stay": true,
    // Ignore Pylint error types. Possible values:
    // "R" : Refactor for a "good practice" metric violation
    // "C" : Convention for coding standard violation
    // "W" : Warning for stylistic problems, or minor programming issues
    // "E" : Error for important programming issues (i.e. most probably bug)
    // "F" : Fatal for errors which prevented further processing
    "ignore": [],
    // a list of strings of individual errors to disable, ex: ["C0301"]
    "disable": [],
    "plugins": []
}
```

## python环境

[blog](http://www.cnblogs.com/jxldjsn/p/6034158.html)

## ConvertToUTF8

## SublimeREPL

## SublimeCodeIntel

## GoSublime & GoImports

# VSCode

## 设置

`Crtl+,`->`用户设置`

```
// Place your settings in this file to overwrite default and user settings.
{
    // 控制字体系列。
    "editor.fontFamily": "Source Code Pro",
    //"workbench.colorTheme": "Seti",
    "extensions.ignoreRecommendations": false,
    "git.enableSmartCommit": true,
    "editor.renderLineHighlight": "none",
    "editor.lineHeight": 24,
    "editor.roundedSelection": false,
    "extensions.autoUpdate": true,
    "editor.fontSize": 14,
    "editor.tabSize": 4,
    "files.associations":{
        "*.ejs": "html",
        "*.wxss": "css"
    },
    "python.linting.pylintEnabled": false
}
```

## 编译C++

[Windows下VSCode编译调试c/c++](http://blog.csdn.net/c_duoduo/article/details/51615381)


# Atom

## Markdown配置

[blog](http://www.cnblogs.com/libin-1/p/6638165.html)

# 字体

[Source Code Pro](https://github.com/adobe-fonts/source-code-pro)

# Markdown相关

[Flow语法](http://blog.csdn.net/KimBing/article/details/52934959?locationNum=2&fps=1)

[序列图和流程图](http://blog.csdn.net/u011729865/article/details/49207455)
