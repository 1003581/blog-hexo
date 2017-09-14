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

## 设置

`Prefences`->`Settings`

```
    "default_encoding":"UTF-8", //unix模式
    "default_line_ending":"unix",

    "translate_tabs_to_spaces": false, //
```

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

## pylint

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

## ConvertToUTF8

## SublimeREPL

## SublimeCodeIntel

# VSCode

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

# 字体

[Source Code Pro](https://github.com/adobe-fonts/source-code-pro)
