---
title: OpenGL容器运行
date: 2017-09-14 16:03:38
tags: opengl
categories: gpu
---

参考

- [Programming OpenGL in Linux: GLX and Xlib - OpenGL Wiki](https://www.khronos.org/opengl/wiki/Programming_OpenGL_in_Linux:_GLX_and_Xlib)
- [2.1 glutInit](https://www.opengl.org/resources/libraries/glut/spec3/node10.html)
- [2.3 glutInitDisplayMode](https://www.opengl.org/resources/libraries/glut/spec3/node12.html)
- [4.1 glutCreateWindow](https://www.opengl.org/resources/libraries/glut/spec3/node16.html)
- [Ubuntu下使用OpenGL图形库](http://ptbsare.org/2014/05/17/ubuntu%E4%B8%8B%E4%BD%BF%E7%94%A8opengl%E5%9B%BE%E5%BD%A2%E5%BA%93/)
- [glew, glee与 gl glu glut glx glext的区别和关系](http://blog.csdn.net/delacroix_xu/article/details/5881942)
- [freeglut / Bugs / #123 2.4.06 doesn't render on remote X server from ubuntu client.](https://sourceforge.net/p/freeglut/bugs/123/)
- [opengl - freeglut (./light): ERROR: Internal error <FBConfig with necessary capabilities not found> in function fgOpenWindow - Stack Overflow](https://stackoverflow.com/questions/45546693/freeglut-light-error-internal-error-fbconfig-with-necessary-capabilities#comment78056960_45546693)


容器启动

<!-- more -->

```
nvidia-docker run -ti  \
--name liqiang_test \
-v /etc/localtime:/etc/localtime:ro \
--net=host \
-e DISPLAY=:10.0 \
-v $HOME/.Xauthority:/root/.Xauthority \
cuda:8.0 bash
```

安装OpenGL

```
apt-get install -y \
build-essential \
libgl1-mesa-dev libglu1-mesa-dev \
freeglut3-dev \
libglew1.10 libglew-dev libgl1-mesa-glx libxmu-dev \
libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev \
mesa-utils
```


代码

[Light]()

```c
/* light.c
此程序利用GLUT绘制一个OpenGL窗口，并显示一个加以光照的球。
*/
/* 由于头文件glut.h中已经包含了头文件gl.h和glu.h，所以只需要include 此文件*/
# include <GL/glut.h>
# include <stdlib.h>

 /* 初始化材料属性、光源属性、光照模型，打开深度缓冲区 */
void init ( void )
{
    GLfloat mat_specular [ ] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat mat_shininess [ ] = { 50.0 };
    GLfloat light_position [ ] = { 1.0, 1.0, 1.0, 0.0 };
    glClearColor ( 0.0, 0.0, 0.0, 0.0 );
    glShadeModel ( GL_SMOOTH );
    glMaterialfv ( GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv ( GL_FRONT, GL_SHININESS, mat_shininess);
    glLightfv ( GL_LIGHT0, GL_POSITION, light_position);
    glEnable (GL_LIGHTING);
    glEnable (GL_LIGHT0);
    glEnable (GL_DEPTH_TEST);
}
/*调用GLUT函数，绘制一个球*/
void display ( void )
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutSolidSphere (1.0, 40, 50);
    glFlush ();
}

int main(int argc, char** argv)
{
    /* GLUT环境初始化*/
    glutInit (&argc, argv);
    /* 显示模式初始化 */
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    /* 定义窗口大小 */
    glutInitWindowSize (300, 300);
    /* 定义窗口位置 */
    glutInitWindowPosition (100, 100);
    /* 显示窗口，窗口标题为执行函数名 */
    glutCreateWindow ( argv [ 0 ] );
    /* 调用OpenGL初始化函数 */
    init();
    /* 注册OpenGL绘图函数 */
    glutDisplayFunc ( display );
    // /* 进入GLUT消息循环，开始执行程序 */
    glutMainLoop( );
    return 0;
}
```

gcc light.c -o light -lGL -lglut


[Quad](https://www.khronos.org/opengl/wiki/Programming_OpenGL_in_Linux:_GLX_and_Xlib)

```c
// -- Written in C -- //

#include<stdio.h>
#include<stdlib.h>
#include<X11/X.h>
#include<X11/Xlib.h>
#include<GL/gl.h>
#include<GL/glx.h>
#include<GL/glu.h>

Display                 *dpy;
Window                  root;
GLint                   att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
XVisualInfo             *vi;
Colormap                cmap;
XSetWindowAttributes    swa;
Window                  win;
GLXContext              glc;
XWindowAttributes       gwa;
XEvent                  xev;

void DrawAQuad() {
 glClearColor(1.0, 1.0, 1.0, 1.0);
 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 glMatrixMode(GL_PROJECTION);
 glLoadIdentity();
 glOrtho(-1., 1., -1., 1., 1., 20.);

 glMatrixMode(GL_MODELVIEW);
 glLoadIdentity();
 gluLookAt(0., 0., 10., 0., 0., 0., 0., 1., 0.);

 glBegin(GL_QUADS);
  glColor3f(1., 0., 0.); glVertex3f(-.75, -.75, 0.);
  glColor3f(0., 1., 0.); glVertex3f( .75, -.75, 0.);
  glColor3f(0., 0., 1.); glVertex3f( .75,  .75, 0.);
  glColor3f(1., 1., 0.); glVertex3f(-.75,  .75, 0.);
 glEnd();
} 
 
int main(int argc, char *argv[]) {

 dpy = XOpenDisplay(NULL);
 
 if(dpy == NULL) {
        printf("\n\tcannot connect to X server\n\n");
        exit(0);
 }
        
 root = DefaultRootWindow(dpy);

 vi = glXChooseVisual(dpy, 0, att);

 if(vi == NULL) {
        printf("\n\tno appropriate visual found\n\n");
        exit(0);
 } 
 else {
        printf("\n\tvisual %p selected\n", (void *)vi->visualid); /* %p creates hexadecimal output like in glxinfo */
 }


 cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);

 swa.colormap = cmap;
 swa.event_mask = ExposureMask | KeyPressMask;
 
 win = XCreateWindow(dpy, root, 0, 0, 600, 600, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);

 XMapWindow(dpy, win);
 XStoreName(dpy, win, "VERY SIMPLE APPLICATION");
 
 glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
 glXMakeCurrent(dpy, win, glc);
 
 glEnable(GL_DEPTH_TEST); 
 
 while(1) {
        XNextEvent(dpy, &xev);
        
        if(xev.type == Expose) {
                XGetWindowAttributes(dpy, win, &gwa);
                glViewport(0, 0, gwa.width, gwa.height);
                DrawAQuad(); 
                glXSwapBuffers(dpy, win);
        }
                
        else if(xev.type == KeyPress) {
                glXMakeCurrent(dpy, None, NULL);
                glXDestroyContext(dpy, glc);
                XDestroyWindow(dpy, win);
                XCloseDisplay(dpy);
                exit(0);
        }
    } /* this closes while(1) { */
} /* this is the } which closes int main(int argc, char *argv[]) { */
```

gcc -o quad quad.c -lX11 -lGL -lGLU

[球](http://ptbsare.org/2014/05/17/ubuntu%E4%B8%8B%E4%BD%BF%E7%94%A8opengl%E5%9B%BE%E5%BD%A2%E5%BA%93/)

```
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
void init();
void display();
int main(int argc, char* argv[])
{
     glutInit(&argc, argv);
     glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
     glutInitWindowPosition(0, 0);
     glutInitWindowSize(300, 300); 
     glutCreateWindow("OpenGL 3D View");     
     init();
     glutDisplayFunc(display);     
     glutMainLoop();
     return 0;
}
void init()
{
     glClearColor(0.0, 0.0, 0.0, 0.0);
     glMatrixMode(GL_PROJECTION);
     glOrtho(-5, 5, -5, 5, 5, 15);
     glMatrixMode(GL_MODELVIEW);
     gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
}

void display()
{
     glClear(GL_COLOR_BUFFER_BIT);      
     glColor3f(1.0, 0, 0);
     glutWireTeapot(3);  
     glFlush();
}
```

gcc example.c -o example.out -lGL -lGLU -lglut

