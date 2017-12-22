---
title: TensorFlow 模型持久化
date: 2017-09-25 14:00:00
tags: 
- tensorflow
categories: tensorflow
---

为了让训练结果可以复用，下面介绍如何将训练得到的网络模型持久化。
<!-- more -->

# 代码实现

## tf.train.Saver

有关[`tf.train.Saver`]类的官网文档见[这里](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/Saver)或者[GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py)

## 简单实现

**保存代码**：

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")
```

运行代码后输出

```
'Saved_model/model.ckpt'
```

观察当前文件夹，新生成了`Saved_model`文件夹，其中包含四个文件：

- `checkpoint`：保存了一个目录下所有的模型文件列表。
- `model.ckpt.data-00000-of-00001`：保存了TensorFlow当前参数值。
- `model.ckpt.index`：保存了TensorFlow当前参数名。
- `model.ckpt.meta`：保存了TensorFlow计算图的结构。

**加载代码**：

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result))
```

输出如下：

```
INFO:tensorflow:Restoring parameters from Saved_model/model.ckpt
[ 3.]
```

该代码首先定义了Tensorflow计算图中的所有运算结构，然后从本地文件中读入变量的值，不需要初始化变量。

## 加载持久化的图

若我们不希望代码中再次定义所有的结构，则可以加载已经保存了的图结构。代码如下：

```python
import tensorflow as tf

saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
```

输出如下

```
INFO:tensorflow:Restoring parameters from Saved_model/model.ckpt
[ 3.]
```

> 上述所有代码，默认保存和加载了TensorFlow计算图中定义的全部变量。

## 保存指定变量

保存代码

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

saver = tf.train.Saver([v1])

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result))
```

> 上述程序会出错，报错信息如下：
> ```
> tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value v2
> ```

## 读取时对变量重命名

保存代码如下：

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")
```

利用字典来重命名变量，key为结构图中的变量name，value为本地变量。加载代码如下：

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})
```

## 保存和加载滑动平均模型

### 使用变量重命名方式

保存代码如下：

```python
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print(variables.name)
    
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))
```

输出如下

```
v:0
v:0
v/ExponentialMovingAverage:0
[10.0, 0.099999905]
```

加载代码，因为滑动平均模型的特性，读取变量v的值，实际是要读取变量v的滑动平均值。

```python
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v))
```

输出如下

```
INFO:tensorflow:Restoring parameters from Saved_model/model2.ckpt
0.0999999
```

### 使用variables_to_restore

为了方便加载时重命名滑动平均变量，`tf.train.ExponentialMovingAverage`类提供了`variables_to_restore`([Docs](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/ExponentialMovingAverage#variables_to_restore),[Github](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/moving_averages.py))函数来生成`tf.train.Saver`类所需要的变量重命名字典。

代码如下：

```python
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v))
```

输出如下：

```
{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
INFO:tensorflow:Restoring parameters from Saved_model/model2.ckpt
0.0999999
```

## PB文件保存

保存

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
           f.write(output_graph_def.SerializeToString())
```

> `graph_def = tf.get_default_graph().as_graph_def()`：导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程。
> 
> `graph_util.convert_variables_to_constants`：将图中的变量和取值转化为常量。
> 
> 此时只生成了一个文件`combined_model.pb`。


输出

```
INFO:tensorflow:Froze 2 variables.
Converted 2 variables to const ops.
```

加载代码

```python
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "Saved_model/combined_model.pb"
   
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
```

输出

```
[array([ 3.], dtype=float32)]
```

# 持久化原理和数据格式

TensorFlow保存的文件为Protocol Buffer形式的。下面首页介绍这种格式的文件。

## Protocol Buffer

Protocol Buffer是Google开发的处理结构化数据的工具。类似的还有XML、JSON。

比如需要保存以下的一些结构化信息：

```
name: 张三
id: 12345
email: zhangsan@abc.com
```

XML保存：

```xml
<user>
    <name>张三</name>
    <id>12345</id>
    <email>zhangsan@abc.com</email>
</user>
```

JSON保存

```json
{
    "name": "张三",
    "id": "12345",
    "email": "zhangsan@abc.com",
}
```

Protocol Buffer与这两者的区别：

- XML和JSON格式的数据，序列化后为可读的字符串，该字符串中包含所有信息。
- Protocol Buffer序列化后为不可读的二进制流，使用Protocol Buffer需先定义数据的格式（schema），还原数据时也需要相应的格式。
- Protocol Buffer序列化后的数据比XML或JSON小3到10倍，解析时间快20到100倍。

格式schema文件定义如下：

```
message user{
    optional string name = 1;
    required int32 id = 2;
    repeated string email = 3;
}
```

## .ckpt.meta —— MetaGraphDef

TensorFlow是一个通过图的形式来表述计算的编程系统，TensorFlow程序中的所有计算都会被表达为计算图上的节点。TensorFlow通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。

类型定义如下，详见[Github](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)：

```
message MetaGraphDef{
    MetaInfoDef meta_info_def = 1;
    GraphDef graph_def = 2;
    SaverDef saver_def = 3;
    map<string, CollectionDef> collection_def = 4;
    map<string, SignatureDef> signature_def = 5;
    repeated AssetFileDef asset_file_def = 6;
}
```

以上信息都保存在了`model.ckpt.meta`文件中，此为二进制文件，无法直接查看。为了方便调试，TensorFlow提供了`export_meta_graph`函数，支持以Json格式导出Protocol Buffer。代码如下

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result1 = v1 + v2

saver = tf.train.Saver()
saver.export_meta_graph("Saved_model/model.ckpt.meta.json", as_text=True)
```

查看Json文件

```
meta_info_def {
  stripped_op_list {
    op {
      name: "Add"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_UINT8
            type: DT_INT8
            type: DT_INT16
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_STRING
          }
        }
      }
    }
    op {
      name: "Assign"
      input_arg {
        name: "ref"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output_ref"
        type_attr: "T"
        is_ref: true
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "validate_shape"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "use_locking"
        type: "bool"
        default_value {
          b: true
        }
      }
      allows_uninitialized_input: true
    }
    op {
      name: "Const"
      output_arg {
        name: "output"
        type_attr: "dtype"
      }
      attr {
        name: "value"
        type: "tensor"
      }
      attr {
        name: "dtype"
        type: "type"
      }
    }
    op {
      name: "Identity"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "NoOp"
    }
    op {
      name: "RestoreV2"
      input_arg {
        name: "prefix"
        type: DT_STRING
      }
      input_arg {
        name: "tensor_names"
        type: DT_STRING
      }
      input_arg {
        name: "shape_and_slices"
        type: DT_STRING
      }
      output_arg {
        name: "tensors"
        type_list_attr: "dtypes"
      }
      attr {
        name: "dtypes"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
      is_stateful: true
    }
    op {
      name: "SaveV2"
      input_arg {
        name: "prefix"
        type: DT_STRING
      }
      input_arg {
        name: "tensor_names"
        type: DT_STRING
      }
      input_arg {
        name: "shape_and_slices"
        type: DT_STRING
      }
      input_arg {
        name: "tensors"
        type_list_attr: "dtypes"
      }
      attr {
        name: "dtypes"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
      is_stateful: true
    }
    op {
      name: "VariableV2"
      output_arg {
        name: "ref"
        type_attr: "dtype"
        is_ref: true
      }
      attr {
        name: "shape"
        type: "shape"
      }
      attr {
        name: "dtype"
        type: "type"
      }
      attr {
        name: "container"
        type: "string"
        default_value {
          s: ""
        }
      }
      attr {
        name: "shared_name"
        type: "string"
        default_value {
          s: ""
        }
      }
      is_stateful: true
    }
  }
  tensorflow_version: "1.3.0"
  tensorflow_git_version: "v1.3.0-rc2-20-g0787eee"
}
graph_def {
  node {
    name: "Const"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
          }
          float_val: 1.0
        }
      }
    }
  }
  node {
    name: "v1"
    op: "VariableV2"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "v1/Assign"
    op: "Assign"
    input: "v1"
    input: "Const"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "v1/read"
    op: "Identity"
    input: "v1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "Const_1"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
          }
          float_val: 2.0
        }
      }
    }
  }
  node {
    name: "v2"
    op: "VariableV2"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "v2/Assign"
    op: "Assign"
    input: "v2"
    input: "Const_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v2"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "v2/read"
    op: "Identity"
    input: "v2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v2"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "add"
    op: "Add"
    input: "v1/read"
    input: "v2/read"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
  }
  node {
    name: "save/Const"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
          }
          string_val: "model"
        }
      }
    }
  }
  node {
    name: "save/SaveV2/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 2
            }
          }
          string_val: "v1"
          string_val: "v2"
        }
      }
    }
  }
  node {
    name: "save/SaveV2/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 2
            }
          }
          string_val: ""
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/SaveV2"
    op: "SaveV2"
    input: "save/Const"
    input: "save/SaveV2/tensor_names"
    input: "save/SaveV2/shape_and_slices"
    input: "v1"
    input: "v2"
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/control_dependency"
    op: "Identity"
    input: "save/Const"
    input: "^save/SaveV2"
    attr {
      key: "T"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@save/Const"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "save/RestoreV2/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "v1"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2/tensor_names"
    input: "save/RestoreV2/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign"
    op: "Assign"
    input: "v1"
    input: "save/RestoreV2"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v1"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/RestoreV2_1/tensor_names"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: "v2"
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_1/shape_and_slices"
    op: "Const"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_STRING
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_STRING
          tensor_shape {
            dim {
              size: 1
            }
          }
          string_val: ""
        }
      }
    }
  }
  node {
    name: "save/RestoreV2_1"
    op: "RestoreV2"
    input: "save/Const"
    input: "save/RestoreV2_1/tensor_names"
    input: "save/RestoreV2_1/shape_and_slices"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            unknown_rank: true
          }
        }
      }
    }
    attr {
      key: "dtypes"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
  }
  node {
    name: "save/Assign_1"
    op: "Assign"
    input: "v2"
    input: "save/RestoreV2_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@v2"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "save/restore_all"
    op: "NoOp"
    input: "^save/Assign"
    input: "^save/Assign_1"
  }
  versions {
    producer: 24
  }
}
saver_def {
  filename_tensor_name: "save/Const:0"
  save_tensor_name: "save/control_dependency:0"
  restore_op_name: "save/restore_all"
  max_to_keep: 5
  keep_checkpoint_every_n_hours: 10000.0
  version: V2
}
collection_def {
  key: "trainable_variables"
  value {
    bytes_list {
      value: "\n\004v1:0\022\tv1/Assign\032\tv1/read:0"
      value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
    }
  }
}
collection_def {
  key: "variables"
  value {
    bytes_list {
      value: "\n\004v1:0\022\tv1/Assign\032\tv1/read:0"
      value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
    }
  }
}
```

### meta_info_def属性

保存了Tensorflow计算图中的元数据和程序中所有用到的运算方法的信息。

定义如下：

```
message MetaInfoDef {
    string meta_graph_version = 1;
    OpList stripped_op_list = 2;
    google.protobuf.Any any_info = 3;
    repeated string tags = 4;
    string tensorflow_version = 5;
    string tensorflow_git_version = 6;
```

`OpList`定义见[Github](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)

在OpDef中的attr属性中，必须包含name为T的属性，指定了运算输入输出允许的参数类型。

### graph_def

[GraphDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)、[NodeDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto)

主要记录计算图上的节点信息。

### saver_def

[SaverDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saver.proto)

主要记录持久化模型时需要用到的一些参数，比如保存到文件的文件名、保存操作和加载操作的名称以及保存频率、清理历史纪录等。

### collection_def

[CollectionDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto#34)

维护不同的集合，是一个从集合名称到集合内容的映射。

## .ckpt

TensorFlow采用`tf.train.NewCheckpointReader`来读取ckpt文件中的所有变量信息。

```python
import tensorflow as tf

reader = tf.train.NewCheckpointReader("Saved_model/model.ckpt")

all_variables = reader.get_variable_to_shape_map()

for variable_name in all_variables:
    print(variable_name, all_variables[variable_name])

print("Value for variable v1 is ", reader.get_tensor("v1"))
```

> `tf.train.NewCheckpointReader`读取ckpt文件中的所有变量。
> `variable_name`为变量名称
> `all_variables[variable_name]`为变量维度

输出如下：

```
v2 [1]
v1 [1]
Value for variable v1 is  [ 1.]
```

## checkpoint

`tf.train.Saver`类自动生成且维护，记录所有Tensorflow模型文件的文件名。**可读**。

格式如下：

```
message CheckpointState{
    string model_checkpoint_path = 1;
    repeated string all_model_checkpoint_paths = 2;
}
```

实例如下：

```
model_checkpoint_path: "model.ckpt"
all_model_checkpoint_paths: "model.ckpt"
```
