---
title: nvidia-smi命令详解
date: 2017-09-14 16:03:29
tags: nvidia-smi
categories: gpu
---

## `nvidia-smi`表含义解释

<!-- more -->

`P100`

```  
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 0000:02:00.0     Off |                    0 |
| N/A   37C    P0    28W / 250W |      0MiB / 16276MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    1     24227    C   /usr/local/MATLAB/R2015b/bin/glnxa64/MATLAB    393MiB |
+-----------------------------------------------------------------------------+
```

`Titan X`

```
Thu Aug  3 14:00:03 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  On   | 0000:84:00.0      On |                  N/A |
| 50%   35C    P8    16W / 250W |      9MiB / 12207MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      5734    G   X                                                7MiB |
+-----------------------------------------------------------------------------+

```

- `GPU`: GPU编号, 从0开始, 若有多台GPU, 则依次累加.
- `Fan`: 风扇转速, 从0%到100%, N/A表示该GPU为被动散热,无风扇
- `Name`: GPU名称, 此处为`P100-PCIE...`和`GTX TIT...`
- `Temp`: 温度, 单位摄氏度
- `Perf`: 性能状态, 从P0到P12, P0为最大性能, P12为最小性能
- `Persistence-M`: 持续模式的状态, 持续模式开启时, 会加速GPU应用的启动时间, 但是会增大功耗.
- `Pwr:Usage/Cap`: 能耗, 当前能耗/最大能耗, 单位瓦
- `Bus-Id`: 设计GPU总线的东西, 其格式为`domain:bus:device.function`
- `Disp.A`: Display Active, 表示GPU的显示是否初始化
- `Memory-Usage`: 显存使用情况, 当前使用显存/总显存
- `Volatile GPU-Util`: 浮动的GPU利用率.
- `Uncorr.ECC`: 关于ECC
- `Compute M.`: 计算模式


## `nvidia-smi -q`

```
==============NVSMI LOG==============

Timestamp                           : Thu Aug  3 14:12:26 2017
Driver Version                      : 375.66

Attached GPUs                       : 1
GPU 0000:84:00.0
    Product Name                    : GeForce GTX TITAN X
    Product Brand                   : GeForce
    Display Mode                    : Disabled
    Display Active                  : Enabled
    Persistence Mode                : Enabled
    Accounting Mode                 : Disabled
    Accounting Mode Buffer Size     : 1920
    Driver Model
        Current                     : N/A
        Pending                     : N/A
    Serial Number                   : 0421216034294
    GPU UUID                        : GPU-8e6a16d9-7f97-f238-2fd7-2ab1ed8f5fb1
    Minor Number                    : 0
    VBIOS Version                   : 84.00.45.00.03
    MultiGPU Board                  : No
    Board ID                        : 0x8400
    GPU Part Number                 : N/A
    Inforom Version
        Image Version               : G001.0000.01.03
        OEM Object                  : 1.1
        ECC Object                  : N/A
        Power Management Object     : N/A
    GPU Operation Mode
        Current                     : N/A
        Pending                     : N/A
    GPU Virtualization Mode
        Virtualization mode         : None
    PCI
        Bus                         : 0x84
        Device                      : 0x00
        Domain                      : 0x0000
        Device Id                   : 0x17C210DE
        Bus Id                      : 0000:84:00.0
        Sub System Id               : 0x17C210DE
        GPU Link Info
            PCIe Generation
                Max                 : 3
                Current             : 1
            Link Width
                Max                 : 16x
                Current             : 16x
        Bridge Chip
            Type                    : N/A
            Firmware                : N/A
        Replays since reset         : 0
        Tx Throughput               : 0 KB/s
        Rx Throughput               : 0 KB/s
    Fan Speed                       : 50 %
    Performance State               : P8
    Clocks Throttle Reasons
        Idle                        : Active
        Applications Clocks Setting : Not Active
        SW Power Cap                : Not Active
        HW Slowdown                 : Not Active
        Sync Boost                  : Not Active
        Unknown                     : Not Active
    FB Memory Usage
        Total                       : 12207 MiB
        Used                        : 9 MiB
        Free                        : 12198 MiB
    BAR1 Memory Usage
        Total                       : 256 MiB
        Used                        : 4 MiB
        Free                        : 252 MiB
    Compute Mode                    : Default
    Utilization
        Gpu                         : 0 %
        Memory                      : 1 %
        Encoder                     : 0 %
        Decoder                     : 0 %
    Encoder Stats
        Active Sessions             : 0
        Average FPS                 : 0
        Average Latency             : 0 ms
    Ecc Mode
        Current                     : N/A
        Pending                     : N/A
    ECC Errors
        Volatile
            Single Bit            
                Device Memory       : N/A
                Register File       : N/A
                L1 Cache            : N/A
                L2 Cache            : N/A
                Texture Memory      : N/A
                Texture Shared      : N/A
                Total               : N/A
            Double Bit            
                Device Memory       : N/A
                Register File       : N/A
                L1 Cache            : N/A
                L2 Cache            : N/A
                Texture Memory      : N/A
                Texture Shared      : N/A
                Total               : N/A
        Aggregate
            Single Bit            
                Device Memory       : N/A
                Register File       : N/A
                L1 Cache            : N/A
                L2 Cache            : N/A
                Texture Memory      : N/A
                Texture Shared      : N/A
                Total               : N/A
            Double Bit            
                Device Memory       : N/A
                Register File       : N/A
                L1 Cache            : N/A
                L2 Cache            : N/A
                Texture Memory      : N/A
                Texture Shared      : N/A
                Total               : N/A
    Retired Pages
        Single Bit ECC              : N/A
        Double Bit ECC              : N/A
        Pending                     : N/A
    Temperature
        GPU Current Temp            : 35 C
        GPU Shutdown Temp           : 97 C
        GPU Slowdown Temp           : 92 C
    Power Readings
        Power Management            : Supported
        Power Draw                  : 16.25 W
        Power Limit                 : 250.00 W
        Default Power Limit         : 250.00 W
        Enforced Power Limit        : 250.00 W
        Min Power Limit             : 150.00 W
        Max Power Limit             : 275.00 W
    Clocks
        Graphics                    : 135 MHz
        SM                          : 135 MHz
        Memory                      : 405 MHz
        Video                       : 405 MHz
    Applications Clocks
        Graphics                    : 1000 MHz
        Memory                      : 3505 MHz
    Default Applications Clocks
        Graphics                    : 1000 MHz
        Memory                      : 3505 MHz
    Max Clocks
        Graphics                    : 1392 MHz
        SM                          : 1392 MHz
        Memory                      : 3505 MHz
        Video                       : 1281 MHz
    Clock Policy
        Auto Boost                  : On
        Auto Boost Default          : On
    Processes
        Process ID                  : 5734
            Type                    : G
            Name                    : X
            Used GPU Memory         : 7 MiB


```

## `nvidia-smi --help`

```
NVIDIA System Management Interface -- v375.66

NVSMI provides monitoring information for Tesla and select Quadro devices.
The data is presented in either a plain text or an XML format, via stdout or a file.
NVSMI also provides several management operations for changing the device state.

Note that the functionality of NVSMI is exposed through the NVML C-based
library. See the NVIDIA developer website for more information about NVML.
Python wrappers to NVML are also available.  The output of NVSMI is
not guaranteed to be backwards compatible; NVML and the bindings are backwards
compatible.

http://developer.nvidia.com/nvidia-management-library-nvml/
http://pypi.python.org/pypi/nvidia-ml-py/
Supported products:
- Full Support
    - All Tesla products, starting with the Fermi architecture
    - All Quadro products, starting with the Fermi architecture
    - All GRID products, starting with the Kepler architecture
    - GeForce Titan products, starting with the Kepler architecture
- Limited Support
    - All Geforce products, starting with the Fermi architecture
nvidia-smi [OPTION1 [ARG1]] [OPTION2 [ARG2]] ...

    -h,   --help                Print usage information and exit.

  LIST OPTIONS:

    -L,   --list-gpus           Display a list of GPUs connected to the system.

  SUMMARY OPTIONS:

    <no arguments>              Show a summary of GPUs connected to the system.

    [plus any of]

    -i,   --id=                 Target a specific GPU.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.

  QUERY OPTIONS:

    -q,   --query               Display GPU or Unit info.

    [plus any of]

    -u,   --unit                Show unit, rather than GPU, attributes.
    -i,   --id=                 Target a specific GPU or Unit.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -x,   --xml-format          Produce XML output.
          --dtd                 When showing xml output, embed DTD.
    -d,   --display=            Display only selected information: MEMORY,
                                    UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK,
                                    COMPUTE, PIDS, PERFORMANCE, SUPPORTED_CLOCKS,
                                    PAGE_RETIREMENT, ACCOUNTING, ENCODER STATS 
                                Flags can be combined with comma e.g. ECC,POWER.
                                Sampling data with max/min/avg is also returned 
                                for POWER, UTILIZATION and CLOCK display types.
                                Doesn't work with -u or -x flags.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.

    -lms, --loop-ms=            Probe until Ctrl+C at specified millisecond interval.

  SELECTIVE QUERY OPTIONS:

    Allows the caller to pass an explicit list of properties to query.

    [one of]

    --query-gpu=                Information about GPU.
                                Call --help-query-gpu for more info.
    --query-supported-clocks=   List of supported clocks.
                                Call --help-query-supported-clocks for more info.
    --query-compute-apps=       List of currently active compute processes.
                                Call --help-query-compute-apps for more info.
    --query-accounted-apps=     List of accounted compute processes.
                                Call --help-query-accounted-apps for more info.
    --query-retired-pages=      List of device memory pages that have been retired.
                                Call --help-query-retired-pages for more info.

    [mandatory]

    --format=                   Comma separated list of format options:
                                  csv - comma separated values (MANDATORY)
                                  noheader - skip the first line with column headers
                                  nounits - don't print units for numerical
                                             values

    [plus any of]

    -i,   --id=                 Target a specific GPU or Unit.
    -f,   --filename=           Log to a specified file, rather than to stdout.
    -l,   --loop=               Probe until Ctrl+C at specified second interval.
    -lms, --loop-ms=            Probe until Ctrl+C at specified millisecond interval.

  DEVICE MODIFICATION OPTIONS:

    [any one of]

    -pm,  --persistence-mode=   Set persistence mode: 0/DISABLED, 1/ENABLED
    -e,   --ecc-config=         Toggle ECC support: 0/DISABLED, 1/ENABLED
    -p,   --reset-ecc-errors=   Reset ECC error counts: 0/VOLATILE, 1/AGGREGATE
    -c,   --compute-mode=       Set MODE for compute applications:
                                0/DEFAULT, 1/EXCLUSIVE_PROCESS,
                                2/PROHIBITED
          --gom=                Set GPU Operation Mode:
                                    0/ALL_ON, 1/COMPUTE, 2/LOW_DP
    -r    --gpu-reset           Trigger reset of the GPU.
                                Can be used to reset the GPU HW state in situations
                                that would otherwise require a machine reboot.
                                Typically useful if a double bit ECC error has
                                occurred.
                                Reset operations are not guarenteed to work in
                                all cases and should be used with caution.
                                --id= switch is mandatory for this switch
    -vm   --virt-mode=          Switch GPU Virtualization Mode:
                                Sets GPU virtualization mode to 3/VGPU or 4/VSGA
                                Virtualization mode of a GPU can only be set when
                                it is running on a hypervisor.
    -ac   --applications-clocks= Specifies <memory,graphics> clocks as a
                                    pair (e.g. 2000,800) that defines GPU's
                                    speed in MHz while running applications on a GPU.
    -rac  --reset-applications-clocks
                                Resets the applications clocks to the default values.
    -acp  --applications-clocks-permission=
                                Toggles permission requirements for -ac and -rac commands:
                                0/UNRESTRICTED, 1/RESTRICTED
    -pl   --power-limit=        Specifies maximum power management limit in watts.
    -am   --accounting-mode=    Enable or disable Accounting Mode: 0/DISABLED, 1/ENABLED
    -caa  --clear-accounted-apps
                                Clears all the accounted PIDs in the buffer.
          --auto-boost-default= Set the default auto boost policy to 0/DISABLED
                                or 1/ENABLED, enforcing the change only after the
                                last boost client has exited.
          --auto-boost-permission=
                                Allow non-admin/root control over auto boost mode:
                                0/UNRESTRICTED, 1/RESTRICTED
   [plus optional]

    -i,   --id=                 Target a specific GPU.

  UNIT MODIFICATION OPTIONS:

    -t,   --toggle-led=         Set Unit LED state: 0/GREEN, 1/AMBER

   [plus optional]

    -i,   --id=                 Target a specific Unit.

  SHOW DTD OPTIONS:

          --dtd                 Print device DTD and exit.

     [plus optional]

    -f,   --filename=           Log to a specified file, rather than to stdout.
    -u,   --unit                Show unit, rather than device, DTD.

    --debug=                    Log encrypted debug information to a specified file. 

 STATISTICS: (EXPERIMENTAL)
    stats                       Displays device statistics. "nvidia-smi stats -h" for more information.

 Device Monitoring:
    dmon                        Displays device stats in scrolling format.
                                "nvidia-smi dmon -h" for more information.

    daemon                      Runs in background and monitor devices as a daemon process.
                                This is an experimental feature.
                                "nvidia-smi daemon -h" for more information.

    replay                      Used to replay/extract the persistent stats generated by daemon.
                                This is an experimental feature.
                                "nvidia-smi replay -h" for more information.

 Process Monitoring:
    pmon                        Displays process stats in scrolling format.
                                "nvidia-smi pmon -h" for more information.

 TOPOLOGY:
    topo                        Displays device/system topology. "nvidia-smi topo -h" for more information.

 DRAIN STATES:
    drain                       Displays/modifies GPU drain states for power idling. "nvidia-smi drain -h" for more information.

 NVLINK:
    nvlink                      Displays device nvlink information. "nvidia-smi nvlink -h" for more information.

 CLOCKS:
    clocks                      Control and query clock information. "nvidia-smi clocks -h" for more information.

 ENCODER SESSIONS:
    encodersessions             Displays device encoder sessions information. "nvidia-smi encodersessions -h" for more information.

 GRID vGPU:
    vgpu                        Displays vGPU information. "nvidia-smi vgpu -h" for more information.

Please see the nvidia-smi(1) manual page for more detailed information.
```

## 特殊命令

### 获取GPU的当前温度

依次获得所有GPU卡的温度

```
nvidia-smi -q 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
```

获得指定GPU卡的温度，添加`-g`参数，后加GPU ID，从0开始索引

```
nvidia-smi -q -g 0 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
```

### 获取GPU当前正在运行的进程

依次获得所有GPU卡的进程

```
nvidia-smi -q 2>&1|grep -i "Process ID"|awk '{print $4}'
```

获得指定GPU卡的温度，添加`-g`参数，后加GPU ID，从0开始索引

### 获取GPU当前使用率

```
nvidia-smi -q 2>&1|grep -i "Process ID"|awk '{print $4}'
```

附： 使用python远程监控gpu状态，并返回json格式的数据。或者简单用shell脚本执行返回。

首先需配置SSH免密钥登录，在执行脚本的机器上重复执行

```
cd ~/.ssh
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.42.10.xx
```

python版本`get-gpu-util.py`

```python
#!/usr/bin/python

import paramiko
import json

def ssh_exec(ip, cmd):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(ip)
    stdin, stdout, stderr = client.exec_command(cmd)
    if len(stderr.readlines()):
        output = ["-1"]
    else:
        output =  stdout.readlines()
    client.close()
    return output

command = "nvidia-smi -q|grep \"Gpu\"|awk '{print $3}'"
hosts = ["10.42.10.35", "10.42.10.41", "10.42.10.62"]


result = []
for host in hosts:
    utils = ssh_exec(host, command)
    for index, util in enumerate(utils):
        util = util.strip()
        info = {}
        info["ip"] = host
        info["card"] = index
        info["gpu-util"] = int(util)
        result.append(info)

print(json.dumps(result))
```

输出如下

```
[{"ip": "10.42.10.35", "card": 0, "gpu-util": 0}, {"ip": "10.42.10.41", "card": 0, "gpu-util": 99}, {"ip": "10.42.10.41", "card": 1, "gpu-util": 65}, {"ip": "10.42.10.41", "card": 2, "gpu-util": 85}, {"ip": "10.42.10.41", "card": 3, "gpu-util": 0}, {"ip": "10.42.10.62", "card": 0, "gpu-util": 90}, {"ip": "10.42.10.62", "card": 1, "gpu-util": 33}, {"ip": "10.42.10.62", "card": 2, "gpu-util": 91}, {"ip": "10.42.10.62", "card": 3, "gpu-util": 87}]
```

Shell版本`get-gpu-util.sh`

```bash
#!/bin/bash
command="nvidia-smi -q|grep \"Gpu\"|awk '{print \$3}'|tr '\n' ','"

result35=`ssh 10.42.10.35 $command`
result41=`ssh 10.42.10.41 $command`
result62=`ssh 10.42.10.62 $command`

result=$result35$result41$result62
result1=${result:0:${#result}-1}

echo $result1
```

运行命令`bash get-gpu-util.sh`