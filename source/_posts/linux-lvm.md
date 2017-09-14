---
title: LVM操作记录
date: 2017-09-14 16:02:41
tags: 
- linux
- lvm
categories: linux
---

# LVM映射

<!-- more -->

## 初始状态

`df -h`

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       189G   43G  136G  25% /
none            4.0K     0  4.0K   0% /sys/fs/cgroup
udev             16G  4.0K   16G   1% /dev
tmpfs           3.2G  1.2M  3.2G   1% /run
none            5.0M     0  5.0M   0% /run/lock
none             16G  348K   16G   1% /run/shm
none            100M     0  100M   0% /run/user
```

`fdisk -l`

```
Disk /dev/sda: 239.5 GB, 239511535616 bytes
255 heads, 63 sectors/track, 29118 cylinders, total 467795968 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disk identifier: 0x00037729

   Device Boot      Start         End      Blocks   Id  System
/dev/sda1   *        2048   401545215   200771584   83  Linux
/dev/sda2       401547262   467793919    33123329    5  Extended
Partition 2 does not start on physical sector boundary.
/dev/sda5       401547264   467793919    33123328   82  Linux swap / Solaris

Disk /dev/sdb: 1199.2 GB, 1199168290816 bytes
255 heads, 63 sectors/track, 145790 cylinders, total 2342125568 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000

Disk /dev/sdb doesn't contain a valid partition table

Disk /dev/sdc: 40002.3 GB, 40002251653120 bytes
255 heads, 63 sectors/track, 4863330 cylinders, total 78129397760 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disk identifier: 0x00000000

Disk /dev/sdc doesn't contain a valid partition table
```

## 安装lvm2,xfs

```
apt-get install lvm2 xfsprogs
```

## 硬盘分区

`parted /dev/sdb`

```
GNU Parted 2.3
Using /dev/sdb
Welcome to GNU Parted! Type 'help' to view a list of commands.
(parted) mklabel                                                          
New disk label type? gpt                                                  
(parted) p                                                                
Model: LSI MR9271-8i (scsi)
Disk /dev/sdb: 1199GB
Sector size (logical/physical): 512B/512B
Partition Table: gpt

Number  Start  End  Size  File system  Name  Flags

(parted) mkpart                                                           
Partition name?  []?                                                      
File system type?  [ext2]? xfs                                            
Start? 1                                                                  
End? -1                                                                   
(parted) p                                                                
Model: LSI MR9271-8i (scsi)
Disk /dev/sdb: 1199GB
Sector size (logical/physical): 512B/512B
Partition Table: gpt

Number  Start   End     Size    File system  Name  Flags
 1      1049kB  1199GB  1199GB

(parted) 
```

`parted /dev/sdc`

```
GNU Parted 2.3
Using /dev/sdc
Welcome to GNU Parted! Type 'help' to view a list of commands.
(parted) mklabel                                                          
New disk label type? gpt                                                  
(parted) mkpart                                                           
Partition name?  []?                                                      
File system type?  [ext2]? xfs
Start? 1                                                                  
End? -1                                                                   
(parted) p                                                                
Model: LSI MR9271-8i (scsi)
Disk /dev/sdc: 40.0TB
Sector size (logical/physical): 512B/4096B
Partition Table: gpt

Number  Start   End     Size    File system  Name  Flags
 1      1049kB  40.0TB  40.0TB

(parted)  
```

此时状态`fdisk -l`

```
Disk /dev/sda: 239.5 GB, 239511535616 bytes
255 heads, 63 sectors/track, 29118 cylinders, total 467795968 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disk identifier: 0x00037729

   Device Boot      Start         End      Blocks   Id  System
/dev/sda1   *        2048   401545215   200771584   83  Linux
/dev/sda2       401547262   467793919    33123329    5  Extended
Partition 2 does not start on physical sector boundary.
/dev/sda5       401547264   467793919    33123328   82  Linux swap / Solaris

WARNING: GPT (GUID Partition Table) detected on '/dev/sdb'! The util fdisk doesn't support GPT. Use GNU Parted.


Disk /dev/sdb: 1199.2 GB, 1199168290816 bytes
255 heads, 63 sectors/track, 145790 cylinders, total 2342125568 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000

   Device Boot      Start         End      Blocks   Id  System
/dev/sdb1               1  2342125567  1171062783+  ee  GPT

WARNING: GPT (GUID Partition Table) detected on '/dev/sdc'! The util fdisk doesn't support GPT. Use GNU Parted.


Disk /dev/sdc: 40002.3 GB, 40002251653120 bytes
255 heads, 63 sectors/track, 4863330 cylinders, total 78129397760 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disk identifier: 0x00000000

   Device Boot      Start         End      Blocks   Id  System
/dev/sdc1               1  4294967295  2147483647+  ee  GPT
Partition 1 does not start on physical sector boundary.
```

## 创建PV

`pvcreate /dev/sdb1`

```
  Physical volume "/dev/sdb1" successfully created
```

`pvcreate /dev/sdc1`

```
  Physical volume "/dev/sdc1" successfully created
```

此时状态`pvdisplay`

```
  "/dev/sdb1" is a new physical volume of "1.09 TiB"
  --- NEW Physical volume ---
  PV Name               /dev/sdb1
  VG Name               
  PV Size               1.09 TiB
  Allocatable           NO
  PE Size               0   
  Total PE              0
  Free PE               0
  Allocated PE          0
  PV UUID               caU0nX-H3jF-X7AU-HwNW-SFU9-oVHk-Rubb0B
   
  "/dev/sdc1" is a new physical volume of "36.38 TiB"
  --- NEW Physical volume ---
  PV Name               /dev/sdc1
  VG Name               
  PV Size               36.38 TiB
  Allocatable           NO
  PE Size               0   
  Total PE              0
  Free PE               0
  Allocated PE          0
  PV UUID               OWy9rl-M7sA-oIx0-IwNZ-dJ5L-dHjj-Y2Ddbo
```

## 创建VG

初始`vgcreate vg61 /dev/sdb1`

```
  Volume group "vg61" successfully created
```

追加`vgextend vg61 /dev/sdc1`

```
  Volume group "vg61" successfully extended
```

此时状态`vgdisplay`

```
  --- Volume group ---
  VG Name               vg61
  System ID             
  Format                lvm2
  Metadata Areas        2
  Metadata Sequence No  2
  VG Access             read/write
  VG Status             resizable
  MAX LV                0
  Cur LV                0
  Open LV               0
  Max PV                0
  Cur PV                2
  Act PV                2
  VG Size               37.47 TiB
  PE Size               4.00 MiB
  Total PE              9823182
  Alloc PE / Size       0 / 0   
  Free  PE / Size       9823182 / 37.47 TiB
  VG UUID               p8E1eC-6frE-2lf1-pai6-50lT-C5Fa-zuvVsB
```

## 创建LV

`lvcreate --name aidata -L 2T vg61`

```
  Logical volume "aidata" created
```

此时状态`lvdisplay`

```
  --- Logical volume ---
  LV Path                /dev/vg61/aidata
  LV Name                aidata
  VG Name                vg61
  LV UUID                407s46-Tv9S-u9Ds-1uWF-dZUm-NfCD-0kUFOL
  LV Write Access        read/write
  LV Creation host, time ubuntu, 2017-07-05 15:15:29 +0800
  LV Status              available
  # open                 0
  LV Size                2.00 TiB
  Current LE             524288
  Segments               1
  Allocation             inherit
  Read ahead sectors     auto
  - currently set to     256
  Block device           252:0
```

格式化`mkfs.xfs /dev/vg61/aidata`

```
meta-data=/dev/vg61/aidata       isize=256    agcount=32, agsize=16777216 blks
         =                       sectsz=4096  attr=2, projid32bit=0
data     =                       bsize=4096   blocks=536870912, imaxpct=5
         =                       sunit=0      swidth=0 blks
naming   =version 2              bsize=4096   ascii-ci=0
log      =internal log           bsize=4096   blocks=262144, version=2
         =                       sectsz=4096  sunit=1 blks, lazy-count=1
realtime =none                   extsz=4096   blocks=0, rtextents=0
```

## 挂载

`mkdir /aidata`

`mount /dev/vg61/aidata /aidata/`

此时状态`df -h`

```
Filesystem               Size  Used Avail Use% Mounted on
/dev/sda1                189G   43G  136G  25% /
none                     4.0K     0  4.0K   0% /sys/fs/cgroup
udev                      16G  4.0K   16G   1% /dev
tmpfs                    3.2G  1.2M  3.2G   1% /run
none                     5.0M     0  5.0M   0% /run/lock
none                      16G  348K   16G   1% /run/shm
none                     100M     0  100M   0% /run/user
/dev/mapper/vg61-aidata  2.0T   71M  1.9T   1% /aidata
```

## 扩展LV

`lvextend -L +18T /dev/vg61/aidata`

```
  Extending logical volume aidata to 20.00 TiB
  Logical volume aidata successfully resized
```

此时状态`lvdisplay`

```
  --- Logical volume ---
  LV Path                /dev/vg61/aidata
  LV Name                aidata
  VG Name                vg61
  LV UUID                407s46-Tv9S-u9Ds-1uWF-dZUm-NfCD-0kUFOL
  LV Write Access        read/write
  LV Creation host, time ubuntu, 2017-07-05 15:15:29 +0800
  LV Status              available
  # open                 1
  LV Size                20.00 TiB
  Current LE             5242880
  Segments               1
  Allocation             inherit
  Read ahead sectors     auto
  - currently set to     256
  Block device           252:0
```

xfs_growfs `xfs_growfs /aidata/` 

其他类型`resize2fs /dev/vg/123`

```
meta-data=/dev/mapper/vg61-aidata isize=256    agcount=32, agsize=16777216 blks
         =                       sectsz=4096  attr=2
data     =                       bsize=4096   blocks=536870912, imaxpct=5
         =                       sunit=0      swidth=0 blks
naming   =version 2              bsize=4096   ascii-ci=0
log      =internal               bsize=4096   blocks=262144, version=2
         =                       sectsz=4096  sunit=1 blks, lazy-count=1
realtime =none                   extsz=4096   blocks=0, rtextents=0
data blocks changed from 536870912 to 5368709120
```

此时状态`df -h`

```
Filesystem               Size  Used Avail Use% Mounted on
/dev/sda1                189G   43G  136G  25% /
none                     4.0K     0  4.0K   0% /sys/fs/cgroup
udev                      16G  4.0K   16G   1% /dev
tmpfs                    3.2G  1.2M  3.2G   1% /run
none                     5.0M     0  5.0M   0% /run/lock
none                      16G  348K   16G   1% /run/shm
none                     100M     0  100M   0% /run/user
/dev/mapper/vg61-aidata   20T   46M   20T   1% /aidata
```

## 永久挂载

