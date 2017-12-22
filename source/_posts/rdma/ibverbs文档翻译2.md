---
title: ibverbs文档翻译2
date: 2017-09-14 15:58:43
tags: 
- rdma
- ibverbs
categories: c++
---

[百度云下载](https://pan.baidu.com/s/1o78iHmi)

<!-- more -->
## 3 VPI Verbs API

### 3.1 Initialization

#### 3.1.1 ibv_fork_init

    int ibv_fork_init(void)

    Input Parameters:
    None
    Output Parameters:
    None
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

- ibv_fork_init初始化libibverbs的数据结构以安全地处理fork()函数，并避免数据损坏，不管fork()是否被明确地或隐式地调用，如在system()调用。    
- 如果所有父进程线程总是被阻塞，直到所有子进程通过exec()操作结束或更改地址空间，则不需要调用ibv_fork_init。  
- 该功能适用​​于支持madvise()（2.6.17及更高版本）的MADV_DONTFORK标志的Linux内核。  
- 将环境变量RDMAV_FORK_SAFE或IBV_FORK_SAFE设置为任何值与调用ibv_fork_init()具有相同的效果。  
- 将环境变量RDMAV_HUGEPAGES_SAFE设置为任何值将告知库检查内核对内存区域使用的底层页大小。如果应用程序通过库（例如libhugetlbfs）直接或间接使用庞大的页面，这是必需的。  
- 调用ibv_fork_init()将会降低性能，因为每个内存注册需要额外的系统调用，并且分配给追踪内存区域的额外内存。精确的性能影响取决于工作负载，通常不会很大。    
- 设置RDMAV_HUGEPAGES_SAFE为所有内存启用增加了额外的开销。

### 3.2 Device Operations

以下命令用于常规设备操作，允许用户查询有关系统上的设备以及打开和关闭特定设备的信息。

#### 3.2.1 ibv_get_device_list

    struct ibv_device **ibv_get_device_list (int *num_devices)

    Input Parameters:
    none
    Output Parameters:
    num_devices     (optional) If non-null, the number of devices returned in the array will be stored here
    Return Value:
    NULL terminated array of VPI devices or NULL on failure.

ibv_get_device_list返回系统上可用的VPI设备列表。 列表中的每个条目都是一个指向struct ibv_device的指针。

struct ibv_device结构体定义如下：

    struct ibv_device
    {
        struct ibv_device_ops ops;
        enum ibv_node_type node_type;
        enum ibv_transport_type transport_type;
        char name[IBV_SYSFS_NAME_MAX];
        char dev_name[IBV_SYSFS_NAME_MAX];
        char dev_path[IBV_SYSFS_PATH_MAX];
        char ibdev_path[IBV_SYSFS_PATH_MAX];
    };

| variable       | description                              | example                             |
| -------------- | ---------------------------------------- | ----------------------------------- |
| ops            | pointers to alloc and free functions     |                                     |
| node_type      | IBV_NODE_UNKNOWN<br>IBV_NODE_CA<br>IBV_NODE_SWITCH<br>IBV_NODE_ROUTER<br>IBV_NODE_RNIC | InfiniBand channel adapter          |
| transport_type | IBV_TRANSPORT_UNKNOWN<br>IBV_TRANSPORT_IB<br>IBV_TRANSPORT_IWARP | 0                                   |
| name           | kernel device name eg “mthca0”           | mlx5_0                              |
| dev_name       | uverbs device name eg “uverbs0”          | uverbs0                             |
| dev_path       | path to infiniband_verbs class device in sysfs | /sys/class/infiniband_verbs/uverbs0 |
| ibdev_path     | path to infiniband class device in sysfs | /sys/class/infiniband/mlx5_0        |

ibv_device结构列表将保持有效，直到列表被释放。 调用ibv_get_device_list后，用户应该打开所需的设备，并通过ibv_free_device_list命令及时释放列表。

#### 3.2.2 ibv_free_device_list

    void ibv_free_device_list(struct ibv_device **list)

    Input Parameters:
    list        list of devices provided from ibv_get_device_list command
    Output Parameters:
    none
    Return Value:
    none

ibv_free_device_list释放由ibv_get_device_list提供的ibv_device结构列表。 在调用此命令之前，应打开任何所需的设备。 一旦列表被释放，列表中的所有ibv_device结构都将无效，不能再使用。

#### 3.2.3 ibv_get_device_name

    const char *ibv_get_device_name(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    Pointer to device name char string or NULL on failure.

ibv_get_device_name返回了ibv_device结构体中的设备名称的指针。

#### 3.2.4 ibv_get_device_guid

    uint64_t ibv_get_device_guid(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    64 bit GUID

ibv_get_device_guid以网络字节顺序返回设备64位全局唯一标识符（GUID）。

#### 3.2.5 ibv_open_device

    struct ibv_context *ibv_open_device(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    A verbs context that can be used for future operations on the device or NULL on failure.

ibv_open_device为用户提供了一个verbs context，它将是所有的verbs操作的对象。

#### 3.2.6 ibv_close_device

    int ibv_close_device(struct ibv_context *context)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_close_device关闭以前使用ibv_open_device打开的verbs context。 此操作不会释放与context相关联的任何其他对象。 为了避免内存泄漏，所有其他对象必须在调用此命令之前独立释放。

#### 3.2.7 ibv_node_type_str

    const char *ibv_node_type_str (enum ibv_node_type node_type)

    Input Parameters:
    node_type   ibv_node_type enum value which may be an HCA, Switch, Router, RNIC or Unknown
    Output Parameters:
    none
    Return Value:
    A constant string which describes the enum value node_type

ibv_node_type_str返回描述节点类型枚举值node_type的字符串。 该值可以是InfiniBand HCA，交换机，路由器，启用RDMA的NIC或未知。

    enum ibv_node_type {
        IBV_NODE_UNKNOWN = -1,
        IBV_NODE_CA = 1,
        IBV_NODE_SWITCH,
        IBV_NODE_ROUTER,
        IBV_NODE_RNIC
    };

#### 3.2.8 ibv_port_state_str

    const char *ibv_port_state_str (enum ibv_port_state port_state)

    Input Parameters:
    port_state The enumerated value of the port state Output Parameters:
    None
    Return Value:
    A constant string which describes the enum value port_state

ibv_port_state_str返回一个描述端口状态枚举值port_state的字符串。

    enum ibv_port_state {
        IBV_PORT_NOP = 0,
        IBV_PORT_DOWN = 1,
        IBV_PORT_INIT = 2,
        IBV_PORT_ARMED = 3,
        IBV_PORT_ACTIVE = 4,
        IBV_PORT_ACTIVE_DEFER = 5
    };

### 3.3 Verb Context Operations

一旦打开设备，就会使用以下命令。 这些命令允许您获取有关设备或其端口之一的更具体信息，创建可用于进一步操作的完成队列（CQ），完成通道（CC）和保护域（PD）。

#### 3.3.1 ibv_query_device

    int ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr)

    Input Parameters:
    context         struct ibv_context from ibv_open_device 
    Output Parameters:
    device_attr     struct ibv_device_attr containing device attributes
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_device检索与设备关联的各种属性。 用户负责对struct ibv_device_attr进行空间分配和释放。其值将会被填写，如果函数成功返回。

结构体定义如下：

    struct ibv_device_attr
    {
        char        fw_ver[64];
        uint64_t    node_guid;
        uint64_t    sys_image_guid;
        uint64_t    max_mr_size;
        uint64_t    page_size_cap;
        uint32_t    vendor_id;
        uint32_t    vendor_part_id;
        uint32_t    hw_ver;
        int         max_qp;
        int         max_qp_wr;
        int         device_cap_flags;
        int         max_sge;
        int         max_sge_rd;
        int         max_cq;
        int         max_cqe;
        int         max_mr;
        int         max_pd;
        int         max_qp_rd_atom;
        int         max_ee_rd_atom;
        int         max_res_rd_atom;
        int         max_qp_init_rd_atom;
        int         max_ee_init_rd_atom;
        enum ibv_atomic_cap atomic_cap;
        int         max_ee;
        int         max_rdd;
        int         max_mw;
        int         max_raw_ipv6_qp;
        int         max_raw_ethy_qp;
        int         max_mcast_grp;
        int         max_mcast_qp_attach;
        int         max_total_mcast_qp_attach;
        int         max_ah;
        int         max_fmr;
        int         max_map_per_fmr;
        int         max_srq;
        int         max_srq_wr;
        int         max_srq_sge;
        uint16_t    max_pkeys;
        uint8_t     local_ca_ack_delay;
        uint8_t     phys_port_cnt;
    }

| variable                  | description                              | example              |
| ------------------------- | ---------------------------------------- | -------------------- |
| fw_ver                    | Firmware version                         | 12.18.1000           |
| node_guid                 | Node global unique identifier (GUID)     | 9814281471509629476  |
| sys_image_guid            | System image GUID                        | 9814281471509629476  |
| max_mr_size               | Largest contiguous block that can be registered | 18446744073709551615 |
| page_size_cap             | Supported page sizes                     | 18446744073709547520 |
| vendor_id                 | Vendor ID, per IEEE                      | 713                  |
| vendor_part_id            | Vendor supplied part ID                  | 4115                 |
| hw_ver                    | Hardware version                         | 0                    |
| max_qp                    | Maximum number of Queue Pairs (QP)       | 131072               |
| max_qp_wr                 | Maximum outstanding work requests (WR) on any queue | 32768                |
| device_cap_flags          | IBV_DEVICE_RESIZE_MAX_WR<br>IBV_DEVICE_BAD_PKEY_CNTR<br>IBV_DEVICE_BAD_QKEY_CNTR<br>IBV_DEVICE_RAW_MULTI<br>IBV_DEVICE_AUTO_PATH_MIG<br>IBV_DEVICE_CHANGE_PHY_PORT<br>IBV_DEVICE_UD_AV_PORT_ENFORCE<br>IBV_DEVICE_CURR_QP_STATE_MOD<br>IBV_DEVICE_SHUTDOWN_PORT<br>IBV_DEVICE_INIT_TYPE<br>IBV_DEVICE_PORT_ACTIVE_EVENT<br>IBV_DEVICE_SYS_IMAGE_GUID<br>IBV_DEVICE_RC_RNR_NAK_GEN<br>IBV_DEVICE_SRQ_RESIZE<br>IBV_DEVICE_N_NOTIFY_CQ<br>IBV_DEVICE_XRC | -914482122           |
| max_sge                   | Maximum scatter/gather entries (SGE) per WR for non-RD QPs | 30                   |
| max_sge_rd                | Maximum SGEs per WR for RD QPs           | 30                   |
| max_cq                    | Maximum supported completion queues (CQ) | 16777216             |
| max_cqe                   | Maximum completion queue entries (CQE) per CQ | 4194303              |
| max_mr                    | Maximum supported memory regions (MR)    | 16777216             |
| max_pd                    | Maximum supported protection domains (PD) | 16777216             |
| max_qp_rd_atom            | Maximum outstanding RDMA read and atomic operations per QP | 16                   |
| max_ee_rd_atom            | Maximum outstanding RDMA read and atomic operations per End to End (EE) context (RD connections) | 0                    |
| max_res_rd_atom           | Maximum resources used for incoming RDMA read and atomic operations | 2097152              |
| max_qp_init_rd_atom       | Maximium RDMA read and atomic operations that may be initiated per QP | 16                   |
| max_ee_init_atom          | Maximum RDMA read and atomic operations that may be initiated per EE | 0                    |
| atomic_cap                | IBV_ATOMIC_NONE - no atomic guarantees<br>IBV_ATOMIC_HCA - atomic guarantees within this device<br>IBV_ATOMIC_GLOB - global atomic guarantees | 1                    |
| max_ee                    | Maximum supported EE contexts            | 0                    |
| max_rdd                   | Maximum supported RD domains             | 0                    |
| max_mw                    | Maximum supported memory windows (MW)    | 16777216             |
| max_raw_ipv6_qp           | Maximum supported raw IPv6 datagram QPs  | 0                    |
| max_raw_ethy_qp           | Maximum supported ethertype datagram QPs | 0                    |
| max_mcast_grp             | Maximum supported multicast groups       | 2097152              |
| max_mcast_qp_attach       | Maximum QPs per multicast group that can be attached | 48                   |
| max_total_mcast_qp_attach | Maximum total QPs that can be attached to multicast groups | 100663296            |
| max_ah                    | Maximum supported address handles (AH)   | 2147483647           |
| max_fmr                   | Maximum supported fast memory regions (FMR) | 0                    |
| max_map_per_fmr           | Maximum number of remaps per FMR before an unmap operation is required | 2147483647           |
| max_srq                   | Maximum supported shared receive queues (SRCQ) | 8388608              |
| max_srq_wr                | Maximum work requests (WR) per SRQ       | 32767                |
| max_srq_sge               | Maximum SGEs per SRQ                     | 31                   |
| max_pkeys                 | Maximum number of partitions             | 128                  |
| local_ca_ack_delay        | Local CA ack delay                       | 16                   |
| phys_port_cnt             | Number of physical ports                 | 1                    |

#### 3.3.2 ibv_query_port

    int ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    port_num        physical port number (1 is first port)
    Output Parameters:
    port_attr       struct ibv_port_attr containing port attributes
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_port检索与端口关联的各种属性。 用户应该分配一个结构体ibv_port_attr，将其传递给函数，并在成功返回时填写它。 用户有责任释放这个结构。

结构体定义如下：

    struct ibv_port_attr
    {
        enum ibv_port_state     state;
        enum ibv_mtu            max_mtu;
        enum ibv_mtu            active_mtu;
        int                     gid_tbl_len;
        uint32_t                port_cap_flags;
        uint32_t                max_msg_sz;
        uint32_t                bad_pkey_cntr;
        uint32_t                qkey_viol_cntr;
        uint16_t                pkey_tbl_len;
        uint16_t                lid;
        uint16_t                sm_lid;
        uint8_t                 lmc;
        uint8_t                 max_vl_num;
        uint8_t                 sm_sl;
        uint8_t                 subnet_timeout;
        uint8_t                 init_type_reply;
        uint8_t                 active_width;
        uint8_t                 active_speed;
        uint8_t                 phys_state;
    };

| variable        | description                              |
| --------------- | ---------------------------------------- |
| state           | IBV_PORT_NOP<br>IBV_PORT_DOWN<br>IBV_PORT_INIT<br>IBV_PORT_ARMED<br>IBV_PORT_ACTIVE<br>IBV_PORT_ACTIVE_DEFER |
| max_mtu         | Maximum Transmission Unit (MTU) supported by port. Can be:<br>IBV_MTU_256<br>IBV_MTU_512<br>IBV_MTU_1024<br>IBV_MTU_2048<br>IBV_MTU_4096 |
| active_mtu      | Actual MTU in use                        |
| gid_tbl_len     | Length of source global ID (GID) table   |
| port_cap_flags  | Supported capabilities of this port. There are currently no enumerations/defines declared in verbs.h |
| max_msg_sz      | Maximum message size                     |
| bad_pkey_cntr   | Bad P_Key counter                        |
| qkey_viol_cntr  | Q_Key violation counter                  |
| pkey_tbl_len    | Length of partition table                |
| lid First local | identifier (LID) assigned to this port   |
| sm_lid          | LID of subnet manager (SM)               |
| lmc LID         | Mask control (used when multiple LIDs are assigned to port) |
| max_vl_num      | Maximum virtual lanes (VL)               |
| sm_sl           | SM service level (SL)                    |
| subnet_timeout  | Subnet propagation delay                 |
| init_type_reply | Type of initialization performed by SM   |
| active_width    | Currently active link width              |
| active_speed    | Currently active link speed              |
| phys_state      | Physical port state                      |

#### 3.3.3 ibv_query_gid

    int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    port_num        physical port number (1 is first port)
    index           which entry in the GID table to return (0 is first)
    Output Parameters:
    gid             union ibv_gid containing gid information
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_gid检索端口全局标识符（GID）表中的条目。 每个端口由子网管理器（SM）分配至少一个GID。 GID是由全局唯一标识符（GUID）和由SM分配的前缀组成的有效IPv6地址。 GID [0]是唯一的，包含端口的GUID。  
用户应该分配一个联合ibv_gid，将其传递给命令，并在成功返回时填写它。 用户有责任释放这个联合。  
union ibv_gid定义如下：  

    union ibv_gid
    {
        uint8_t         raw[16];
        struct
        {
            uint64_t    subnet_prefix;
            uint64_t    interface_id;
        } global;
    }

#### 3.3.4 ibv_query_pkey

    int ibv_query_pkey(struct ibv_context *context, uint8_t port_num, int index, uint16_t *pkey)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    port_num    physical port number (1 is first port)
    index       which entry in the pkey table to return (0 is first)
    Output Parameters:
    pkey        desired pkey
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_pkey检索端口分区密钥（pkey）表中的条目。 每个端口由子网管理器（SM）分配至少一个pkey。 pkey标识端口所属的分区。 pkey类似于以太网中的VLAN ID。  
用户将一个指针传递给一个将使用请求的pkey填充的uint16。 用户有责任释放这个uint16。

#### 3.3.5 ibv_alloc_pd

    struct ibv_pd *ibv_alloc_pd(struct ibv_context *context)

    Input Parameters:
    context struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    Pointer to created protection domain or NULL on failure.

ibv_alloc_pd创建一个保护域（PD）。 PD限制可以通过哪些队列对（QP）提供一定程度的保护以防止未经授权的访问来访问哪些存储器区域。  
用户必须创建至少一个PD才能使用VPI verbs。

#### 3.3.6 ibv_dealloc_pd

    int ibv_dealloc_pd(struct ibv_pd *pd)

    Input Parameters:
    pd      struct ibv_pd from ibv_alloc_pd
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_dealloc_pd释放保护域（PD）。 如果任何其他对象当前与指定的PD相关联，则此命令将失败。

#### 3.3.7 ibv_create_cq

    struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    cqe         Minimum number of entries CQ will support
    cq_context  (Optional) User defined value returned with completion events
    channel     (Optional) Completion channel
    comp_vector (Optional) Completion vector
    Output Parameters:
    none
    Return Value:
    pointer to created CQ or NULL on failure.

ibv_create_cq创建一个完成队列（CQ）。完成队列保存完成队列条目（CQE）。每个队列对（QP）具有相关联的发送和接收CQ。单个CQ可以共享用于发送和接收，并且可以跨多个QP共享。  
参数cqe定义队列的最小大小。队列的实际大小可能大于指定的值。  
参数cq_context是用户定义的值。如果在创建CQ时指定，则在使用完成通道（CC）时，该值将作为ibv_get_cq_event中的参数返回。  
参数通道用于指定CC。 CQ只是一个没有内置通知机制的队列。当使用轮询范例进行CQ处理时，不需要CC。用户只需定期轮询CQ。但是，如果您希望使用挂钩范例，则需要CC。 CC是允许用户通知新CQE在CQ上的机制。  
参数comp_vector用于指定用于表示完成事件的完成向量。它必须是 >= 0和 < context->num_comp_vectors。  

#### 3.3.8 ibv_resize_cq

    int ibv_resize_cq(struct ibv_cq *cq, int cqe)

    Input Parameters:
    cq      CQ to resize
    cqe     Minimum number of entries CQ will support
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_resize_cq调整完成队列（CQ）的大小。  
参数cqe必须至少为队列中未决条目的数量。 队列的实际大小可能大于指定的值。 当CQ调整大小时，CQ可以（或可能不）包含完成，可以在CQ工作期间调整大小。

#### 3.3.9 ibv_destroy_cq

    int ibv_destroy_cq(struct ibv_cq *cq)

    Input Parameters:
    cq      CQ to destroy
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_cq释放一个完成队列（CQ）。 如果存在与其相关联的指定CQ的队列对（QP），则此命令将失败。

#### 3.3.10 ibv_create_comp_channel

    struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    pointer to created CC or NULL on failure.

ibv_create_comp_channel创建一个完成通道。 完成通道是当新的完成队列事件（CQE）已经被放置在完成队列（CQ）上时用户接收通知的机制。

#### 3.3.11 ibv_destroy_comp_channel

    int ibv_destroy_comp_channel(struct ibv_comp_channel *channel)

    Input Parameters:
    channel         struct ibv_comp_channel from ibv_create_comp_channel
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_comp_channel释放完成通道。 如果任何完成队列（CQ）仍然与此完成通道相关联，则此命令将失败。

### 3.4 Protection Domain Operations

建立保护域（PD）后，您可以在该域内创建对象。 本节介绍PD上可用的操作。 这些包括注册存储器区域（MR），创建队列对（QP）或共享接收队列（SRQ）和地址句柄（AH）。

#### 3.4.1 ibv_reg_mr

    struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, enum ibv_access_flags access)    

    Input Parameters:
    pd          protection domain, struct ibv_pd from ibv_alloc_pd
    addr        memory base address
    length      length of memory region in bytes
    access      access flags
    Output Parameters:
    none
    Return Value:
    pointer to created memory region (MR) or NULL on failure.    

ibv_reg_mr注册一个内存区域（MR），将其与保护域（PD）相关联，并为其分配本地和远程密钥（lkey，rkey）。 用到内存的所有VPI命令都需要通过该命令注册内存。 相同的物理内存可以被映射到不同的MR，甚至根据用户需求，允许不同的权限或PD被分配给相同的存储器。    
访问标志可能是按位或以下枚举之一：  

| Enum                     | Descript     |
| ------------------------ | ------------ |
| IBV_ACCESS_LOCAL_WRITE   | 允许本地主机写访问    |
| IBV_ACCESS_REMOTE_WRITE  | 允许远程主机写入访问   |
| IBV_ACCESS_REMOTE_READ   | 允许远程主机读取访问权限 |
| IBV_ACCESS_REMOTE_ATOMIC | 允许远程主机进行原子访问 |
| IBV_ACCESS_MW_BIND       | 允许此MR上的内存窗口  |

本地读取访问是隐含和自动的。    
任何违反给定内存操作访问权限的VPI操作都将失败。 请注意，队列对（QP）属性还必须具有正确的权限，否则操作将失败。  
如果设置了IBV_ACCESS_REMOTE_WRITE或IBV_ACCESS_REMOTE_ATOMIC，则也必须设置IBV_ACCESS_LOCAL_WRITE。  

结构体ibv_mr定义如下：

    struct ibv_mr
    {
        struct ibv_context      *context;
        struct ibv_pd           *pd;
        void                    *addr;
        size_t                  length;
        uint32_t                handle;
        uint32_t                lkey;
        uint32_t                rkey;
    };

#### 3.4.2 ibv_dereg_mr

    int ibv_dereg_mr(struct ibv_mr *mr)

    Input Parameters:
    mr              struct ibv_mr from ibv_reg_mr
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_dereg_mr释放内存区域（MR）。 如果任何存储窗口（MW）仍然绑定到MR，操作将失败。

#### 3.4.3 ibv_create_qp

    struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr)

    Input Parameters:
    pd                  struct ibv_pd from ibv_alloc_pd
    qp_init_attr        initial attributes of queue pair
    Output Parameters:
    qp_init_attr        actual values are filled in
    Return Value:
    pointer to created queue pair (QP) or NULL on failure.

ibv_create_qp创建QP。 当创建QP时，将其置于RESET状态。  
struct qp_init_attr定义如下：

    struct ibv_qp_init_attr
    {
        void                *qp_context;
        struct ibv_cq       *send_cq;
        struct ibv_cq       *recv_cq;
        struct ibv_srq      *srq;
        struct ibv_qp_cap   cap;
        enum ibv_qp_type    qp_type;
        int                 sq_sig_all;
        struct ibv_xrc_domain *xrc_domain;
    };

| variable   | description                              | example        |
| ---------- | ---------------------------------------- | -------------- |
| qp_context | (optional) user defined value associated with QP | 未指定            |
| send_cq    | send CQ. This must be created by the user prior to calling ibv_create_qp | adapter.cq_    |
| recv_cq    | receive CQ. This must be created by the user prior to calling ibv_create_qp. It may be the same as send_cq. | adapter.cq_    |
| srq        | (optional) shared receive queue. Only used for SRQ QP’s. | 未指定            |
| cap        | defined below.                           | defined below. |
| qp_type    | must be one of the following:<br>IBV_QPT_RC = 2IBV_QPT_UC<br>IBV_QPT_UD<br>IBV_QPT_XRC<br>IBV_QPT_RAW_PACKET = 8<br>IBV_QPT_RAW_ETH = 8 | IBV_QPT_RC     |
| sq_sig_all | If this value is set to 1, all send requests (WR) will generate completion queue events (CQE). If this value is set to 0, only WRs that are flagged will generate CQE’s (see ibv_post_send). | 未指定            |
| xrc_domain | (Optional) Only used for XRC operations. | 未指定            |

    struct ibv_qp_cap
    {
        uint32_t max_send_wr;
        uint32_t max_recv_wr;
        uint32_t max_send_sge;
        uint32_t max_recv_sge;
        uint32_t max_inline_data;
    };

| variable        | description                              | example                     |
| --------------- | ---------------------------------------- | --------------------------- |
| max_send_wr     | Maximum number of outstanding send requests in the send queue | MAX_CONCURRENT_WRITES(1000) |
| max_recv_wr     | Maximum number of outstanding receive requests (buffers) in the receive queue. | MAX_CONCURRENT_WRITES(1000) |
| max_send_sge    | Maximum number of scatter/gather elements (SGE) in a WR on the send queue. | 1                           |
| max_recv_sge    | Maximum number of SGEs in a WR on the receive queue. | 1                           |
| max_inline_data | Maximum size in bytes of inline data on the send queue. | 未指定                         |

#### 3.4.4 ibv_destroy_qp

    int ibv_destroy_qp(struct ibv_qp *qp)

    Input Parameters:
    qp              struct ibv_qp from ibv_create_qp
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_qp释放队列对（QP）。

#### 3.4.5 ibv_create_srq

    struct ibv_srq *ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr)

    Input Parameters:
    pd              The protection domain associated with the shared receive queue (SRQ)
    srq_init_attr   A list of initial attributes required to create the SRQ
    Output Parameters:
    ibv_srq__attr   Actual values of the struct are set
    Return Value:
    A pointer to the created SRQ or NULL on failure

ibv_create_srq创建一个共享接收队列（SRQ）。 读取srq_attr-> max_wr和srq_attr-> max_sge以确定所需的SRQ大小，并将其设置为返回时分配的实际值。 如果ibv_create_srq成功，那么max_wr和max_sge将至少与请求的值一样大。  
struct ibv_srq定义如下：

    struct ibv_srq {
        struct ibv_context      *context;       struct ibv_context from ibv_open_device
        void                    *srq_context;
        struct ibv_pd           *pd;            Protection domain
        uint32_t                handle;
        pthread_mutex_t         mutex;
        pthread_cond_t          cond;
        uint32_t                events_completed;
    }

struct ibv_srq_init_attr定义如下：

    struct ibv_srq_init_attr
    {
        void *srq_context;
        struct ibv_srq_attr attr;
    };

--- | ---
srq_context  | struct ibv_context from ibv_open_device
attr         | An ibv_srq_attr struct defined as follows

struct ibv_srq_attr定义如下：

    struct ibv_srq_attr
    {
        uint32_t    max_wr;
        uint32_t    max_sge;
        uint32_t    srq_limit;
    };

--- | ---
max_wr      | Requested maximum number of outstanding WRs in the SRQ
max_sge     | Requested number of scatter elements per WR
srq_limit   | The limit value of the SRQ (irrelevant for ibv_create_srq)

#### 3.4.6 ibv_modify_srq

    int ibv_modify_srq (struct ibv_srq *srq, struct ibv_srq_attr *srq_attr, int srq_attr_mask)

    Input Parameters:
    srq             The SRQ to modify
    srq_attr        Specifies the SRQ to modify (input)/the current values of the selected SRQ attributes are returned (output)
    srq_attr_mask   A bit-mask used to specify which SRQ attributes are being modified
    Output Parameters:
    srq_attr        The struct ibv_srq_attr is returned with the updated values
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_modify_srq使用srq_attr中的属性值根据掩码srq_attr_mask修改SRQ srq的属性。 srq_attr是ibv_create_srq下面定义的ibv_srq_attr结构。 参数srq_attr_mask指定要修改的SRQ属性。 它是一个或多个标志的0或按位OR：

--- | ---
IBV_SRQ_MAX_WR  | Resize the SRQ
IBV_SRQ_LIMIT   | Set the SRQ limit

如果要修改的任何属性无效，则不会修改任何属性。 此外，并非所有设备都支持调整SRQ的大小。 要检查设备是否支持调整大小，请检查设备能力标志中是否设置了IBV_DEVICE_SRQ_RESIZE位。  
一旦SRQ中的WR数降到SRQ限制以下，修改SRQ限制就会使SRQ产生一个IBV_EVENT_SRQ_LIMIT_REACHED'低水印'异步事件。

#### 3.4.7 ibv_destroy_srq

#### 3.4.8 ibv_open_xrc_domain

#### 3.4.9 ibv_create_rc_srq

#### 3.4.10 ibv_close_xrc_domain

#### 3.4.11 ibv_create_xrc_rcv_qp

#### 3.4.12 ibv_modify_xrc_rcv_qp

#### 3.4.13 ibv_reg_xrc_rcv_qp

#### 3.4.14 ibv_unreg_xrc_rcv_qp

#### 3.4.15 ibv_create_ah

#### 3.4.16 ibv_destroy_ah

### 3.5 Queue Pair Bringup (ibv_modify_qp)

#### 3.5.1 ibv_modify_qp

    int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, enum ibv_qp_attr_mask attr_mask)

    struct ibv_qp_attr
    {
        enum ibv_qp_state qp_state;
        enum ibv_qp_state cur_qp_state;
        enum ibv_mtu path_mtu;
        enum ibv_mig_state path_mig_state;
        uint32_t qkey;
        uint32_t rq_psn;
        uint32_t sq_psn;
        uint32_t dest_qp_num;
        int qp_access_flags;
        struct ibv_qp_cap cap;
        struct ibv_ah_attr ah_atpsntr;
        struct ibv_ah_attr alt_ah_attr;
        uint16_t pkey_index;
        uint16_t alt_pkey_index;
        uint8_t en_sqd_async_notify;
        uint8_t sq_draining;
        uint8_t max_rd_atomic;
        uint8_t max_dest_rd_atomic;
        uint8_t min_rnr_timer;
        uint8_t port_num;
        uint8_t timeout;
        uint8_t retry_cnt;
        uint8_t rnr_retry;
        uint8_t alt_port_num;
        uint8_t alt_timeout;
    };
    
    IBV_QP_STATE
    IBV_QP_CUR_STATE
    IBV_QP_EN_SQD_ASYNC_NOTIFY
    IBV_QP_ACCESS_FLAGS
    IBV_QP_PKEY_INDEX
    IBV_QP_PORT
    IBV_QP_QKEY
    IBV_QP_AV
    IBV_QP_PATH_MTU
    IBV_QP_TIMEOUT
    IBV_QP_RETRY_CNT
    IBV_QP_RNR_RETRY
    IBV_QP_RQ_PSN
    IBV_QP_MAX_QP_RD_ATOMIC
    IBV_QP_ALT_PATH
    IBV_QP_MIN_RNR_TIMER
    IBV_QP_SQ_PSN
    IBV_QP_MAX_DEST_RD_ATOMIC
    IBV_QP_PATH_MIG_STATE
    IBV_QP_CAP
    IBV_QP_DEST_QPN

#### 3.5.2 RESET to INIT

#### 3.5.3 INIT to RTR

#### 3.5.4 RTR to RTS

### 3.6 Active Queue Pair Operations

#### 3.6.1 ibv_query_qp

#### 3.6.2 ibv_query_srq

#### 3.6.3 ibv_query_xrc_rcv_qp

#### 3.6.4 ibv_post_recv

    int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr)

    Input Parameters:

    qp struct ibv_qp from ibv_create_qp
    wr first work request (WR) containing receive buffers
    
    Output Parameters:
    bad_wr pointer to first rejected WR
    
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.
    
    struct ibv_recv_wr
    {
        uint64_t wr_id;             //user assigned work request ID
        struct ibv_recv_wr *next;   //pointer to next WR, NULL if last one.
        struct ibv_sge *sg_list;    //scatter array for this WR
        int num_sge;                //number of entries in sg_list
    };
    
    struct ibv_sge
    {
        uint64_t addr;              //address of buffer
        uint32_t length;            //length of buffer
        uint32_t lkey;              //local key (lkey) of buffer from ibv_reg_mr
    };

ibv_post_recv将一系列WRs发布到QP的接收队列。应将至少一个接收缓冲区发布到接收队列，以将QP转换为RTR。 随着远程执行发送、立即发送和使用即时操作的RDMA写入、接收缓冲区被消耗。 接收缓冲区不用于其他RDMA操作。 WR列表的处理在第一个错误上停止，并且在bad_wr中返回指向违规WR的指针。

#### 3.6.5 ibv_post_send

```c++
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr)

Input Parameters:
qp      struct ibv_qp from ibv_create_qp
Output Parameters:
bad_wr  pointer to first rejected WR
Return Value:
0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.
```

ibv_post_send将一个WR list链接到队列对（QP）发送队列。 此操作用于启动所有通信，包括RDMA操作。 WR列表的处理在第一个错误上停止，并且在bad_wr中返回指向违规WR的指针。
用户不应该更改或销毁与WR相关联的AH，直到请求完全执行，并从对应的完成队列（CQ）中检索到完成队列条目（CQE）以避免意外的行为。
WR完全执行后才能安全地重复使用WR缓冲区，并从对应的CQ中检索出WCE。 但是，如果设置了IBV_SEND_INLINE标志，缓冲区可以在调用返回后立即重新使用。
struct ibv_send_wr定义如下：

```c++
struct ibv_send_wr
{
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    enum ibv_send_flags send_flags;
    uint32_t imm_data;/* network byte order */
    union
    {
        struct
        {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
        struct
        {
            uint64_t remote_addr;
            uint64_t compare_add;
            uint64_t swap;
            uint32_t rkey;
        } atomic;
        struct
        {
            struct ibv_ah *ah;
            uint32_t remote_qpn;
            uint32_t remote_qkey;
        } ud;
    } wr;
    uint32_t xrc_remote_srq_num;
};
```

| variable           | description                              |
| ------------------ | ---------------------------------------- |
| wr_id              | user assigned work request ID            |
| next               | pointer to next WR, NULL if last one.    |
| sg_list            | scatter/gather array for this WR         |
| num_sge            | number of entries in sg_list             |
| opcode             | IBV_WR_RDMA_WRITE<br>IBV_WR_RDMA_WRITE_WITH_IMM<br>IBV_WR_SEND<br>IBV_WR_SEND_WITH_IMM<br>IBV_WR_RDMA_READ<br>IBV_WR_ATOMIC_CMP_AND_SWP<br>IBV_WR_ATOMIC_FETCH_AND_ADD |
| send_flags         | (optional) - this is a bitwise OR of the flags. See the details below. |
| imm_data           | immediate data to send in network byte order |
| remote_addr        | remote virtual address for RDMA/atomic operations |
| rkey               | remote key (from ibv_reg_mr on remote) for RDMA/atomic operations |
| compare_add        | compare value for compare and swap operation |
| swap               | swap value                               |
| ah                 | address handle (AH) for datagram operations |
| remote_qpn         | remote QP number for datagram operations |
| remote_qkey        | Qkey for datagram operations             |
| xrc_remote_srq_num | shared receive queue (SRQ) number for the destination extended reliable connection (XRC). Only used for XRC operations. |



| send flags              | description                              |
| ----------------------- | ---------------------------------------- |
| IBV_SEND_FENCE          | set fence indicator                      |
| IBV_SEND_SIGNALED       | send completion event for this WR. Only meaningful for QPs that had the sq_sig_all set to 0<br>发送此WR的完成事件。 只对sq_sig_all设置为0的QP有意义 |
| IBV_SEND_SEND_SOLICITED | set solicited event indicator            |
| IBV_SEND_INLINE         | send data in sge_list as inline data.    |

#### 3.6.6 ibv_post_srq_recv

#### 3.6.7 ibv_req_notify_cq

    int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only)

#### 3.6.8 ibv_get_cq_event

#### 3.6.9 ibv_ack_cq_events

```c++
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents)
Input Parameters:
cq          struct ibv_cq from ibv_create_cq
nevents     number of events to acknowledge (1...n)
Output Parameters:
None
Return Value:
None
```

ibv_ack_cq_events确认从ibv_get_cq_event接收到的事件。 虽然从ibv_get_cq_event接收到的每个通知只算一个事件，但用户可以通过单次调用ibv_ack_cq_events来确认多个事件。 要确认的事件数量在nevents中传递，应至少为1个。由于此操作需要互斥体，因此在某个调用中确认多个事件可能会提供更好的性能。
有关其他详细信息，请参阅ibv_get_cq_event。

#### 3.6.10 ibv_poll_cq

```c++
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
Input Parameters:
cq              struct ibv_cq from ibv_create_cq
num_entries     maximum number of completion queue entries (CQE) to return
Output Parameters:
wc              CQE array
Return Value:
Number of CQEs in array wc or -1 on error
```

ibv_poll_cq从完成队列（CQ）中检索CQE。 用户应该分配一个struct ibv_wc的数组，并将其传递给wc中的调用。 wc中可用的条目数应该以num_entries的形式传递。 释放这个内存是用户的责任。
实际检索的CQE数量作为返回值。
定期检查CQ以防止超限。 在超载的情况下，CQ将被关闭，并且将发送异步事件IBV_EVENT_CQ_ERR。
struct ibv_wc定义如下：

```c++
struct ibv_wc
{
    uint64_t wr_id;
    enum ibv_wc_status status;
    enum ibv_wc_opcode opcode;
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data;/* network byte order */
    uint32_t qp_num;
    uint32_t src_qp;
    enum ibv_wc_flags wc_flags;
    uint16_t pkey_index;
    uint16_t slid;
    uint8_t sl;
    uint8_t dlid_path_bits;
};
```

| variable       | 描述                                       |
| -------------- | ---------------------------------------- |
| wr_id          | user specified work request id as given in ibv_post_send or ibv_post_recv |
| status         | IBV_WC_SUCCESS<br>IBV_WC_LOC_LEN_ERR<br>IBV_WC_LOC_QP_OP_ERR<br>IBV_WC_LOC_EEC_OP_ERR<br>IBV_WC_LOC_PROT_ERR<br>IBV_WC_WR_FLUSH_ERR<br>IBV_WC_MW_BIND_ERR<br>IBV_WC_BAD_RESP_ERR<br>IBV_WC_LOC_ACCESS_ERR<br>IBV_WC_REM_INV_REQ_ERR<br>IBV_WC_REM_ACCESS_ERR<br>IBV_WC_REM_OP_ERR<br>IBV_WC_RETRY_EXC_ERR<br>IBV_WC_RNR_RETRY_EXC_ERR<br>IBV_WC_LOC_RDD_VIOL_ERR<br>IBV_WC_REM_INV_RD_REQ_ERR<br>IBV_WC_REM_ABORT_ERR<br>IBV_WC_INV_EECN_ERR<br>IBV_WC_INV_EEC_STATE_ERR<br>IBV_WC_FATAL_ERR<br>IBV_WC_RESP_TIMEOUT_ERR<br>IBV_WC_GENERAL_ERR |
| opcode         | IBV_WC_SEND,<br>IBV_WC_RDMA_WRITE,<br>IBV_WC_RDMA_READ,<br>IBV_WC_COMP_SWAP,<br>IBV_WC_FETCH_ADD,<br>IBV_WC_BIND_MW,<br>IBV_WC_RECV= 1 << 7,<br>IBV_WC_RECV_RDMA_WITH_IMM |
| vendor_err     | vendor specific error                    |
| byte_len       | number of bytes transferred              |
| imm_data       | immediate data                           |
| qp_num         | local queue pair (QP) number             |
| src_qp         | remote QP number                         |
| wc_flags       | IBV_WC_GRH global route header (GRH) is present in UD packet<br>IBV_WC_WITH_IMM immediate data value is valid |
| pkey_index     | index of pkey (valid only for GSI QPs)   |
| slid           | source local identifier (LID)            |
| sl             | service level (SL)                       |
| dlid_path_bits | destination LID path bits                |

#### 3.6.11 ibv_init_ah_from_wc

#### 3.6.12 ibv_create_ah_from_wc

#### 3.6.13 ibv_attach_mcast

#### 3.6.14 ibv_detach_mcast

### 3.7 Event Handing Operations

#### 3.7.1 ibv_get_async_event

#### 3.7.2 ibv_ack_async_enentibv_reg_mr

#### 3.7.3 ibv_event_type_str

### 3.8 Experimental APIs

#### 3.8.1 ibv_exp_query_device

#### 3.8.2 ibv_exp_create_qp

#### 3.8.3 ibv_exp_post_send

## 7 IBV Verbs编程示例

### 7.1 使用IBV Verbs的RDMA_RC示例概要

以下是编程示例中的函数的概要，按照它们的调用顺序。

代码如 RDMA_RC_example.c

解析命令行。 用户可以设置测试的TCP端口，设备名称和设备端口。 如果设置，这些值将覆盖config中的默认值。 最后一个参数是服务器名称。 如果设置了服务器名称，则指定要连接的服务器，因此将程序放入客户端模式。 否则程序处于服务器模式。

1. 调用**print_config**打印输出配置信息。
2. 调用**resources_init**清除资源结构。
3. 调用**resources_create**
    1. 调用**sock_connect**使用TCP套接字与对方进行连接。
        1. 如果是客户端，则解析服务器的DNS地址并发起连接。
        2. 如果是服务器，则在指定端口上侦听传入连接。
    2. 获取设备列表，找到我们想要的设备，并打开它。
    3. 释放设备列表。
    4. 获取端口信息。
    5. 创建一个PD。
    6. 创建一个CQ。
    7. 分配一个缓冲区，初始化并注册。
    8. 创建一个QP。
4. 调用**connect_qp**
    1. 调用**sock_sync_data**来在服务器和客户端之间交换地址信息，使用由sock_connect创建的TCP套接字，在客户端和服务器之间同步给定的一组数据。 由于这个功能是阻塞的，所以也使用dummy data来同步客户端和服务器的时序。
    2. 调用**modify_qp_to_init** 将QP转换到INIT状态。
    3. 如果是客户端，则调用**post_receive**。
        1. 为接收缓冲区准备一个 scatter/gather 实体。
        2. 准备一个RR。
        3. post这个RR。
    4. 调用**modify_qp_to_rtr** 将QP转换到RTR状态。
    5. 调用**modify_qp_to_rts** 将QP转换到RTS状态。
    6. 调用**sock_sync_data**来同步客户端<->服务器的时序。
5. 如果在服务器模式下，请使用 IBV_WR_SEND 操作调用**post_send**。
    1. 为要发送的数据（或在RDMA读取情况下接收的数据）准备分散/收集条目。
    2. 创建SR。 请注意，IBV_SEND_SIGNALED是多余的。
    3. 如果是RDMA操作，请设置地址和密钥。
    4. post SR。
6. 调用**poll_completion**。 请注意，服务器端期望从SEND请求完成，并且客户端期望RECEIVE完成。
    1. 轮询CQ直到找到条目或达到MAX_POLL_CQ_TIMEOUT毫秒。
7. 如果在客户端模式下，则显示通过RECEIVE操作收到的消息。
8. 如果在服务器模式下，则使用新的消息填充缓冲区。
9. 同步 客户端<->服务器 的时序。
10. 在这一点上，服务器直接进入下一个时序同步。 所有RDMA操作都由客户端完成。  
11. **客户端**操作
    1. 使用IBV_WR_RDMA_READ调用**post_send**，执行服务器缓冲区的RDMA读取。  
    2. 调用**poll_completion**。  
    3. 显示服务器的消息。  
    4. 使用新消息设置发送缓冲区。  
    5. 使用IBV_WR_RDMA_WRITE调用**post_send**以执行服务器缓冲区的RDMA写入。  
    6. 调用**poll_completion**。  
12. 同步客户端< - >服务器的时序。  
13. 如果服务器模式，显示缓冲区，证明RDMA已经写入。  
14. 调用**resources_destroy** release/free/deallocate 资源结构中的所有项目。。  
15. 释放设备名称字符串。  
16. 完成。  
