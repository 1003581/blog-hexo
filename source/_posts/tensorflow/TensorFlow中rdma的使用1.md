---
title: TensorFlow中rdma的使用1
date: 2017-09-14 15:59:32
tags: 
- tensorflow
- rdma
categories: tensorflow
---

# rdma.cc
<!-- more -->

## 相关结构体

```c++
// structure to save the address of remote channels.
struct RdmaAddress {
  uint32_t lid;
  uint32_t qpn;
  uint32_t psn;
  uint64_t snp;
  uint64_t iid;
};
// structure to save information for remote memory regions.
struct RemoteMR {
  uint64_t remote_addr;
  uint32_t rkey;
};
enum BufferStatus { none, idle, busy };
enum Location { local, remote };
enum BufferType { ACK, MESSAGE, TENSOR };
enum RdmaMessageType {
  RDMA_MESSAGE_ACK,
  RDMA_MESSAGE_BUFFER_IDLE,
  RDMA_MESSAGE_BUFFER_REQUEST,
  RDMA_MESSAGE_BUFFER_RESPONSE,
  RDMA_MESSAGE_TENSOR_REQUEST,
  RDMA_MESSAGE_TENSOR_WRITE
};
```

## RdmaAdapter

### RdmaAdapter(const WorkerEnv* worker_env)

1. dev_list = ibv_get_device_list(NULL);获取主机可用的VPI设备列表
1. ib_dev = dev_list[0]默认取第一个设备
1. ibv_context* context = ibv_open_device(ib_dev);打开设备获取context
1. ibv_pd* pd = ibv_alloc_pd(context);创建Protection Domain
1. worker_env\_ (worker_env)拷贝工作环境参数
1. event_channel\_ = ibv_create_comp_channel(context\_);创建完成通道，用于通知完成队列
1. cq_ = ibv_create_cq(context\_, MAX_CONCURRENT_WRITES * 2, NULL, event_channel_, 0);创建完成队列
1. ibv_req_notify_cq(cq_, 0)完成完成队列与完成通道的关联
1. 启动处理线程Process_CQ()

### ~RdmaAdapter()

1. ibv_destroy_cq(cq_)
1. ibv_destroy_comp_channel(event_channel_)
1. ibv_dealloc_pd(pd_)
1. ibv_close_device(context_)

### void Process_CQ()

1. ibv_get_cq_event(event_channel_, &cq, &cq_context)阻塞等待队列中进入新元素
1. ibv_ack_cq_events(cq, 1);确认收到的事件
1. ibv_req_notify_cq(cq_, 0)重新注册，等待下次事件触发
1. `int ne = ibv_poll_cq(cq\_, MAX_CONCURRENT_WRITES * 2, static_cast<ibv_wc*>(wc_));` 从CQ队列中获取所有的事件,ne表示事件个数
1. 遍历每个cqe
   1. 判断wc_[i].status == IBV_WC_SUCCESS，检查wr的状态是否正确
   1. 若wc_[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM
      1. RdmaChannel* rc = reinterpret_cast<RdmaChannel*>(wc_[i].wr_id);若是接收事件，则wr_id中存放本地Channel的指针
      1. rc->Recv();让channel做好接收的准备
      1. RdmaBuffer* rb = rc->FindBuffer(wc_[i].imm_data); 利用imm_data来寻找buffer地址
      1. RdmaMessage::ParseMessage(rm, rb->buffer_); 将buffer中的信息解析成Message
      1. 判断rm.type
         1. 若rm.type_ == RDMA_MESSAGE_ACK
            1. 将本地的tx_message_buffer_的remote状态设置为空闲。
            1. 本地tx_message_buffer_发送下一条Message
         1. 若rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST
            1. 首先通过本地的tx_ack_buffer_发送ack，使得对方释放它的message buffer。
            1. RdmaBuffer* tb = rc->FindOrCreateBuffer(rm.name_); 通过name来寻找buffer，此处的buffer为tensor buffer。
            1. string key_with_step_id = VerbsUtil::AppendStepidToKey(rm.name\_, rm.step_id\_); 生成一个类似"tx_tensor_buffer123456"的标识符。
            1. tb->EnqueueItem(key_with_step_id); 将此标识符放入处理队列中。
            1. worker_env_->compute_pool->Schedule([tb\]() { tb->SendNextItem(); }); 设置定时任务，使得tx_tensor_buffer开始发送数据。
         1. 若rm.type_ == RDMA_MESSAGE_BUFFER_IDLE
            1. 首先通过本地的tx_ack_buffer_发送ack，使得对方释放它的message buffer。
            1. RdmaBuffer* tb = rc->FindBuffer(rm.name_); 通过name来寻找tensor buffer。
            1. 设置此tx_tensor_buffer的remote状态为空闲，表示对方已经就绪。
            1. 设置定时任务，使得tx_tensor_buffer开始发送数据。
         1. 若rm.type_ == RDMA_MESSAGE_BUFFER_REQUEST
            1. 首先通过本地的tx_ack_buffer_发送ack，使得对方释放它的message buffer。
            1. 收到此消息时，表示发送方说“现有的tensor buffer不够大小了，我已经重新创建了，大小告诉你了，你也重新创一个，我们再建立连接。”
            1. RdmaBuffer* tb = rc->FindOrCreateBuffer(rm.name_, TENSOR); 寻找这个tensor buffer，找到以后进行空间处理(tb->CreateCPUBuffer(rm.buffer_size\_);)、连接处理(tb->SetRemoteMR(rmr, true);)。
            1. 创建成功后，回复发送者说，我创建好了，你来跟我建立连接吧。准备发送RDMA_MESSAGE_BUFFER_RESPONSE消息。
            1. 通过tx_message_buffer发送此消息，消息入队。
            1. tx_message_buffer开始发送下一条消息。
         1. 若rm.type_ == RDMA_MESSAGE_BUFFER_RESPONSE
            1. 首先通过本地的tx_ack_buffer_发送ack，使得对方释放它的message buffer。
            1. 寻找本地的tx_tensor_buffer，来和接收者建立链接。
            1. 将tx_tensor_buffer的local和remote状态都设置为空闲，准备发送数据。
            1. 设置定时任务，使得tx_tensor_buffer开始发送数据。
         1. 若rm.type_ == RDMA_MESSAGE_TENSOR_WRITE
            1. 设置定时任务，通过key_with_step_id (ex: "tx_tensor_buffer123456")来运行指定的callback函数。
   1. 若wc_[i].opcode == IBV_WC_RDMA_WRITE
      1. RdmaBuffer* rb = reinterpret_cast<RdmaBuffer*>(wc_[i].wr_id);  若为本地后台发来的消息，则wr_id中存放buffer地址。一般为tx_message_buffer。
      1. 将该buffer的local状态设置为空闲。
      1. 解析buffer中的消息。
      1. 若buffer中的消息类型是RDMA_MESSAGE_ACK，则不做任何处理。否则创建定时任务，使得tx_message_buffer开始发送下一条数据。

## RdmaChannel

### RdmaChannel(const RdmaAdapter* adapter, const string local_name, const string remote_name_)

1. qp_ = ibv_create_qp(adapter_->pd_, &attr); 创建Queue Pair
1. ibv_modify_qp(qp_, &attr, mask) 初始化QP
1. 创建4个buffer并建立hash，同时加入索引表，tx_message_buffer_ = new RdmaMessageBuffer(this, buffer_names[0]);
1. 执行100次Recv() (ibv_post_recv())，使得buffer准备好接收。

### ~RdmaChannel()

1. ibv_destroy_qp(qp_) 销毁QP
1. 销毁buffer

## TensorFlow相关

### 类关系

```flow
core:refCounts=>start: core:refCounts
Rendezvous=>start: Rendezvous
RendezvousCond=>condition: Derived Class
LocalRendezvousImpl=>start: LocalRendezvousImpl(.cc文件中)
RemoteRendezvous=>start: ReomoteRendezvous(纯虚类)
BaseRemoteRendezvous=>start: BaseRemoteRendezvous
RdmaRemoteRedezvous=>start: RdmaRemoteRedezvous

core:refCounts->Rendezvous->RendezvousCond
RendezvousCond(yes)->RemoteRendezvous->BaseRemoteRendezvous->RdmaRemoteRedezvous
RendezvousCond(no)->LocalRendezvousImpl
```

```flow
RendezvousMgrInterface=>start: RendezvousMgrInterface(纯虚类)
BaseRendezvousMgr=>start: BaseRendezvousMgr
RdmaRendezvousMgr=>start: RdmaRendezvousMgr

RendezvousMgrInterface->BaseRendezvousMgr->RdmaRendezvousMgr
```

### 初始化过程

1. verbs_server_lib.cc文件中存在静态变量 `static VerbsServerRegistrar registrar;`

1. 该静态变量的构造函数中包含VERBS_SERVER服务的注册，和VerbsServerFactory服务对象的创建。

   ```c++
   class VerbsServerRegistrar {
    public:
     VerbsServerRegistrar() {
       gpr_allocation_functions alloc_fns;
       alloc_fns.malloc_fn = port::Malloc;
       alloc_fns.realloc_fn = port::Realloc;
       alloc_fns.free_fn = port::Free;
       gpr_set_allocation_functions(alloc_fns);
       ServerFactory::Register("VERBS_SERVER", new VerbsServerFactory());
     }
   };

   /* static */
   void ServerFactory::Register(const string& server_type,
                                ServerFactory* factory) {
     mutex_lock l(*get_server_factory_lock());
     if (!server_factories()->insert({server_type, factory}).second) {
       LOG(ERROR) << "Two server factories are being registered under "
                  << server_type;
     }
   }
   ```

1. VerbsServerFactory类的重写函数中包含VerbsServer的创建。

   ```c++
   std::unique_ptr<ServerInterface> svr;
   TF_CHECK_OK(NewServer(server, &svr));
   TF_CHECK_OK(svr->Start());
   TF_CHECK_OK(svr->Join());

   class VerbsServerFactory : public ServerFactory {
    public:
     bool AcceptsOptions(const ServerDef& server_def) override {
       return server_def.protocol() == "grpc+verbs";
     }

     Status NewServer(const ServerDef& server_def,
                      std::unique_ptr<ServerInterface>* out_server) override {
       return VerbsServer::Create(server_def, Env::Default(), out_server);
     }
   };
   ```

1. VerbsServer::Create是静态函数，该函数中包含对VerbsService类的对象化、VerbsServer的对象化以及INIT和RdmaRendezvousMgr的对象化(RdmaRemoteRendezvous类被TF_DISALLOW_COPY_AND_ASSIGN修饰，导致RdmaRemoteRendezvous类的拷贝构造函数和复制构造函数为私有)。

   ```c++
   /* static */
   Status VerbsServer::Create(const ServerDef& server_def, Env* env,
                              std::unique_ptr<ServerInterface>* out_server) {
     std::unique_ptr<VerbsServer> ret(new VerbsServer(server_def, Env::Default()));
     ServiceInitFunction service_func = [&ret](const WorkerEnv* worker_env,
                                               ::grpc::ServerBuilder* builder) {
       return SetNewVerbsService(&ret->verbs_service_, worker_env, builder);
     };
     TF_RETURN_IF_ERROR(ret->Init(service_func, NewRdmaRendezvousMgr));
     *out_server = std::move(ret);
     return Status::OK();
   }

   RendezvousMgrInterface* NewRdmaRendezvousMgr(const WorkerEnv* env) {
     return new RdmaRendezvousMgr(env);
   }

   Status VerbsServer::Init(ServiceInitFunction service_func,
                            RendezvousMgrCreationFunction rendezvous_mgr_func) {
     Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
     {
       mutex_lock l(mu_);
       CHECK_EQ(verbs_state_, DISCONNECTED);
       CHECK(ChannelCacheFactory(server_def(), &channel_cache_).ok());
       rdma_mgr_ = new RdmaMgr(worker_env(), channel_cache_);
       // set rdma_mgr for verbs_service and rdma_rendezvous_mgr
       verbs_service_->SetRdmaMgr(rdma_mgr_);
       dynamic_cast<RdmaRendezvousMgr*>(worker_env()->rendezvous_mgr)
           ->SetRdmaMgr(rdma_mgr_);
     }
     return s;
   }

   // Create a GrpcVerbsService, then assign it to a given handle.
   void SetNewVerbsService(GrpcVerbsService** handle, const WorkerEnv* worker_env,
                           ::grpc::ServerBuilder* builder) {
     *handle = new GrpcVerbsService(worker_env, builder);
   }

   RdmaMgr::RdmaMgr(const WorkerEnv* const worker_env,
                    GrpcChannelCache* const channel_cache)
       : worker_env_(worker_env), channel_cache_(channel_cache) {
     rdma_adapter_ = new RdmaAdapter(worker_env_);
     // hardcoded to default session (legacy_session_)
     // TODO: use WorkerSessionForSession
     // need to pass in session handle
     local_worker_ = worker_env_->session_mgr->LegacySession()->worker_name;
     std::vector<string> workers;
     worker_env_->session_mgr->LegacySession()->worker_cache->ListWorkers(
         &workers);
     num_remote_workers_ = workers.size() - 1;
     VLOG(2) << "rmda_mgr on local worker: " << local_worker_;
     for (size_t i = 0; i < workers.size(); i++) {
       if (local_worker_.compare(workers[i]) != 0) {
         channel_table_.insert(
             {workers[i],
              new RdmaChannel(rdma_adapter_, local_worker_, workers[i])});
       }
     }
   }
   ```

1. VerbsServer的Start函数，包含了gRPC线程的创建和Channel的设置。

   ```c++
   Status VerbsServer::Start() {
     Status s = GrpcServer::Start();
     {
       mutex_lock l(mu_);
       if (verbs_state_ == DISCONNECTED) {
         // verbs_thread needs to be initiated
         // before rdma_mgr sets up the rdma channels.
         verbs_thread_.reset(worker_env()->env->StartThread(
             ThreadOptions(), "TF_verbs_service",
             [this] { verbs_service_->HandleRPCsLoop(); }));
         rdma_mgr_->SetupChannels();
         verbs_state_ = CONNECTED;
       }
     }
     return s;
   }

   // This method blocks forever handling requests from the completion queue.
   void GrpcVerbsService::HandleRPCsLoop() {
     for (int i = 0; i < 10; ++i) {
       ENQUEUE_REQUEST(GetRemoteAddress, false);
     }

     void* tag;
     bool ok;

     while (cq_->Next(&tag, &ok)) {
       UntypedCall<GrpcVerbsService>::Tag* callback_tag =
           static_cast<UntypedCall<GrpcVerbsService>::Tag*>(tag);
       if (callback_tag) {
         callback_tag->OnCompleted(this, ok);
       } else {
         cq_->Shutdown();
       }
     }
   }
   ```

1. VerbsServer的Join函数

   ```c++
   Status VerbsServer::Join() {
     Status s = GrpcServer::Join();
     {
       mutex_lock l(mu_);
       if (verbs_state_ == CONNECTED) {
         verbs_state_ = DISCONNECTED;
         verbs_thread_.reset();
       }
     }
     return s;
   }

   Status GrpcServer::Join() {
     mutex_lock l(mu_);
     switch (state_) {
       case NEW:
         // Prevent the server from being started subsequently.
         state_ = STOPPED;
         return Status::OK();
       case STARTED:
       case STOPPED:
         master_thread_.reset();
         worker_thread_.reset();
         return Status::OK();
       default:
         CHECK(false);
     }
   }
   ```

### Rendezvous

Rendezvous的基类为core::RefCounted，声明如下

```c++
namespace core {

// 自动回收机制
class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted();

  // Increments reference count by one.
  void Ref() const;

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.
  bool Unref() const;

  // Return whether the reference count is one.
  // If the reference count is used in the conventional way, a
  // reference count of 1 implies that the current thread owns the
  // reference and no other thread shares it.
  // This call performs the test for a reference count of one, and
  // performs the memory barrier needed for the owning thread
  // to act on the object, knowing that it has exclusive access to the
  // object.
  bool RefCountIsOne() const;

 protected:
  // Make destructor protected so that RefCounted objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  virtual ~RefCounted();

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};
```

Rendezvous 类声明

```c++
// A Rendezvous is an abstraction for passing a Tensor
// from a producer to a consumer, where the consumer may safely
// request the Tensor before or after it has been produced.  A
// producer never blocks when using a Rendezvous.  A consumer has the
// choice of making a blocking call or providing a callback: in either
// case, the consumer receives the Tensor as soon as it is available.
//
// A Rendezvous key encodes a single <producer, consumer> pair.  It is
// an error to call Send() or Recv*() more than once with the same
// key.
class Rendezvous : public core::RefCounted {
 public:
  struct Args {
    DeviceContext* device_context = nullptr;
    AllocatorAttributes alloc_attrs;
  };

  // Constructs a rendezvous key for the tensor of "name" sent from
  // "src_device" to "dst_device". The tensor is generated in the frame
  // and iteration specified by "frame_iter".
  static string CreateKey(const string& src_device, uint64 src_incarnation,
                          const string& dst_device, const string& name,
                          const FrameAndIter& frame_iter);

  // Parses the key constructed by CreateKey and parse src/dst device
  // names into structures respectively.
  struct ParsedKey {
    StringPiece src_device;
    DeviceNameUtils::ParsedName src;
    uint64 src_incarnation = 0;
    StringPiece dst_device;
    DeviceNameUtils::ParsedName dst;
    StringPiece edge_name;

    ParsedKey() {}
    ParsedKey(const ParsedKey& b) { *this = b; }

    ParsedKey& operator=(const ParsedKey& b);
    StringPiece FullKey() const { return buf_; }

   private:
    friend class Rendezvous;
    friend class SendOp;
    friend class RecvOp;
    string buf_;
  };
  static Status ParseKey(StringPiece key, ParsedKey* out);

  // The caller is a tensor producer and it sends a message (a tensor
  // "val" and a bool "is_dead") under the given "key".
  //
  // {val, is_dead} is bundled as a message sent and received.
  // Typically, is_dead is set by some control flow nodes
  // (e.g., a not-taken branch).  args is passed by Send to the
  // Recv function to communicate any information that the Recv
  // function might need.  This is typically only necessary for
  // Send/Recv on the same worker.
  //
  // Send() never blocks.
  virtual Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
                      const bool is_dead) = 0; // 纯虚函数

  // Callback provided by a tensor consumer waiting on the rendezvous.
  // It will be invoked when the tensor is available, or when a non-OK
  // status arises in the production of that tensor.  It also gets
  // two Rendezvous::Args, one provided by the sender, the other by the
  // receiver, which may be needed when a non-CPU device is in use
  // by either side.
  typedef std::function<void(const Status&, const Args&, const Args&,
                             const Tensor&, const bool)>
      DoneCallback;

  virtual void RecvAsync(const ParsedKey& key, const Args& args,
                         DoneCallback done) = 0; // 纯虚函数

  // Synchronous wrapper for RecvAsync.
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead, int64 timeout_ms);
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead);

  // Aborts all pending and future Send/Recv with the given "status".
  //
  // StartAbort() does not wait for ongoing calls to finish.
  // REQUIRES: !status.ok()
  virtual void StartAbort(const Status& status) = 0;

 protected:
  ~Rendezvous() override;
};
```

rendezvous.cc中包含了class LocalRendezvousImpl实现，实现了Send和RecvAsync的具体实现。对外提供了如下接口：

```c++
// 具体定义，在cc文件中
class LocalRendezvousImpl : public Rendezvous {
  // ...
}

// rendezvous.h中的接口在cc文件中的实现
Rendezvous* NewLocalRendezvous(bool tolerate_dup_recv) {
  return new LocalRendezvousImpl(tolerate_dup_recv);
}
```

Rendezvous的一个子类RemoteRendezvous为纯虚类

```c++
// RemoteRendezvous follow a 2-part initialization. First the objects are
// constructed. Eventually, they will be initialized. Clients of the
// RendezvousMgrInterface must guarantee to call Initialize on the returned
// RemoteRendezvous eventually.
//
// Partially initialized RemoteRendezvous must respect the Rendezvous interface
// (i.e. Send() must never block), however implementations are not expected to
// actually perform the underlying operations until after the RemoteRendezvous
// has been Initialize'd.

// RemoteRendezvous遵循2部分的初始化。首先构建对象。最终，它们将被初始化。 RendezvousMgrInterface的客户端必须保证最终在返回的RemoteRendezvous上调用Initialize。
//部分初始化RemoteRendezvous必须遵循Rendezvous接口（即Send（）不能阻塞），但是在RemoteRendezvous被初始化之前，实现不会实际执行底层操作。
class RemoteRendezvous : public Rendezvous {
 public:
  // Fully construct the RemoteRendezvous.
  virtual Status Initialize(WorkerSession* session) = 0;
};
```

RemoteRendezvous类的子类BaseRemoteRendezvous

```c++
// RemoteRendezvous is a Rendezvous which can handle either
// the producer or consumer being in a remote process.
//
// Buffering of Tensor values is delegated to a "local" Rendezvous
// obtained from NewLocalRendezvous().  This class just adds
// functionality to coordinate with remote workers.
class BaseRemoteRendezvous : public RemoteRendezvous {
 public:
  BaseRemoteRendezvous(const WorkerEnv* env, int64 step_id,
                       bool tolerate_dup_recv);

  // Upgrades the BaseRemoteRendezvous to full initialization.
  Status Initialize(WorkerSession* session) override;

  // Forwards to local_, where the Tensor "val" will be buffered and
  // any waiting callback stored.
  Status Send(const ParsedKey& key, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

  // This method is called only by the RecvOp.  It tests to see
  // whether the value will be produced by a local or remote device
  // and handles accordingly.  In the local case it forwards to
  // local_, in the remote case it initiates an RPC request.
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;

  void StartAbort(const Status& status) override;

  // This method is called only by the local Worker, forwarded through
  // the same method on RendezvousMgr.  This occurs when the Worker
  // has received a RecvTensor request, either locally or over the
  // network.  In either case it needs to retrieve a locally buffered
  // value from local_, and give it to its caller.
  //
  // Runs "done" as soon as the tensor for "parsed" is available or an error
  // is detected.
  //
  // REQUIRES: "parsed" is one that will be Saved into the local rendezvous.
  void RecvLocalAsync(const ParsedKey& parsed, DoneCallback done);

 protected:
  virtual void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                                   const Rendezvous::Args& args,
                                   DoneCallback done) = 0;

  // Returns true if "src" and "dst" are located in the same worker,
  // and hence may use a local rendezvous.
  virtual bool IsSameWorker(DeviceNameUtils::ParsedName src,
                            DeviceNameUtils::ParsedName dst);

  // If aborted, aborts "call". Otherwise, adds "call" into active_.
  void RegisterCall(BaseRecvTensorCall* call);

  // Removes "call" from active_ if "call" is in active_.
  void DeregisterCall(BaseRecvTensorCall* call);

  WorkerSession* session();

  bool is_initialized();

  ~BaseRemoteRendezvous() override;

  const WorkerEnv* const env_;  // Not owned.
  const int64 step_id_;

 private:
  Rendezvous* local_;  // Owns a Ref on this object.

  mutable mutex mu_;

  // Status given by StartAbort() if any.
  Status status_ GUARDED_BY(mu_);
  WorkerSession* session_ GUARDED_BY(mu_);  // Not owned.

  // Data structures to handle calls when partially initialized.
  struct DeferredCall {
    const ParsedKey parsed;
    DoneCallback done;

    DeferredCall(const ParsedKey& parsed, DoneCallback done);
  };
  std::vector<DeferredCall> deferred_calls_ GUARDED_BY(mu_);

  // Active outstanding RecvTensor calls.
  gtl::FlatSet<BaseRecvTensorCall*> active_ GUARDED_BY(mu_);

  bool is_initialized_locked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return session_ != nullptr;
  }

  // If "is_src" is true, checks that the rendezvous key "parsed"'s
  // source is in this process. If "is_src" is false, checks that the
  // rendezvous key "parsed"'s destination is in this process.
  Status ValidateDevices(const Rendezvous::ParsedKey& parsed, bool is_src);

  // Callback handling the case when a rendezvous has been
  // accomplished in local_ and the consumer is local to this process.
  // Tensor "in" will be copied into "out". The key "parsed" encodes
  // the src and dst devices.
  void SameWorkerRecvDone(const Rendezvous::ParsedKey& parsed,
                          const Rendezvous::Args& in_args,
                          const Rendezvous::Args& out_args, const Tensor& in,
                          Tensor* out, StatusCallback done);

  // Must be called only if fully initialized.
  void RecvLocalAsyncInternal(const ParsedKey& parsed, DoneCallback done);

  TF_DISALLOW_COPY_AND_ASSIGN(BaseRemoteRendezvous);
};
```

在rdma_rendezvous_mgr.cc文件中声明且定义了RdmaRemoteRendezvous子类，RdmaRemoteRendezvous不对外开放

```c++
class RdmaRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RdmaRemoteRendezvous(const WorkerEnv* env,
                       int64 step_id, RdmaMgr* rdma_mgr)
      : BaseRemoteRendezvous(env, step_id, true),
        rdma_mgr_(rdma_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RdmaRemoteRendezvous() override {}
  RdmaMgr* rdma_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRemoteRendezvous);
};
```

### RendezvousMgr

虚基类RendezvousMgrInterface，无具体实现

```c++
// RendezvousMgr keeps track of a set of local rendezvous instances.
// All tensors sent by this worker are buffered in a RendezvousMgr
// until the tensor is received.  Each global unique "step_id"
// corresponds to one local rendezvous instance managed by a
// RendezvousMgr.
//
// E.g.,
//   Rendezvous* rendez = worker_env->rendezvous_mgr->Find(0x8935);
//   fork execution of an graph executor using "rendez"  on thread 1;
//   fork execution of another graph executor using "rendez" on thread 2;
//   ...
//   join threads 1 and 2;
//
// In the example above, execution in thread 1 and 2 communicates with
// each other by send/recv operations through the "rend".
//
// Tensors sent and recved through rendezvous managed by this
// RendezvousMgr must have keys generated by Rendezvous::CreateKey.
class RendezvousMgrInterface {
 public:
  RendezvousMgrInterface() {}
  virtual ~RendezvousMgrInterface() {}

  // Returns Rendezvous supporting send and recv among workers in the
  // "step_id".  The caller takes ownership of one reference on the
  // returned Rendezvous instance.
  //
  // Note: the caller must guarantee to eventually call Initialize on the
  // returned RemoteRendezvous
  virtual RemoteRendezvous* Find(int64 step_id) = 0;

  // Finds the local rendezvous instance for the "step_id".  Runs
  // "done" when the tensor for "key" is produced or an error occurs.
  //
  // This method is used by the rpc handler of RecvTensor.
  virtual void RecvLocalAsync(int64 step_id,
                              const Rendezvous::ParsedKey& parsed,
                              Rendezvous::DoneCallback done) = 0;

  // Synchronous wrapper for RecvLocalAsync.
  virtual Status RecvLocal(int64 step_id, const Rendezvous::ParsedKey& parsed,
                           Tensor* val, bool* is_dead) = 0;

  // Removes rendezvous for "step_id".
  //
  // TODO(zhifengc): Have a background thread in worker that
  // periodically calls CleanupAll().
  virtual void Cleanup(int64 step_id) = 0;

  // Removes all rendezvous.
  virtual void CleanupAll() = 0;
};
```

RendezvousMgrInterface的子类 BaseRendezvousMgr

```c++
// RendezvousMgr keeps track of a set of local rendezvous instances.
// All tensors sent by this worker are buffered in a RendezvousMgr
// until the tensor is received.  Each global unique "step_id"
// corresponds to one local rendezvous instance managed by a
// RendezvousMgr.
//
// E.g.,
//   Rendezvous* rendez = worker_env->rendezvous_mgr->Find(0x8935);
//   fork execution of a graph executor using "rendez" on thread 1;
//   fork execution of another graph executor using "rendez" on thread 2;
//   ...
//   join threads 1 and 2;
//
// In the example above, execution in thread 1 and 2 communicates with
// each other by send/recv operations through `rendez`.
//
// Tensors sent and received through a rendezvous managed by this
// RendezvousMgr must have keys generated by Rendezvous::CreateKey().
class BaseRendezvousMgr : public RendezvousMgrInterface {
 public:
  // 将worker_env赋值给类的成员变量worker_env_
  explicit BaseRendezvousMgr(const WorkerEnv* worker_env);
  // 释放Table中所有的BaseRemoteRendezvous
  ~BaseRendezvousMgr() override;

  // Returns Rendezvous supporting send and recv among workers in the
  // "step_id".  The caller takes ownership of one reference on the
  // returned Rendezvous instance.
  //
  // Note: the caller must guarantee to eventually call Initialize on the
  // returned RemoteRendezvous
  // 调用私有函数FindOrCreate，在调用这个函数后，调用者会建立自己与Rendez的关系，后续调用会跳过Mgr
  RemoteRendezvous* Find(int64 step_id) override;

  // Finds the local rendezvous instance for the "step_id".  Runs
  // "done" when the tensor for "key" is produced or an error occurs.
  //
  // This method is used by the rpc handler of RecvTensor.
  // 利用step_id来FindOrCreate来寻找rendez
  // 利用std::placeholders和std::bind来创建一个Lambda函数
  // 匿名函数: done_cb(std::move(done), _1, _2, _3, _4, _5)
  // 调用rendez的RecvLocalAsync，rendez为BaseRemoteRendezvous基类指针，但是Rdma未重写RecvLocalAsync，所以调用下文的函数
  void RecvLocalAsync(int64 step_id, const Rendezvous::ParsedKey& parsed,
                      Rendezvous::DoneCallback done) override;

  // Synchronous wrapper for RecvLocalAsync.
  // 对前一个函数通过WaitForNotification的方式进行同步管理
  Status RecvLocal(int64 step_id, const Rendezvous::ParsedKey& parsed,
                   Tensor* val, bool* is_dead) override;

  // Removes rendezvous for "step_id".
  //
  // TODO(zhifengc): Have a background thread in worker that
  // periodically calls CleanupAll().
  void Cleanup(int64 step_id) override;

  // Removed all rendezvous.
  void CleanupAll() override;

 protected:
  virtual BaseRemoteRendezvous* Create(int64 step_id,
                                       const WorkerEnv* worker_env) = 0;

 private:
  // Maps step_id to rendezvous.
  typedef gtl::FlatMap<int64, BaseRemoteRendezvous*> Table;

  // Not owned.
  const WorkerEnv* const worker_env_;

  mutex mu_;
  Table table_ GUARDED_BY(mu_);

  // 根据step_id在Table中查找或者创建一个新的，并返回这个Rendezevous，创建时调用Create纯虚函数(继承类实现之)
  BaseRemoteRendezvous* FindOrCreate(int64 step_id);

  TF_DISALLOW_COPY_AND_ASSIGN(BaseRendezvousMgr);
};
```

BaseRendezvousMgr的子类RdmaRendezvousMgr

```c++
// RendezvousMgr keeps track of a set of local rendezvous instances.
// All tensors sent by this worker are buffered in a RendezvousMgr
// until the tensor is received.  Each global unique "step_id"
// corresponds to one local rendezvous instance managed by a
// RendezvousMgr.
//
// E.g.,
//   Rendezvous* rendez = worker_env->rendezvous_mgr->Find(0x8935);
//   fork execution of an graph executor using "rendez"  on thread 1;
//   fork execution of another graph executor using "rendez" on thread 2;
//   ...
//   join threads 1 and 2;
//
// In the example above, execution in thread 1 and 2 communicates with
// each other by send/recv operations through the "rend".
//
// Tensors sent and recved through rendezvous managed by this
// RendezvousMgr must have keys generated by Rendezvous::CreateKey.
class RdmaRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit RdmaRendezvousMgr(const WorkerEnv* env);
  void SetRdmaMgr(RdmaMgr* rdma_mgr) { rdma_mgr_ = rdma_mgr; }

 protected:
  // 子类实现 返回创建的RdmaRemoteRendezvous对象，该对象将被放入BaseRendezvousMgr的Table中
  BaseRemoteRendezvous* Create(int64 step_id,
                               const WorkerEnv* worker_env) override;

 private:
  RdmaMgr* rdma_mgr_;
  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRendezvousMgr);
};
```

### 发送

注：GraphMgr类中的ExecuteAsync函数进行Rendezvous的Init操作。

```c++
void GraphMgr::ExecuteAsync(const string& handle, const int64 step_id,...)
  // ...
  RemoteRendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = rendezvous->Initialize(session);
  // ...
}
```

应用层先寻找Rendezvous

/tensorflow/core/distributed_runtime/graph_mgr.cc:418

```c++
Status GraphMgr::SendInputs(const int64 step_id, const NamedTensors& in) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = SendInputsToRendezvous(rendezvous, in);
  rendezvous->Unref();
  return s;
}

Status GraphMgr::SendInputsToRendezvous(Rendezvous* rendezvous,
                                        const NamedTensors& in) {
  Rendezvous::ParsedKey parsed;
  for (const auto& p : in) {
    const string& key = p.first;
    const Tensor& val = p.second;

    Status s = Rendezvous::ParseKey(key, &parsed);
    if (s.ok()) {
      s = rendezvous->Send(parsed, Rendezvous::Args(), val, false);
    }
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}
```

Rendezvous.send为纯虚函数，实现为BaseRemoteRendezvous::Send。Rdma中没有重写。

```c++
// The caller is a tensor producer and it sends a message (a tensor
// "val" and a bool "is_dead") under the given "key".
//
// {val, is_dead} is bundled as a message sent and received.
// Typically, is_dead is set by some control flow nodes
// (e.g., a not-taken branch).  args is passed by Send to the
// Recv function to communicate any information that the Recv
// function might need.  This is typically only necessary for
// Send/Recv on the same worker.
//
// Send() never blocks.
virtual Status Rendezvous::Send(const ParsedKey& key, const Args& args, const Tensor& val, const bool is_dead) = 0;

Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {
  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << parsed.FullKey();
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    if (!IsLocalDevice(session_->worker_name, parsed.src_device)) {
      return errors::InvalidArgument(
          "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
          session_->worker_name);
    }
  }
  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

// 通用Send函数
Status LocalRendezvousImpl::Send(const ParsedKey& key, const Args& send_args, const Tensor& val,const bool is_dead) override {
  DoneCallback waiter = nullptr; // 接收者的回调函数
  Args recv_args;
  uint64 key_hash = KeyHash(key.FullKey()); // key_hash唯一标识tensor
  VLOG(2) << "Send " << this << " " << key_hash << " " << key.FullKey();
  {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      return status_;
    }
    Item* item = nullptr;
    Table::iterator iter = table_.find(key_hash); // 在本地table中寻找指定tensor，若有，说明已有接收者在等待，若无，说明无接收者
    if (iter == table_.end()) {
      // There is no waiter for this message. Insert the message
      // into the waiters table. The waiter will pick it up when
      // arrives.
      // 将Item放入数据后存入Table中，结束Send操作。
      item = new Item;
      item->waiter = nullptr;
      item->value = val;
      item->is_dead = is_dead;
      if (send_args.device_context) {
        send_args.device_context->Ref();
        item->send_dev_context = send_args.device_context;
      }
      item->recv_dev_context = nullptr;

      // The allocator attributes of item->value.
      item->send_alloc_attrs = send_args.alloc_attrs;

      CHECK(table_.insert({key_hash, item}).second);
      return Status::OK();
    } else {
      item = iter->second;

      if (item->waiter == nullptr) {
        // There is already a message in the table under the key.
        // Should not happen unless it has a waiter.
        return errors::Aborted("Duplicated send: ", key.FullKey());
      }
      // Mark item as complete.
      item->has_been_recvd = true;

      // Get item->waiter function into waiter and set item->waiter to null
      std::swap(item->waiter, waiter);
      DCHECK(item->waiter == nullptr);
      DCHECK(waiter != nullptr);

      // The ref on recv_dev_context transfers below.
      recv_args.device_context = item->recv_dev_context;
      recv_args.alloc_attrs = item->recv_alloc_attrs;
      item->recv_dev_context = nullptr;
      if (tolerate_dup_recv_) {
        item->value = val;
        item->is_dead = is_dead;
        if (send_args.device_context) {
          send_args.device_context->Ref();
          item->send_dev_context = send_args.device_context;
        }
        item->send_alloc_attrs = send_args.alloc_attrs;
      }
    }
  }  // mutex
  // Notify the waiter by invoking its done closure, outside scope
  // of the table lock.
  // 直接运行接收者的回调函数，回调函数中进行数据传输操作
  waiter(Status::OK(), send_args, recv_args, val, is_dead);
  if (recv_args.device_context) recv_args.device_context->Unref();
  return Status::OK();
}
```

### 接收

应用层先寻找Rendezvous

/tensorflow/core/distributed_runtime/graph_mgr.cc:425

```c++
// 阻塞等待式接收
Status GraphMgr::RecvOutputs(const int64 step_id, NamedTensors* out) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = RecvOutputsFromRendezvous(rendezvous, out);
  rendezvous->Unref();
  return s;
}

Status GraphMgr::RecvOutputsFromRendezvous(Rendezvous* rendezvous,
                                           NamedTensors* out) {
  // Receives values requested by the caller.
  Rendezvous::ParsedKey parsed;
  for (auto& p : *out) {
    const string& key = p.first;
    Tensor* val = &p.second;
    bool is_dead = false;
    Status s = Rendezvous::ParseKey(key, &parsed);
    if (s.ok()) {
      s = rendezvous->Recv(parsed, Rendezvous::Args(), val, &is_dead);
    }
    if (is_dead) {
      s = errors::InvalidArgument("The tensor returned for ", key,
                                  " was not valid.");
    }
    if (!s.ok()) return s;
  }
  return Status::OK();
}

Status Rendezvous::Recv(const ParsedKey& key, const Args& recv_args,
                        Tensor* val, bool* is_dead, int64 timeout_ms) {
  Status ret;
  Notification n;
  RecvAsync(key, recv_args,
            [&ret, &n, val, is_dead](const Status& s, const Args& send_args,
                                     const Args& recv_args, const Tensor& v,
                                     const bool dead) {
              ret = s;
              *val = v;
              *is_dead = dead;
              n.Notify();
            });
  if (timeout_ms > 0) {
    int64 timeout_us = timeout_ms * 1000;
    bool notified = WaitForNotificationWithTimeout(&n, timeout_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    n.WaitForNotification();
  }
  return ret;
}

// 提供Callback的异步接收(RDMA重写此部分)
void GraphMgr::RecvOutputsAsync(const int64 step_id, NamedTensors* out,
                                StatusCallback done) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  RecvOutputsFromRendezvousAsync(rendezvous, out,
                                 [done, rendezvous](const Status s) {
                                   rendezvous->Unref();
                                   done(s);
                                 });
}

void GraphMgr::RecvOutputsFromRendezvousAsync(Rendezvous* rendezvous,
                                              NamedTensors* out,
                                              const StatusCallback& done) {
  if (out->empty()) {
    done(Status::OK());
    return;
  }
  // We compute the args before calling RecvAsync because we need to ensure that
  // out isn't being iterated over after done is called, since done deletes out.
  std::vector<std::tuple<string, Tensor*, Rendezvous::ParsedKey>> args;
  for (auto& p : *out) {
    Rendezvous::ParsedKey parsed;
    Status s = Rendezvous::ParseKey(p.first, &parsed);
    if (!s.ok()) {
      done(s);
      return;
    }
    args.push_back(std::make_tuple(p.first, &p.second, parsed));
  }

  typedef struct {
    mutex mu;
    int done_counter;
    Status shared_status = Status::OK();
  } CallState;
  CallState* call_state = new CallState;
  call_state->done_counter = out->size();
  for (auto& p : args) {
    const string& key = std::get<0>(p);
    Tensor* val = std::get<1>(p);
    Rendezvous::ParsedKey parsed = std::get<2>(p);
    rendezvous->RecvAsync(
        parsed, Rendezvous::Args(),
        [val, done, key, call_state](const Status& s,
                                     const Rendezvous::Args& send_args,
                                     const Rendezvous::Args& recv_args,
                                     const Tensor& v, const bool is_dead) {
          Status status = s;
          if (status.ok()) {
            *val = v;
            if (is_dead) {
              status = errors::InvalidArgument("The tensor returned for ", key,
                                               " was not valid.");
            }
          }
          call_state->mu.lock();
          call_state->shared_status.Update(status);
          call_state->done_counter--;
          // If we are the last async call to return, call the done callback.
          if (call_state->done_counter == 0) {
            const Status& final_status = call_state->shared_status;
            call_state->mu.unlock();
            done(final_status);
            delete call_state;
            return;
          }
          call_state->mu.unlock();
        });
  }
}
```

RdmaRemoteRendezvous中没有重写RecvAsync，故调用基类的RecvAsync函数

不管是阻塞式接收还是Callback式接收，都调用以下函数。阻塞式接收的Callback函数自动生成为notify函数。

```c++
// This method is called only by the RecvOp.  It tests to see
// whether the value will be produced by a local or remote device
// and handles accordingly.  In the local case it forwards to
// local_, in the remote case it initiates an RPC request.
void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
  VLOG(1) << "RemoteRendezvous Recv " << this << " " << parsed.FullKey();
  CHECK(is_initialized()) << "RecvAsync called when uninitialized.";
  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const Status& s) {
            done(s, send_args, recv_args, *out, is_dead);
            delete out;
          };

          if (status.ok()) {
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               std::move(final_callback));
          } else {
            final_callback(status);
          }
        });
    return;
  } else {
    RecvFromRemoteAsync(parsed, recv_args, std::move(done));
  }
}
```

若是本地操作，调用LocalRendezvousImpl类的RecvAsync。
