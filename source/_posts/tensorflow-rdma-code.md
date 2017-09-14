---
title: TensorFlow中rdma的使用
date: 2017-09-14 15:59:32
tags: tensorflow rdma
categories: rdma
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

若是远程操作，调用RDMA中的重写部分。

```c++
void RdmaRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  Status s;
  // parse src_name and dst_name
  string src_name, dst_name, unused;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                        &unused)) {
    s = errors::Internal("Could not parse src name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  if (!DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &unused)) {
    s = errors::Internal("Could not parse dst name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);
  RdmaChannel* rc = rdma_mgr_->FindChannel(src_name);
  string key(std::move(parsed.FullKey().ToString()));
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);
  // insert callback
  rc->InsertRecvCallback(key_with_step_id, [this, key, key_with_step_id, rc,
                                            recv_args, parsed, done]() {
    Status s;
    Device* src_dev;
    s = env_->device_mgr->LookupDevice("CPU:0", &src_dev);
    CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
    if (!s.ok()) {
      done(s, Args(), recv_args, Tensor(), true);
      return;
    }
    Device* dst_dev;
    s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
    CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
    if (!s.ok()) {
      done(s, Args(), recv_args, Tensor(), true);
      return;
    }
    RdmaBuffer* rb = rc->FindBuffer(key);
    RdmaMessage rm;
    CHECK(rb->size_ >= RdmaMessage::kMessageTotalBytes);
    RdmaMessage::ParseMessage(rm, rb->buffer_);
    CHECK(rm.type_ == RDMA_MESSAGE_TENSOR_WRITE);
    Tensor val;
    if (!rm.is_dead_) {
      void* input = static_cast<char*>(rb->buffer_) +
                    RdmaMessage::kTensorBufferStartIndex;
      // 反向解析proto
      TensorProto proto;
      CHECK(rm.tensor_bytes_ + RdmaMessage::kTensorBufferStartIndex <=
            rb->size_);
      CHECK(ParseProtoUnlimited(&proto, input, rm.tensor_bytes_))
          << "fail to parse proto from array";
      s = dst_dev->MakeTensorFromProto(proto, recv_args.alloc_attrs, &val);
    }

    rc->RemoveRecvCallback(key_with_step_id);
    // create message
    RdmaMessage br;
    br.type_ = RDMA_MESSAGE_BUFFER_IDLE;
    br.name_size_ = key.size();
    br.name_ = key;
    string message = RdmaMessage::CreateMessage(br);
    RdmaBuffer* tb = rc->tx_message_buffer_;
    tb->EnqueueItem(message);
    tb->SendNextItem();
    done(s, Args(), recv_args, val, rm.is_dead_);
  });
  // append key to message queue
  // 将Callback放入Channel后，发送TENSOR_REQUEST消息。
  RdmaBuffer* rb = rc->tx_message_buffer_;
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_REQUEST;
  rm.name_size_ = key.size();
  rm.name_ = key; // key为tensor buffer名称
  rm.step_id_ = step_id_; //标识tensor
  string message = RdmaMessage::CreateMessage(rm);
  rb->EnqueueItem(message);
  rb->SendNextItem();
}


// Send the next tensor from the buffer's job queue.
void RdmaTensorBuffer::SendNextItem() {
  // get the key
  string key_with_step_id = "";
  {
    mutex_lock lock{mu_};
    if (!queue_.empty()) {
      key_with_step_id = queue_.front();
      queue_.pop();
    }
  }
  // send the tensor if a key is acquired.
  if (key_with_step_id != "") {
    VLOG(2) << "try to send tensor: " << key_with_step_id;
    string key;
    int64 step_id;
    VerbsUtil::GetKeyAndStepId(key_with_step_id, key, step_id);
    CHECK(key.compare(name_) == 0);
    Rendezvous::ParsedKey parsed;
    Rendezvous::ParseKey(key, &parsed);
    Rendezvous::DoneCallback cb = [this, key_with_step_id, key, step_id,
                                   parsed](const Status& status,
                                           const Rendezvous::Args& send_args,
                                           const Rendezvous::Args& recv_args,
                                           const Tensor& in, bool is_dead) {
      CHECK(status.ok()) << "RecvLocalAsync was not ok, key" << key_with_step_id
                         << " error message: " << status.error_message();
      size_t buffer_size = RdmaMessage::kMessageTotalBytes;
      size_t tensor_bytes = 0;
      TensorProto proto;
      // Figures out which device the tensor is hosted on.
      Device* src_dev = nullptr;
      Status s = channel_->adapter_->worker_env_->device_mgr->LookupDevice(
          parsed.src_device, &src_dev);
      CHECK(s.ok()) << "src device not found";
      // Does the device have the right incarnation number we expect?
      CHECK(src_dev->attributes().incarnation() == parsed.src_incarnation)
          << "RecvTensor expects a different device incarnation: "
          << parsed.src_incarnation << " vs. "
          << src_dev->attributes().incarnation()
          << ". Your worker job was probably restarted. Check your "
          << "worker job for the reason why it was restarted.";
      Device* dst_dev = nullptr;
      // destination is on CPU.
      s = channel_->adapter_->worker_env_->device_mgr->LookupDevice("CPU:0",
                                                                    &dst_dev);
      CHECK(s.ok()) << "dst device not found";
      AllocatorAttributes dst_alloc_attr;
      dst_alloc_attr.set_on_host(true);
      // string tensor needs to be serialized
      if (src_dev->tensorflow_gpu_device_info() &&
          (!send_args.alloc_attrs.on_host())) {
        CHECK(send_args.device_context)
            << "send dev name: " << src_dev->name()
            << " gpu_info: " << src_dev->tensorflow_gpu_device_info();
        // "val" is on a GPU. Uses GPUUtil to fill the proto.
        s = VerbsUtil::SetProtoFromGPUSync(
            in, src_dev, send_args.device_context, &proto, is_dead);
        CHECK(s.ok()) << "set proto from gpu sync";
      } else {
        // tensor is in CPU memory.
        in.AsProtoTensorContent(&proto);
      }
      tensor_bytes = proto.ByteSize();
      // maybe some margin for string tensor?
      buffer_size += tensor_bytes;
      // prepare message
      RdmaMessage rm;
      rm.name_size_ = key.size();
      rm.name_ = key;
      rm.tensor_shape_ = in.shape();
      rm.data_type_ = in.dtype();
      rm.step_id_ = step_id;
      rm.is_dead_ = is_dead;
      rm.tensor_bytes_ = tensor_bytes;
      rm.buffer_size_ = buffer_size;
      mu_.lock();
      if (local_status_ == none ||
          (buffer_size > size_ && local_status_ == idle &&
           remote_status_ == idle)) {
        if ((local_status_ != none) && (buffer_size > size_)) {
          CHECK(rm.data_type_ == DT_STRING)
              << "Only string tensor allows to change size";
        }
        CreateCPUBuffer(buffer_size, false);
        mu_.unlock();
        // put back the key since it is not sent;
        EnqueueItem(key_with_step_id);
        // ask the remote to create the same buffer
        rm.type_ = RDMA_MESSAGE_BUFFER_REQUEST;
        rm.remote_addr_ = reinterpret_cast<uint64_t>(buffer_);
        rm.rkey_ = self_->rkey;
        string message = RdmaMessage::CreateMessage(rm);
        channel_->tx_message_buffer_->EnqueueItem(message);
        channel_->tx_message_buffer_->SendNextItem();
      } else if ((local_status_ == idle) && (remote_status_ == idle)) {
        // both buffers are ready, send the tensor
        local_status_ = busy;
        remote_status_ = busy;
        // local/remote_status_ won't be set back to idle
        // unitl Write() is successful
        mu_.unlock();
        CHECK((buffer_size == size_ && rm.data_type_ != DT_STRING) ||
              (buffer_size <= size_ && rm.data_type_ == DT_STRING))
            << "tensor and buffer size do not agree!"
            << " buffer_size = " << size_
            << " requested tensor size = " << buffer_size << in.DebugString();
        uint32_t imm_data = LookupBufferIndex(key);
        rm.type_ = RDMA_MESSAGE_TENSOR_WRITE;
        string message = RdmaMessage::CreateMessage(rm);
        memcpy(buffer_, message.data(), message.size());
        if (!is_dead) {
          // copy the tensor buffer content
          void* output =
              static_cast<void*>(static_cast<char*>(buffer_) +
                                 RdmaMessage::kTensorBufferStartIndex);
          CHECK(tensor_bytes + RdmaMessage::kTensorBufferStartIndex <= size_);
          proto.SerializeToArray(output, tensor_bytes);
        } else {
          buffer_size = RdmaMessage::kMessageTotalBytes;
        }
        Write(imm_data, buffer_size);
      } else {
        mu_.unlock();
        // put back the key since it is not sent;
        EnqueueItem(key_with_step_id);
      }
    };
    channel_->adapter_->worker_env_->rendezvous_mgr
        ->RecvLocalAsync(step_id, parsed, cb);
  }
}


void BaseRendezvousMgr::RecvLocalAsync(int64 step_id,
                                       const Rendezvous::ParsedKey& parsed,
                                       Rendezvous::DoneCallback done) {
  BaseRemoteRendezvous* rendez = FindOrCreate(step_id);
  using namespace std::placeholders;
  Rendezvous::DoneCallback done_cb = std::bind(
      [rendez](Rendezvous::DoneCallback done,
               // Begin unbound arguments.
               const Status& s, const Rendezvous::Args& send_args,
               const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
        rendez->Unref();
        done(s, send_args, recv_args, v, dead);
      },
      std::move(done), _1, _2, _3, _4, _5);
  rendez->RecvLocalAsync(parsed, std::move(done_cb));
}

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed,
                                          DoneCallback done) {
  {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // RecvLocalAsync can be called (due to an incoming RecvTensor RPC from a
      // remote worker) before the RunStep (or PartialRunStep) RPC from the
      // master arrives. RecvLocalAsync thus buffers the arguments until after
      // the RemoteRendezvous is Initialize()'d, when it completes the
      // rendezvous logic. At some point after Initialize() is called, a Tensor
      // is produced locally that will then be sent in response to the incoming
      // RPC.
      DeferredCall call(parsed, std::move(done));
      deferred_calls_.push_back(call);
      return;
    }
  }
  RecvLocalAsyncInternal(parsed, std::move(done));
}

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed,
                                                  DoneCallback done) {
  Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_->RecvAsync(parsed, Args(), std::move(done));
}
```

```c++
// Callback provided by a tensor consumer waiting on the rendezvous.
// It will be invoked when the tensor is available, or when a non-OK
// status arises in the production of that tensor.  It also gets
// two Rendezvous::Args, one provided by the sender, the other by the
// receiver, which may be needed when a non-CPU device is in use
// by either side.
typedef std::function<void(const Status&, const Args&, const Args&,
                           const Tensor&, const bool)>
  DoneCallback;

void RecvAsync(const ParsedKey& key, const Args& recv_args,
               DoneCallback done) override {
  uint64 key_hash = KeyHash(key.FullKey()); // 获取tensor的hash值
  VLOG(2) << "Recv " << this << " " << key_hash << " " << key.FullKey();
  mu_.lock();
  if (!status_.ok()) {
    // Rendezvous has been aborted.
    Status s = status_;
    mu_.unlock();
    done(s, Args(), recv_args, Tensor(), false); // 运行回调函数，但是status为坏
    return;
  }
  Table::iterator iter = table_.find(key_hash); // 在table中寻找有无接收者要的tensor
  if (iter != table_.end()) { // 若有，则
    Item* item = iter->second; // 得到该tensor
    if (item->has_been_recvd && !tolerate_dup_recv_) { // 判断该tensor是否被读取过和是否容忍重复读取
      mu_.unlock();
      done(errors::Aborted("Duplicated recv: ", key.FullKey()), Args(),
           recv_args, Tensor(), false);
    } else if (item->waiter == nullptr || tolerate_dup_recv_) { // 该Tensor没有接收者或者允许重复接收
      // A message has already arrived and is stored in the table
      // under this key.  Consumes the message and invokes the done
      // closure.
      Tensor v = item->value;
      if (!tolerate_dup_recv_) { // 若不允许重复接收，则将table中的值清空
        item->value = Tensor();
      }
      item->has_been_recvd = true;
      // Before dropping the table lock, capture the item values.
      // DeviceContext is only non-null for non-CPU devices.
      // If we capture the send_dev_context, we need to hold a ref on
      // it.  Our caller will have a ref on the recv_dev_context,
      // which is not in our table.
      DeviceContext* send_dev_context = item->send_dev_context;
      if (send_dev_context) send_dev_context->Ref();
      bool is_dead = item->is_dead;
      Args send_args;
      send_args.device_context = item->send_dev_context;
      send_args.alloc_attrs = item->send_alloc_attrs;
      mu_.unlock();
      // 运行回调函数
      done(Status::OK(), send_args, recv_args, v, is_dead);
      if (send_dev_context) send_dev_context->Unref();
    } else {
      // Already have a waiter in the waiters table under this key,
      // which should not happen.
      mu_.unlock();
      done(errors::Aborted("Duplicated recv: ", key.FullKey()), Args(),
           recv_args, Tensor(), false);
    }
    return;
  }
  // Waiting for a message that has not arrived yet. Insert into the
  // waiting table. The done closure will be invoked when the
  // message arrives.
  // 若无tensor，则将hash+回调函数放入table中。
  Item* item = new Item;
  item->waiter = std::move(done);
  item->recv_alloc_attrs = recv_args.alloc_attrs;
  if (recv_args.device_context) {
    item->recv_dev_context = recv_args.device_context;
    item->recv_dev_context->Ref();
  }
  CHECK(table_.insert({key_hash, item}).second);
  mu_.unlock();
  return;
}
```

```c++
// Rdma-Write the content of the buffer
void RdmaBuffer::Write(uint32_t imm_data, size_t buffer_size) {
  struct ibv_sge list;
  list.addr = (uint64_t)buffer_;
  list.length = buffer_size;
  list.lkey = self_->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)this;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = (uint64_t)remote_.remote_addr;
  wr.wr.rdma.rkey = remote_.rkey;

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel_->qp_, &wr, &bad_wr)) << "Failed to post send";
}
```

接收详细操作：

```flow
st=>start: start
GraphMgr_RecvOutputsAsync=>subroutine: GraphMgr::RecvOutputsAsync
(step_id, outTensor, callback1)
GraphMgr_RecvOutputsFromRendezvousAsync=>subroutine: GraphMgr::RecvOutputsFromRendezvousAsync
(由redezvousMgr通过step_id找到的redezvous, 由callback1封装成的callback2)
BaseRemoteRendezvous_RecvAsync=>subroutine: BaseRemoteRendezvous::RecvAsync
(parsedKey, callback2)
BaseRemoteRendezvous_RecvAsync_Cond=>condition: src == dest
LocalRendezvousImpl_RecvAsync1=>subroutine: LocalRendezvousImpl::RecvAsync
(parsedKey, callback3(封装callback2))
RdmaRemoteRendezvous_RecvFromRemoteAsync=>subroutine: RdmaRemoteRendezvous::RecvFromRemoteAsync
(parsedKey, callback2) | approved:> www.baidu.com[blank]
BaseRendezvousMgr_RecvLocalAsync1=>subroutine: BaseRendezvousMgr::RecvLocalAsync
(step_id, 通过key新生成的parsed, rdma callback1)
BaseRendezvousMgr_RecvLocalAsync2=>operation: 通过step_id查找BaseRemoteRendezvous
同时将rdma callback1封装为rdma callback2
BaseRendezvousMgr_RecvLocalAsync3=>subroutine: BaseRendezvousMgr::RecvLocalAsync
(parsed, rdma callback2)
BaseRendezvousMgr_RecvLocalAsync4=>condition: 判断该Rendezvous
是否被init
BaseRendezvousMgr_RecvLocalAsync4_No=>operation: 将parsed和rdma callback2
组装为DeferredCall放入对列中
直到Rendezvous被初始化
BaseRemoteRendezvous_RecvLocalAsyncInternal=>subroutine: BaseRemoteRendezvous::RecvLocalAsyncInternal
(parsed, rdma callback2)
LocalRendezvousImpl_RecvAsync2=>subroutine: LocalRendezvousImpl::RecvAsync
(parsedKey, rdma callback2)
LocalRendezvousImpl_RecvAsync=>subroutine: 在Table中寻找tensor
LocalRendezvousImpl_RecvAsync_Cond=>condition: 是否找到
LocalRendezvousImpl_RecvAsync_Cond_No=>subroutine: 存储callback
等待tensor
LocalRendezvousImpl_RecvAsync_Cond_Yes=>subroutine: 运行callback | approved:> www.baidu.com[blank]
e=>end: end

st->GraphMgr_RecvOutputsAsync->GraphMgr_RecvOutputsFromRendezvousAsync->BaseRemoteRendezvous_RecvAsync->BaseRemoteRendezvous_RecvAsync_Cond
BaseRemoteRendezvous_RecvAsync_Cond(no,bottom)->RdmaRemoteRendezvous_RecvFromRemoteAsync->BaseRendezvousMgr_RecvLocalAsync1->BaseRendezvousMgr_RecvLocalAsync2->BaseRendezvousMgr_RecvLocalAsync3->BaseRendezvousMgr_RecvLocalAsync4
BaseRendezvousMgr_RecvLocalAsync4(yes)->BaseRemoteRendezvous_RecvLocalAsyncInternal->LocalRendezvousImpl_RecvAsync2
BaseRendezvousMgr_RecvLocalAsync4(no)->BaseRendezvousMgr_RecvLocalAsync4_No
LocalRendezvousImpl_RecvAsync2->LocalRendezvousImpl_RecvAsync
BaseRemoteRendezvous_RecvAsync_Cond(yes)->LocalRendezvousImpl_RecvAsync1->LocalRendezvousImpl_RecvAsync
LocalRendezvousImpl_RecvAsync->LocalRendezvousImpl_RecvAsync_Cond
LocalRendezvousImpl_RecvAsync_Cond(yes, left)->LocalRendezvousImpl_RecvAsync_Cond_Yes->e
LocalRendezvousImpl_RecvAsync_Cond(no)->LocalRendezvousImpl_RecvAsync_Cond_No->e
```

```sequence
Title: RdmaRemoteRendezvous::RecvFromRemoteAsync
note left of Receiver_Recv(): 将CallBack2封装成Callback3\n后放入本地Channel中
note left of Receiver_Recv(): 将RDMA_MESSAGE_TENSOR_REQUEST\n消息放入tx_message_buffer
Receiver_Process_CQ()->Receiver_Process_CQ(): null
Receiver_Recv()->Sender_Process_CQ(): 发送RDMA_MESSAGE_TENSOR_REQUEST消息
note right of Sender_Process_CQ(): 收到消息
Sender_Process_CQ()->Receiver_Process_CQ(): 回复RDMA_MESSAGE_ACK消息
note right of Sender_Process_CQ(): 检测本地有无该Tensor(利用key搜索)\n将(key+step_id)组合入Queue\n即将发送数据
note right of Sender_Process_CQ(): 从Queue中获得key_with_step_id\n同时创建新的rdma callback1\n调用BaseRendezvousMgr::RecvLocalAsync\n(step_id, 通过key新生成的parsed, rdma callback1)
```

1. 执行GraphMgr::RecvOutputsAsync，输入参数包含step_id、输出tensor和callback1。
1. 执行GraphMgr::RecvOutputsFromRendezvousAsync，输入参数由step_id替换为从rendezvous_mgr找到的redezvous。函数中用callback2封装了callback1。
1. 执行BaseRemoteRendezvous::RecvAsync，输入参数主要有parsed key和callback2。
1. 判断src和dest。
   1. 若src和dest一致，则调用LocalRendezvousImpl::RecvAsync，参数主要有parsed key和callback3(封装callback2)。
   1. 若不一致，调用RdmaRemoteRendezvous::RecvFromRemoteAsync。参数不变。
      1. 接收方Recv函数：将CallBack2封装成Callback3后放入本地Channel中。
      1. 接收方Recv函数：将RDMA_MESSAGE_TENSOR_REQUEST消息放入tx_message_buffer。
      1. 接收方Message Buffer：发送消息。
      1. 发送方Process_CQ函数：收到RDMA_MESSAGE_TENSOR_REQUEST。
      1. 发送方Ack Buffer：回复RDMA_MESSAGE_ACK消息。
      1. 发送方Process_CQ函数：检测本地有无该Tensor(利用key搜索)，将(key+step_id)组合入Queue，即将发送。
      1. 发送方Tensor Buffer：从Queue中获得key_with_step_id，同时创建新的rdma callback1。调用BaseRendezvousMgr::RecvLocalAsync(三参数)，输入参数有step_id、通过key新生成的parsed 和 rdma callback1。
      1. BaseRendezvousMgr::RecvLocalAsync(三参数)通过step_id查找BaseRemoteRendezvous，同时将rdma callback1封装为rdma callback2。调用BaseRendezvousMgr::RecvLocalAsync(二参数)。
      1. BaseRendezvousMgr::RecvLocalAsync(二参数)将判断该Rendezvous是否被init，若未init，则将parsed和rdma callback2组装为DeferredCall放入对列中，直到Rendezvous被初始化。若已经INIT，则执行BaseRemoteRendezvous::RecvLocalAsyncInternal，输入参数不变(parsed和rdma callback2)。
      1. 调用LocalRendezvousImpl::RecvAsync，参数主要有parsed和rdma callback2。同4.1
1. 在Table中寻找tensor。
   1. 未找到，存储回调，等待tensor。
   1. 若找到，则运行回调。若为RDMA回调，则运行如下：
      1. 将Tensor转为Proto。
      1. 若local和remote状态没准备好，则将(key+step_id)入队，对应4.2.7。
      1. 若local和remote状态就绪，但是tensor buffer空间不够。
         1. 自己创建足够大的buffer。
         1. 将(key+step_id)入队，对应4.2.7。
         1. 创建RDMA_MESSAGE_BUFFER_REQUEST消息，并发送之。
         1. 接收方收到后回复Ack
         1. ...
      1. 若一切就绪。
         1. 创建RDMA_MESSAGE_TENSOR_WRITE消息，放入proto。
         1. 用IBV_WR_RDMA_WRITE_WITH_IMM写入远程信息。

### LOG

```c++
#define CHECK(condition)              \
  if (TF_PREDICT_FALSE(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

#define LOG(severity) _TF_LOG_##severity

#define _TF_LOG_INFO \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)
#define _TF_LOG_WARNING \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::WARNING)
#define _TF_LOG_ERROR \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::ERROR)
#define _TF_LOG_FATAL \
  ::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__)


#define VLOG(lvl)      \
  if (TF_PREDICT_FALSE(VLOG_IS_ON(lvl))) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

#define VLOG_IS_ON(lvl) \
  ((lvl) <= ::tensorflow::internal::LogMessage::MinVLogLevel())

int64 LogMessage::MinVLogLevel() {
  static int64 min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

int64 MinVLogLevelFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

// Parse log level (int64) from environment variable (char*)
int64 LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string min_log_level(tf_env_var_val);
  std::istringstream ss(min_log_level);
  int64 level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}
```

### 结构体

#### WorkEnv

/tensorflow/core/distributed_runtime/worker_env.h

```c++
// The worker environment class, which holds a bag of pointers to
// per-worker singletons.
//
// WorkerEnv does not own its member pointers.
struct WorkerEnv {
  Env* env = nullptr;

  // session_mgr encapsulates state for each session.
  SessionMgr* session_mgr = nullptr;

  // The local devices of this worker. Devices are owned by the device_mgr.
  //
  // REQUIRES: !local_devices.empty().
  std::vector<Device*> local_devices;

  // device_mgr manages local devices (cpu and gpu). The WorkerService
  // is the network interface for managed devices.
  DeviceMgr* device_mgr = nullptr;

  // A set of rendezvous keyed by step ids.
  RendezvousMgrInterface* rendezvous_mgr = nullptr;

  // A pool of threads for scheduling compute work.
  thread::ThreadPool* compute_pool = nullptr;
};
```

#### Item

/tensorflow/core/framework/rendezvous.cc

```c++
struct Item {
  DoneCallback waiter = nullptr;
  Tensor value;
  bool is_dead = false;
  bool has_been_recvd = false;
  DeviceContext* send_dev_context = nullptr;
  DeviceContext* recv_dev_context = nullptr;
  AllocatorAttributes send_alloc_attrs;
  AllocatorAttributes recv_alloc_attrs;

  ~Item() {
    if (send_dev_context) {
      send_dev_context->Unref();
    }
    if (recv_dev_context) {
      recv_dev_context->Unref();
    }
  }
};
```

#### ParseKey

```c++
// Constructs a rendezvous key for the tensor of "name" sent from
// "src_device" to "dst_device". The tensor is generated in the frame
// and iteration specified by "frame_iter".
/*  static */
string Rendezvous::CreateKey(const string& src_device, uint64 src_incarnation,
                             const string& dst_device, const string& name,
                             const FrameAndIter& frame_iter) {
  // NOTE: ';' is not used in the device name's job name.
  //
  // We include both sender and receiver in the key to facilitate
  // debugging. For correctness, we only need to encode the receiver.
  //
  // "src_incarnation" is used to distinguish a worker when it
  // restarts.
  char buf[strings::kFastToBufferSize];
  return strings::StrCat(
      src_device, ";", strings::Uint64ToHexString(src_incarnation, buf), ";",
      dst_device, ";", name, ";", frame_iter.frame_id, ":", frame_iter.iter_id);
}

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
```

### 调试

10.42.10.36

```python
#coding=utf-8
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '5'

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  #server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index,protocol="grpc+verbs")
  issync = FLAGS.issync
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
      global_step = tf.Variable(0, name='global_step', trainable=False)

      input = tf.placeholder("float")
      label = tf.placeholder("float")

      weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
      biase  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
      pred = tf.multiply(input, weight) + biase

      loss_value = loss(label, pred)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)

      grads_and_vars = optimizer.compute_gradients(loss_value)
      if issync == 1:
        #同步模式计算更新梯度
        rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=len(
                                                  worker_hosts),
                                                replica_id=FLAGS.task_index,
                                                total_num_replicas=len(
                                                  worker_hosts),
                                                use_locking=True)
        train_op = rep_op.apply_gradients(grads_and_vars,
                                       global_step=global_step)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
      else:
        #异步模式计算更新梯度
        train_op = optimizer.apply_gradients(grads_and_vars,
                                       global_step=global_step)


      init_op = tf.initialize_all_variables()

      saver = tf.train.Saver()
      tf.summary.scalar('cost', loss_value)
      summary_op = tf.summary.merge_all()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir="./checkpoint/",
                            init_op=init_op,
                            summary_op=None,
                            saver=saver,
                            global_step=global_step,
                            save_model_secs=60)

    with sv.prepare_or_wait_for_session(server.target) as sess:
      # 如果是同步模式
      if FLAGS.task_index == 0 and issync == 1:
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_token_op)
      step = 0
      while  step < 1000000:
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.33  + 10
        _, loss_v, step = sess.run([train_op, loss_value,global_step], feed_dict={input:train_x, label:train_y})
        if step % steps_to_validate == 0:
          w,b = sess.run([weight,biase])
          print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))

    sv.stop()

def loss(label, pred):
  return tf.square(label - pred)

if __name__ == "__main__":
  tf.app.run()
```

PS端

```shell
docker run -it \
--rm \
-v nvidia_driver_367.57:/usr/local/nvidia:ro \
--device=/dev/nvidiactl \
--device=/dev/nvidia-uvm \
--device=/dev/nvidia-uvm-tools \
-e LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-375:/usr/lib32/nvidia-375 \
-e LIBRARY_PATH=/usr/local/cuda/lib64/stubs: \
-v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
-v /usr/lib/x86_64-linux-gnu/libcuda.so.375.39:/usr/lib/x86_64-linux-gnu/libcuda.so.375.39 \
-v /usr/lib32/nvidia-375:/usr/lib32/nvidia-375 \
-v /usr/lib/nvidia-375:/usr/lib/nvidia-375 \
--privileged \
-p 2227:2227 \
-v /etc/libibverbs.d:/etc/libibverbs.d:ro \
-v /usr/lib/libibverbs:/usr/lib/libibverbs:ro \
-v /usr/lib/libibverbs.so.1:/usr/lib/libibverbs.so.1:ro \
-v /usr/lib/librxe-rdmav2.so:/usr/lib/librxe-rdmav2.so:ro \
-v /sys/class/infiniband_verbs:/sys/class/infiniband_verbs:ro \
-v /lib/x86_64-linux-gnu/libnl-3.so.200:/lib/x86_64-linux-gnu/libnl-3.so.200:ro \
-v /usr/lib/x86_64-linux-gnu/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1:ro \
-v /usr/lib/x86_64-linux-gnu/libnl-route-3.so.200:/usr/lib/x86_64-linux-gnu/libnl-route-3.so.200:ro \
-v /root/TF/distributeTensorflowExample:/Example \
--name rdma-test-ps \
tensorflow:latest-devel-gpu-rdma-hadoop
```

进入容器后执行

```shell
cd /Example
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=0.0.0.0:2227 --worker_hosts=10.42.10.36:2228 --job_name=ps --task_index=0
```

Worker端

```shell
docker run -it \
--rm \
-v nvidia_driver_367.57:/usr/local/nvidia:ro \
--device=/dev/nvidiactl \
--device=/dev/nvidia-uvm \
--device=/dev/nvidia-uvm-tools \
-e LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia-375:/usr/lib32/nvidia-375 \
-e LIBRARY_PATH=/usr/local/cuda/lib64/stubs: \
-v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
-v /usr/lib/x86_64-linux-gnu/libcuda.so.375.39:/usr/lib/x86_64-linux-gnu/libcuda.so.375.39 \
-v /usr/lib32/nvidia-375:/usr/lib32/nvidia-375 \
-v /usr/lib/nvidia-375:/usr/lib/nvidia-375 \
--privileged \
-p 2228:2228 \
-v /etc/libibverbs.d:/etc/libibverbs.d:ro \
-v /usr/lib/libibverbs:/usr/lib/libibverbs:ro \
-v /usr/lib/libibverbs.so.1:/usr/lib/libibverbs.so.1:ro \
-v /usr/lib/librxe-rdmav2.so:/usr/lib/librxe-rdmav2.so:ro \
-v /sys/class/infiniband_verbs:/sys/class/infiniband_verbs:ro \
-v /lib/x86_64-linux-gnu/libnl-3.so.200:/lib/x86_64-linux-gnu/libnl-3.so.200:ro \
-v /usr/lib/x86_64-linux-gnu/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1:ro \
-v /usr/lib/x86_64-linux-gnu/libnl-route-3.so.200:/usr/lib/x86_64-linux-gnu/libnl-route-3.so.200:ro \
-v /root/TF/distributeTensorflowExample:/Example \
--name rdma-test-worker \
tensorflow:latest-devel-gpu-rdma-hadoop
```

进入容器后执行

```shell
cd /Example
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=10.42.10.36:2227 --worker_hosts=0.0.0.0:2228 --job_name=worker --task_index=0
```
