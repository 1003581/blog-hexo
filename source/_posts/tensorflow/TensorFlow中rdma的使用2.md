---
title: TensorFlow中rdma的使用2
date: 2017-09-14 15:59:32
tags: 
- tensorflow
- rdma
categories: tensorflow
---

接上文
<!-- more -->
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
