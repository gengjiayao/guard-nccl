# NCCL 接收方驱动的流量控制

## 概述

本文档描述了在 NCCL 网络传输层中实现的**接收方驱动、基于速率的流量控制**机制的完整实现。接收方维护一个入向活跃流列表，为每条流计算公平份额的授权发送速率，并通过专用 TCP 控制套接字将速率推送给对应的发送方。发送方使用令牌桶（Token Bucket）算法强制执行该速率。

**适用范围**：仅应用于 P2P 操作（AlltoAll、ncclSend/ncclRecv）。集合通信操作（AllReduce、AllGather、ReduceScatter 使用的 Ring/Tree/NVLS/PAT 算法）不受影响，因为它们要求所有 rank 全局同步推进。

## 架构

```
                    连接建立阶段
                    ============

Rank A（接收方）                                Rank B（发送方）
────────────────                                ────────────────
recvProxySetup:
  ncclNet->listen() → netHandle
  ncclSocket listen → controlAddr
  返回 {netHandle, controlAddr}
         ──── connectInfo 交换 ────→
                                               sendProxyConnect:
                                                 ncclNet->connect(netHandle)
                                                 ncclSocket connect(controlAddr)
recvProxyConnect:
  ncclNet->accept() → netRecvComm
  ncclSocket accept() → controlSock


                    数据传输阶段
                    ============

recvProxyProgress:                             sendProxyProgress:
  新流到达 → AddFlow()                            （速率尚未获知）
  rate = 总带宽 / 活跃流数
  ── ncclFcRateMsg{ch,op,rate} ──(TCP)──→       fcPollRateUpdates()
                                                 → 更新令牌桶速率
                                                 → isend() 受令牌桶限制
  收到字节 → ReportProgress()
    （DRR 配额检查 → 可能降级/轮转）
  ── ncclFcRateMsg{ch,op,newRate} ─(TCP)──→     → 速率已更新

  流完成 → RemoveFlow()
  rate' = 总带宽 / (N-1)
  ── ncclFcRateMsg{ch,op,rate'} ──(TCP)──→      剩余流加速
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NCCL_RECV_FC_BW_GBS` | 25 | 总接收带宽，单位 GB/s |
| `NCCL_RECV_FC_MAX_CONCURRENT` | 0（无限制） | 最大同时准入流数。0 = 不限制。 |
| `NCCL_RECV_FC_QUOTA_MB` | 4 | DRR 每轮字节配额，单位 MB。0 = 禁用 DRR 轮转。 |

## 修改的文件

### 新增文件

- `src/include/flowcontrol.h` — 流量控制头文件
- `src/flowcontrol.cc` — 流量控制实现

### 修改的文件

- `src/include/proxy.h` — 新增 `recvFlowControl` 和 `fcControlContext` 字段
- `src/proxy.cc` — 流量控制状态的初始化和清理
- `src/transport/net.cc` — 控制套接字建立、速率反馈、令牌桶限速
- `src/Makefile` — 将 `flowcontrol.cc` 加入构建

---

## 详细改动

### 1. `src/include/flowcontrol.h`（新增文件）

定义接收方流量控制接口的完整头文件。

#### 核心数据结构

```cpp
// 速率更新消息：从接收方 proxy → 发送方 proxy，通过 TCP 控制套接字传输
struct ncclFcRateMsg {
  uint32_t magic;           // NCCL_FC_RATE_MSG_MAGIC (0x4E464352)
  int senderRank;           // 流标识：发送方 rank
  int channelId;            // 流标识：通道 ID
  uint64_t opCount;         // 流标识：操作计数
  double authorizedRate;    // 接收方授权的速率，单位 字节/秒（0 = 暂停）
};
```

```cpp
// 回调函数类型 — 当流的速率发生变化时被调用。
// 在持有互斥锁的情况下被调用，实现必须是非阻塞的。
typedef void (*ncclFcRateChangeCallback_t)(void* context, const struct ncclFcRateMsg* msg);
```

```cpp
// 流状态
enum ncclFlowState {
  NCCL_FLOW_FREE     = 0,  // 槽位可用
  NCCL_FLOW_ACTIVE   = 1,  // 活跃流，拥有公平份额速率
  NCCL_FLOW_PENDING  = 2,  // 等待准入（速率 = 0）
};
```

```cpp
// 单条流的条目
struct ncclFlowEntry {
  struct ncclFlowId id;         // {senderRank, channelId, opCount}
  double authorizedRate;        // 字节/秒
  ssize_t totalBytes;           // 该流的总字节数
  ssize_t completedBytes;       // 进度追踪
  ssize_t remainingQuota;       // DRR 轮转配额
  enum ncclFlowState state;
  uint64_t arrivalOrder;        // FIFO 排序，用于 pending→active 提升
};
```

```cpp
// 每个接收方的流量控制状态
struct ncclRecvFlowControl {
  struct ncclFlowEntry flows[NCCL_FC_MAX_FLOWS];  // 512 个槽位
  int numActive;
  int numPending;
  int maxConcurrent;         // 0 = 无限制
  ssize_t quotaPerRound;     // DRR 配额（0 = 禁用）
  double totalRecvBandwidth; // 字节/秒

  ncclFcRateChangeCallback_t rateChangeCallback;
  void* rateChangeContext;

  std::mutex mutex;
};
```

#### API 函数

| 函数 | 说明 |
|------|------|
| `ncclFlowControlInit()` | 使用带宽和环境变量进行初始化 |
| `ncclFlowControlSetCallback()` | 注册速率变化回调 |
| `ncclFlowControlAddFlow()` | 注册新的入向流，返回初始速率 |
| `ncclFlowControlRemoveFlow()` | 移除已完成的流，重新计算速率 |
| `ncclFlowControlReportProgress()` | 报告已接收字节数，触发 DRR 轮转 |
| `ncclFlowControlGetRate()` | 查询某条流当前的速率 |
| `ncclFlowControlDestroy()` | 清理销毁 |

---

### 2. `src/flowcontrol.cc`（新增文件）

接收方流量控制逻辑的完整实现。

#### 核心速率计算

```cpp
// 每当流被添加/移除/轮转时调用
static void recalculateAndNotify(struct ncclRecvFlowControl* fc) {
  double fairRate = fc->totalRecvBandwidth / (double)fc->numActive;
  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state == NCCL_FLOW_ACTIVE) {
      fc->flows[i].authorizedRate = fairRate;
      notifyRateChange(fc, &fc->flows[i]);  // 触发回调 → 发送给发送方
    }
  }
}
```

#### 准入控制

- 当 `maxConcurrent == 0`（默认值）：所有流立即以 ACTIVE 状态准入
- 当 `maxConcurrent > 0`：超出的流进入 PENDING 状态（速率 = 0）
- PENDING 流按 FIFO 顺序被提升为 ACTIVE（当有 ACTIVE 槽位空出时）

#### DRR（亏损轮询调度）轮转

当 `quotaPerRound > 0`（默认值：4MB）：
1. 每条 ACTIVE 流拥有一个字节配额
2. 当配额耗尽且有 PENDING 流等待时：
   - 该流被**降级**回 PENDING（获得 `arrivalOrder = nextArrivalOrder++`，排到队尾）
   - 最老的 PENDING 流被**提升**为 ACTIVE
   - 向被降级的发送方发送 rate = 0 通知
   - 向所有 ACTIVE 发送方发送新的公平份额速率

#### 速率变化通知

每次速率变化（添加、移除、提升、降级、重新计算）都会触发：
```cpp
static void notifyRateChange(struct ncclRecvFlowControl* fc, const struct ncclFlowEntry* f) {
  if (fc->rateChangeCallback && f->state != NCCL_FLOW_FREE) {
    struct ncclFcRateMsg msg = makeRateMsg(f);
    fc->rateChangeCallback(fc->rateChangeContext, &msg);
  }
}
```

---

### 3. `src/include/proxy.h`（修改文件）

#### 改动

新增前向声明和两个字段到 `ncclProxyState`：

```diff
+struct ncclRecvFlowControl;
+
 // ... （已有代码） ...

 struct ncclProxyState {
   // ... （已有字段） ...

   struct ncclExpectedProxyResponse* expectedResponses;
+
+  // 接收方驱动的流量控制
+  struct ncclRecvFlowControl* recvFlowControl;
+  void* fcControlContext;  // ncclFcControlContext*，使用 void* 避免头文件依赖
 };
```

- `recvFlowControl`：每个 proxy 的流量控制状态，所有接收连接共享
- `fcControlContext`：senderRank → 控制套接字的映射，用于分发速率更新

---

### 4. `src/proxy.cc`（修改文件）

#### 初始化（在 `ncclProxyCreate` 中）

```diff
+#include "flowcontrol.h"

 // 在 ncclProxyCreate() 中，buffSizes 初始化之后：
+    proxyState->recvFlowControl = new ncclRecvFlowControl{};
+    double defaultBw = 25.0 * 1024.0 * 1024.0 * 1024.0;
+    NCCLCHECK(ncclFlowControlInit(proxyState->recvFlowControl, proxyState->tpRank, defaultBw));
+    proxyState->fcControlContext = nullptr;
```

#### 清理（在 `ncclProxyDestroy` 中）

```diff
+    if (sharedProxyState->recvFlowControl) {
+      ncclFlowControlDestroy(sharedProxyState->recvFlowControl);
+      delete sharedProxyState->recvFlowControl;
+    }
+    if (sharedProxyState->fcControlContext) {
+      free(sharedProxyState->fcControlContext);
+    }
```

---

### 5. `src/transport/net.cc`（修改文件 — 改动量最大）

该文件包含核心集成：控制套接字生命周期、速率反馈和令牌桶限速。

#### 5.1 新增头文件引入

```diff
+#include <time.h>
+#include <fcntl.h>
+#include <errno.h>
 #include "register_inline.h"
+#include "flowcontrol.h"
+#include "socket.h"
```

#### 5.2 新增数据结构（位于 `sendNetResources` 之前）

**令牌桶** — 发送方速率限制：

```cpp
struct ncclFcTokenBucket {
  double tokens;           // 可发送的字节数
  double rate;             // 授权速率，字节/秒（0 = 暂停）
  double burstSize;        // 最大令牌数（限制空闲后的突发量）
  struct timespec lastRefill;
  bool initialized;
};
```

**发送方流表** — 每个发送方的速率状态：

```cpp
#define NCCL_FC_SENDER_MAX_FLOWS 64

struct ncclFcSenderFlow {
  int channelId;
  uint64_t opCount;
  struct ncclFcTokenBucket bucket;
  bool active;
};

struct ncclFcSenderState {
  struct ncclFcSenderFlow flows[NCCL_FC_SENDER_MAX_FLOWS];
  struct ncclSocket controlSock;  // 连接到接收方控制监听套接字
  bool hasControlSock;
};
```

#### 5.3 扩展的资源结构体

**sendNetResources** — 新增发送方流量控制状态：

```diff
 struct sendNetResources {
   // ... 已有字段 ...
   size_t maxP2pBytes;
+
+  struct ncclFcSenderState fcState;
 };
```

**recvNetResources** — 新增用于速率反馈的控制套接字：

```diff
 struct recvNetResources {
   // ... 已有字段 ...
   size_t maxP2pBytes;
+
+  struct ncclSocket controlListenSock;  // 在 recvProxySetup 中创建
+  struct ncclSocket controlSock;        // 在 recvProxyConnect 中接受连接
+  bool hasControlSock;
 };
```

#### 5.4 扩展 `netSendConnectArgs`

```diff
 struct netSendConnectArgs {
   ncclNetHandle_t handle;
   ncclNetAttr_t netAttr;
+  union ncclSocketAddress fcControlAddr;  // 接收方的控制监听地址
+  int hasFcControl;                       // 是否有流量控制功能
 };
```

新增静态断言，验证扩展后的 connectInfo 能容纳所有字段：

```cpp
static_assert(sizeof(ncclNetHandle_t) + sizeof(int) + sizeof(union ncclSocketAddress) + sizeof(int) <= CONNECT_SIZE,
              "ncclConnect 空间不足以容纳 ncclNetHandle_t + useGdr + fcControlAddr + hasFcControl");
```

#### 5.5 `recvSetup` — 扩展响应

`recvProxySetup` 的响应现在包含控制套接字地址：

```diff
-  NCCLCHECK(ncclProxyCallBlocking(..., connectInfo, sizeof(ncclNetHandle_t)));
-  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
+  int recvSetupRespSize = sizeof(ncclNetHandle_t) + sizeof(union ncclSocketAddress) + sizeof(int);
+  char recvSetupResp[...];
+  NCCLCHECK(ncclProxyCallBlocking(..., recvSetupResp, recvSetupRespSize));
+  memcpy(connectInfo, recvSetupResp, sizeof(ncclNetHandle_t));
+  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
+  // 在 useGdr 之后复制流量控制地址
+  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t) + sizeof(int),
+         recvSetupResp + sizeof(ncclNetHandle_t),
+         sizeof(union ncclSocketAddress) + sizeof(int));
```

**connectInfo 内存布局**（共 256 字节可用）：

| 偏移量 | 大小 | 内容 |
|--------|------|------|
| 0 | 128 | `ncclNetHandle_t`（网络插件监听句柄） |
| 128 | 4 | `useGdr` 标志 |
| 132 | 28 | `union ncclSocketAddress`（控制套接字地址） |
| 160 | 4 | `hasFcControl` 标志 |
| 164 | 92 | （未使用） |

#### 5.6 `sendConnect` — 提取控制地址

```diff
     netSendConnectArgs args = {0};
     memcpy(&args.handle, connectInfo, sizeof(ncclNetHandle_t));
+    memcpy(&args.fcControlAddr,
+           (uint8_t*)connectInfo + sizeof(ncclNetHandle_t) + sizeof(int),
+           sizeof(union ncclSocketAddress));
+    memcpy(&args.hasFcControl,
+           (uint8_t*)connectInfo + sizeof(ncclNetHandle_t) + sizeof(int) + sizeof(union ncclSocketAddress),
+           sizeof(int));
```

#### 5.7 `recvProxySetup` — 创建控制监听套接字

```diff
-  if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
+  if (respSize != sizeof(ncclNetHandle_t) + sizeof(union ncclSocketAddress) + sizeof(int))
+    return ncclInternalError;
   NCCLCHECK(proxyState->ncclNet->listen(..., respBuff, &resources->netListenComm));
+
+  // 创建 TCP 控制监听套接字，用于流量控制速率反馈
+  resources->hasControlSock = false;
+  union ncclSocketAddress controlAddr;
+  memset(&controlAddr, 0, sizeof(controlAddr));
+  int hasFcControl = 0;
+  ncclResult_t fcRet = ncclSocketInit(&resources->controlListenSock, NULL, NCCL_SOCKET_MAGIC,
+                                      ncclSocketTypeProxy, proxyState->abortFlag, 0);
+  if (fcRet == ncclSuccess) {
+    fcRet = ncclSocketListen(&resources->controlListenSock);
+    if (fcRet == ncclSuccess) {
+      fcRet = ncclSocketGetAddr(&resources->controlListenSock, &controlAddr);
+      if (fcRet == ncclSuccess) hasFcControl = 1;
+    }
+  }
+  memcpy((uint8_t*)respBuff + sizeof(ncclNetHandle_t), &controlAddr, sizeof(controlAddr));
+  memcpy((uint8_t*)respBuff + sizeof(ncclNetHandle_t) + sizeof(union ncclSocketAddress),
+         &hasFcControl, sizeof(int));
```

#### 5.8 `sendProxyConnect` — 连接控制套接字

在数据连接建立后（`*done = 1`）：

```cpp
  resources->fcState.hasControlSock = false;
  memset(resources->fcState.flows, 0, sizeof(resources->fcState.flows));
  if (req->hasFcControl) {
    ncclResult_t fcRet = ncclSocketInit(&resources->fcState.controlSock, &req->fcControlAddr,
                                        NCCL_SOCKET_MAGIC, ncclSocketTypeProxy, proxyState->abortFlag, 0);
    if (fcRet == ncclSuccess) {
      fcRet = ncclSocketConnect(&resources->fcState.controlSock);
      if (fcRet == ncclSuccess) {
        // 设置为非阻塞，以便在 progress 循环中轮询
        int fd = -1;
        ncclSocketGetFd(&resources->fcState.controlSock, &fd);
        if (fd >= 0) {
          int flags = fcntl(fd, F_GETFL, 0);
          if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        resources->fcState.hasControlSock = true;
      }
    }
  }
```

#### 5.9 `recvProxyConnect` — 接受控制连接

在数据连接被接受、监听套接字关闭之后：

```cpp
  resources->hasControlSock = false;
  {
    ncclResult_t fcRet = ncclSocketAccept(&resources->controlSock, &resources->controlListenSock);
    if (fcRet == ncclSuccess) {
      int fd = -1;
      ncclSocketGetFd(&resources->controlSock, &fd);
      if (fd >= 0) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
      }
      resources->hasControlSock = true;
    }
    ncclSocketClose(&resources->controlListenSock);
  }
```

#### 5.10 `sendProxyFree` / `recvProxyFree` — 套接字清理

两个函数都在释放资源前关闭控制套接字：

```cpp
  // 在 sendProxyFree 中：
  if (resources && resources->fcState.hasControlSock) {
    ncclSocketClose(&resources->fcState.controlSock);
  }

  // 在 recvProxyFree 中：
  if (resources && resources->hasControlSock) {
    ncclSocketClose(&resources->controlSock);
  }
```

#### 5.11 令牌桶辅助函数（位于 `sendProxyProgress` 之前）

**`fcPollRateUpdates()`** — 非阻塞轮询速率消息：

```cpp
static void fcPollRateUpdates(struct ncclFcSenderState* state) {
  if (!state->hasControlSock) return;
  int fd = -1;
  ncclSocketGetFd(&state->controlSock, &fd);
  if (fd < 0) return;

  struct ncclFcRateMsg msg;
  while (true) {
    ssize_t ret = recv(fd, &msg, sizeof(msg), MSG_DONTWAIT);
    if (ret != sizeof(msg)) break;
    if (msg.magic != NCCL_FC_RATE_MSG_MAGIC) continue;

    // 查找或创建流条目，更新 bucket.rate
    // ...
  }
}
```

**`fcTokenBucketAllow()`** — 检查是否允许发送：

```cpp
static bool fcTokenBucketAllow(struct ncclFcSenderState* state, int channelId,
                               uint64_t opCount, int bytes, int stepSize) {
  // 查找该流的令牌桶
  struct ncclFcTokenBucket* b = ...;

  if (b->rate <= 0) return false;  // 已暂停

  // 首次使用时初始化：burstSize = 4 * stepSize，tokens = burstSize
  if (!b->initialized) { ... }

  // 补充令牌：tokens += rate * 经过时间
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  double elapsed = ...;
  b->tokens += b->rate * elapsed;
  if (b->tokens > b->burstSize) b->tokens = b->burstSize;
  b->lastRefill = now;

  // 检查并消费
  if (b->tokens >= (double)bytes) {
    b->tokens -= (double)bytes;
    return true;
  }
  return false;  // 被限速
}
```

**`fcSenderRemoveFlow()`** — 完成时清理：

```cpp
static void fcSenderRemoveFlow(struct ncclFcSenderState* state, int channelId, uint64_t opCount) {
  // 查找并标记流为非活跃
}
```

#### 5.12 `sendProxyProgress` — 集成

```diff
 static ncclResult_t sendProxyProgress(...) {
+  bool useSendFc = (args->pattern == ncclPatternSend || args->pattern == ncclPatternRecv);

   // ...（ncclProxyOpReady 初始化不变）...

   if (args->state == ncclProxyOpProgress) {
+    // 轮询接收方的速率更新（非阻塞，仅 P2P）
+    if (useSendFc && args->nsubs > 0) {
+      struct sendNetResources* r0 = ...;
+      fcPollRateUpdates(&r0->fcState);
+    }

     // ... 对每个 sub ...
           if (ready) {
+            // isend 前检查令牌桶（仅 P2P）
+            if (useSendFc) {
+              if (!fcTokenBucketAllow(&resources->fcState, sub->channelId,
+                                     args->opCount, size, stepSize)) {
+                continue;  // 被限速 — 跳过本次迭代
+              }
+            }
             // ... isend() ...
           }

     // ... 完成时 ...
           if (sub->done == sub->nsteps) {
             args->done++;
+            if (useSendFc) {
+              fcSenderRemoveFlow(&resources->fcState, sub->channelId, args->opCount);
+            }
           }
   }
 }
```

#### 5.13 接收方回调基础设施（位于 `recvProxyProgress` 之前）

**`ncclFcControlContext`** — senderRank 到控制套接字的映射：

```cpp
#define NCCL_FC_MAX_SENDERS 512

struct ncclFcControlContext {
  struct {
    struct ncclSocket* sock;  // 指向 recvNetResources::controlSock 的指针
    int senderRank;
    bool valid;
  } senders[NCCL_FC_MAX_SENDERS];
  int numSenders;
};
```

**`fcRateChangeCallback()`** — 向发送方分发速率更新：

```cpp
static void fcRateChangeCallback(void* context, const struct ncclFcRateMsg* msg) {
  struct ncclFcControlContext* ctx = (struct ncclFcControlContext*)context;
  for (int i = 0; i < ctx->numSenders; i++) {
    if (ctx->senders[i].valid && ctx->senders[i].senderRank == msg->senderRank) {
      int fd = -1;
      ncclSocketGetFd(ctx->senders[i].sock, &fd);
      if (fd >= 0) {
        // 非阻塞发送；丢弃消息是可以接受的
        send(fd, msg, sizeof(*msg), MSG_NOSIGNAL | MSG_DONTWAIT);
      }
      return;
    }
  }
}
```

#### 5.14 `recvProxyProgress` — 集成

```diff
 static ncclResult_t recvProxyProgress(...) {
+  bool useFlowControl = proxyState->recvFlowControl &&
+      (args->pattern == ncclPatternSend || args->pattern == ncclPatternRecv);

   if (args->state == ncclProxyOpReady) {
     // ... sub 初始化 ...
+      if (useFlowControl) {
+        // 延迟初始化回调上下文
+        if (proxyState->fcControlContext == nullptr) {
+          proxyState->fcControlContext = calloc(1, sizeof(struct ncclFcControlContext));
+          ncclFlowControlSetCallback(proxyState->recvFlowControl,
+                                     fcRateChangeCallback, proxyState->fcControlContext);
+        }
+        // 注册发送方的控制套接字
+        if (resources->hasControlSock) {
+          fcControlContextAddSender(..., resources->tpRemoteRank, &resources->controlSock);
+        }
+        // 注册流 — 触发 recalculateAndNotify → 回调 → TCP 发送
+        double rate;
+        ncclFlowControlAddFlow(proxyState->recvFlowControl,
+                               resources->tpRemoteRank, sub->channelId,
+                               args->opCount, sub->nbytes, &rate);
+      }
   }

   // irecv 投递：接收方以完整流水线深度投递（无深度限制）
-  // （旧的 ComputeAllowedSteps 代码已移除）
+  // 速率限制在发送方通过令牌桶执行
   if (sub->posted >= sub->done + maxDepth) { subCount = 0; break; }

   // 收到数据时：
+  if (useFlowControl && sizes[i] > 0) {
+    ncclFlowControlReportProgress(proxyState->recvFlowControl,
+                                  resources->tpRemoteRank, sub->channelId,
+                                  args->opCount, sizes[i]);
+  }

   // 流完成时：
+  if (useFlowControl) {
+    ncclFlowControlRemoveFlow(proxyState->recvFlowControl,
+                              fcRes->tpRemoteRank, sub->channelId, args->opCount);
+  }
 }
```

---

### 6. `src/Makefile`（修改文件）

```diff
 LIBSRCFILES := \
 	bootstrap.cc channel.cc collectives.cc debug.cc enqueue.cc group.cc \
 	init.cc proxy.cc transport.cc mnnvl.cc allocator.cc dev_runtime.cc sym_kernels.cc ce_coll.cc mem_manager.cc \
+	flowcontrol.cc \
```

---

## 设计决策

### 为什么使用 TCP 控制套接字（而非 RDMA / 共享内存）？

- NCCL 的网络传输使用 RDMA 传输数据；`sendMem`/`recvMem` 是本地 GPU 内存，不在机器间共享
- 运行时不存在现有的 proxy 间通信通道
- TCP 简单、可靠，控制消息很小（40 字节）且不频繁（仅在流增减时发送）
- 非阻塞 `send(MSG_DONTWAIT)` 确保回调永远不会阻塞流量控制互斥锁

### 为什么使用令牌桶（而非漏桶 / 窗口）？

- 令牌桶允许空闲后突发（重要：NCCL 的流水线在各步骤之间天然存在空闲间隙）
- `burstSize = 4 * stepSize` 允许流水线在速率增加后快速填充
- 使用 `clock_gettime(CLOCK_MONOTONIC)` 的简单实现 — 不需要定时器线程

### 为什么仅限 P2P？

- AllReduce（Ring/Tree）：所有节点必须同步推进；限制任何节点都会卡住整条 ring
- AlltoAll（P2P）：流相互独立；限制一个发送方不影响其他发送方
- 模式检查：`args->pattern == ncclPatternSend || args->pattern == ncclPatternRecv`

### 为什么 `maxConcurrent` 默认为 0（无限制）？

- 在基于速率的模型中，所有活跃流获得 `rate = 总带宽 / N`
- 无需人为限制并发 — 公平份额速率自然地划分带宽
- DRR 轮转在流数量非常多时处理公平性（可通过 `NCCL_RECV_FC_QUOTA_MB` 配置）

### 控制消息丢弃

- `send(MSG_DONTWAIT)` 在 TCP 发送缓冲区满时可能失败
- 这是可以接受的：发送方继续以上一次授权的速率运行
- 下一次流变化会触发新的速率更新，快速收敛
- 生产环境建议：考虑增大控制套接字的 `SO_SNDBUF`

---

## 数据流示例：4 rank AlltoAll

Rank 0 从 rank 1、2、3 接收数据：

```
时间  事件                                        每流速率
─────────────────────────────────────────────────────────────
t0    Rank 1 开始发送 → AddFlow                   25/1 = 25 GB/s
      → 发送速率 25 GB/s 给 rank 1

t1    Rank 2 开始发送 → AddFlow                   25/2 = 12.5 GB/s
      → 发送速率 12.5 GB/s 给 rank 1
      → 发送速率 12.5 GB/s 给 rank 2

t2    Rank 3 开始发送 → AddFlow                   25/3 ≈ 8.33 GB/s
      → 发送速率 8.33 GB/s 给 rank 1
      → 发送速率 8.33 GB/s 给 rank 2
      → 发送速率 8.33 GB/s 给 rank 3

t3    Rank 1 完成 → RemoveFlow                    25/2 = 12.5 GB/s
      → 发送速率 0 给 rank 1（已终止）
      → 发送速率 12.5 GB/s 给 rank 2
      → 发送速率 12.5 GB/s 给 rank 3

t4    Rank 2 完成 → RemoveFlow                    25/1 = 25 GB/s
      → 发送速率 0 给 rank 2（已终止）
      → 发送速率 25 GB/s 给 rank 3

t5    Rank 3 完成 → RemoveFlow                    （无活跃流）
```

---

## 构建验证

全部三个修改的 .cc 文件编译成功：

```
编译  proxy.cc         > build/obj/proxy.o          ✓
编译  flowcontrol.cc   > build/obj/flowcontrol.o    ✓
编译  transport/net.cc > build/obj/transport/net.o   ✓
```
