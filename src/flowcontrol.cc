/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "flowcontrol.h"
#include "debug.h"
#include "param.h"
#include <cstring>
#include <climits>
#include <time.h>

// Environment variables
NCCL_PARAM(RecvFlowControlBandwidthGBs, "RECV_FC_BW_GBS", 25);
NCCL_PARAM(RecvFlowControlMaxConcurrent, "RECV_FC_MAX_CONCURRENT", 0);
NCCL_PARAM(RecvFlowControlQuotaMB, "RECV_FC_QUOTA_MB", 4);
// Guard-style admission threshold: flows with totalBytes ≤ this skip fair-share
// (sender gets full rate, no other flow throttled by it). 0 = disabled.
NCCL_PARAM(RecvFlowControlBdpBytes, "RECV_FC_BDP_BYTES", 0);
// Guard-style EWMA proactive release: when v_remain < est_rate * baseRtt, the
// flow exits the fair-share denominator early so remaining flows speed up.
NCCL_PARAM(RecvFlowControlProactiveRelease, "RECV_FC_PROACTIVE_RELEASE", 0);
NCCL_PARAM(RecvFlowControlBaseRttUs, "RECV_FC_BASE_RTT_US", 22);

// EWMA constants matching ns-3 guard (rdma-hw.cc:485,498)
static constexpr double kEwmaBeta = 0.125;     // m_est_rate = β·old + (1-β)·inst
static constexpr double kReleaseGamma = 1.0;   // v_th = est * baseRtt * γ

static inline uint64_t monotonicNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline int flowIdMatch(const struct ncclFlowId* a, int senderRank, int channelId, uint64_t opCount) {
  return a->senderRank == senderRank && a->channelId == channelId && a->opCount == opCount;
}

// Build a rate message for a flow entry
static inline struct ncclFcRateMsg makeRateMsg(const struct ncclFlowEntry* f) {
  struct ncclFcRateMsg msg;
  msg.magic = NCCL_FC_RATE_MSG_MAGIC;
  msg.senderRank = f->id.senderRank;
  msg.channelId = f->id.channelId;
  msg.opCount = f->id.opCount;
  msg.authorizedRate = f->authorizedRate;
  return msg;
}

// Invoke the rate-change callback for a single flow (must hold mutex)
static void notifyRateChange(struct ncclRecvFlowControl* fc, const struct ncclFlowEntry* f) {
  if (fc->rateChangeCallback && f->state != NCCL_FLOW_FREE) {
    struct ncclFcRateMsg msg = makeRateMsg(f);
    fc->rateChangeCallback(fc->rateChangeContext, &msg);
  }
}

// Recalculate rates for all active flows and notify senders (must hold mutex)
static void recalculateAndNotify(struct ncclRecvFlowControl* fc) {
  if (fc->numActive == 0) return;
  double fairRate = fc->totalRecvBandwidth / (double)fc->numActive;
  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state == NCCL_FLOW_ACTIVE) {
      fc->flows[i].authorizedRate = fairRate;
      notifyRateChange(fc, &fc->flows[i]);
    }
  }
  INFO(NCCL_NET, "FlowControl rank %d: %d active (rate=%.2f GB/s each), %d pending",
       fc->tpRank, fc->numActive, fairRate / (1024.0 * 1024.0 * 1024.0), fc->numPending);
}

// Promote the oldest pending flow to active (must hold mutex)
static void promotePending(struct ncclRecvFlowControl* fc) {
  if (fc->numPending == 0) return;
  if (fc->maxConcurrent > 0 && fc->numActive >= fc->maxConcurrent) return;

  int best = -1;
  uint64_t bestOrder = UINT64_MAX;
  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state == NCCL_FLOW_PENDING && fc->flows[i].arrivalOrder < bestOrder) {
      best = i;
      bestOrder = fc->flows[i].arrivalOrder;
    }
  }
  if (best >= 0) {
    fc->flows[best].state = NCCL_FLOW_ACTIVE;
    fc->flows[best].remainingQuota = fc->quotaPerRound;
    fc->numPending--;
    fc->numActive++;
    INFO(NCCL_NET, "FlowControl rank %d: promoted flow (sender %d ch %d op %lu) → active, %d active, %d pending",
         fc->tpRank, fc->flows[best].id.senderRank, fc->flows[best].id.channelId,
         fc->flows[best].id.opCount, fc->numActive, fc->numPending);
    recalculateAndNotify(fc);
  }
}

// Remove a flow by index (must hold mutex), then promote + recalculate
static void removeFlowByIndex(struct ncclRecvFlowControl* fc, int idx) {
  enum ncclFlowState oldState = fc->flows[idx].state;

  // Notify sender that this flow's rate is now 0 (terminated)
  fc->flows[idx].authorizedRate = 0;
  notifyRateChange(fc, &fc->flows[idx]);

  fc->flows[idx].state = NCCL_FLOW_FREE;
  if (oldState == NCCL_FLOW_ACTIVE) {
    fc->numActive--;
  } else if (oldState == NCCL_FLOW_PENDING) {
    fc->numPending--;
  }
  // OFFLOAD flows are tracked but not counted in numActive/numPending,
  // so no decrement needed.

  INFO(NCCL_NET, "FlowControl rank %d: removed flow (sender %d ch %d op %lu, was %s), %d active, %d pending",
       fc->tpRank, fc->flows[idx].id.senderRank, fc->flows[idx].id.channelId,
       fc->flows[idx].id.opCount,
       oldState == NCCL_FLOW_ACTIVE ? "active" :
       oldState == NCCL_FLOW_PENDING ? "pending" : "offload",
       fc->numActive, fc->numPending);

  if (oldState == NCCL_FLOW_ACTIVE) {
    promotePending(fc);
    // If nothing was promoted, still recalculate (fewer flows = more bw each)
    if (fc->numPending == 0 && fc->numActive > 0) recalculateAndNotify(fc);
  }
}

// Guard-style proactive release: transition ACTIVE→OFFLOAD without
// touching the sender's authorized rate. The flow continues at its
// last-known rate (probably enough to drain the small remaining bytes),
// while the remaining ACTIVE flows are recalculated to a higher fair share.
// Must hold mutex.
static void proactiveReleaseFlow(struct ncclRecvFlowControl* fc, int idx) {
  fc->flows[idx].state = NCCL_FLOW_OFFLOAD;
  fc->flows[idx].proactiveReleased = 1;
  fc->numActive--;
  INFO(NCCL_NET, "FlowControl rank %d: PROACTIVE-RELEASE flow (sender %d ch %d op %lu), "
       "remain=%ld est=%.2fGB/s, %d active remaining",
       fc->tpRank, fc->flows[idx].id.senderRank, fc->flows[idx].id.channelId,
       fc->flows[idx].id.opCount,
       (long)(fc->flows[idx].totalBytes - fc->flows[idx].completedBytes),
       fc->flows[idx].estRate / (1024.0 * 1024.0 * 1024.0),
       fc->numActive);
  // Promote pending or recalculate so remaining actives speed up
  promotePending(fc);
  if (fc->numPending == 0 && fc->numActive > 0) recalculateAndNotify(fc);
}

// Demote an active flow back to pending for DRR rotation (must hold mutex)
static void demoteActiveFlow(struct ncclRecvFlowControl* fc, int idx) {
  fc->flows[idx].state = NCCL_FLOW_PENDING;
  fc->flows[idx].authorizedRate = 0;
  fc->flows[idx].arrivalOrder = fc->nextArrivalOrder++;
  fc->numActive--;
  fc->numPending++;

  // Notify sender: rate = 0 (paused during rotation)
  notifyRateChange(fc, &fc->flows[idx]);

  INFO(NCCL_NET, "FlowControl rank %d: DEMOTED flow (sender %d ch %d op %lu) for DRR rotation, "
       "%d active, %d pending",
       fc->tpRank, fc->flows[idx].id.senderRank, fc->flows[idx].id.channelId,
       fc->flows[idx].id.opCount, fc->numActive, fc->numPending);

  promotePending(fc);
}

ncclResult_t ncclFlowControlInit(struct ncclRecvFlowControl* fc, int tpRank, double totalRecvBandwidth) {
  memset(fc->flows, 0, sizeof(fc->flows));
  fc->numActive = 0;
  fc->numPending = 0;
  fc->tpRank = tpRank;
  fc->nextArrivalOrder = 0;
  fc->rateChangeCallback = nullptr;
  fc->rateChangeContext = nullptr;

  int64_t bwGBs = ncclParamRecvFlowControlBandwidthGBs();
  if (bwGBs > 0) {
    fc->totalRecvBandwidth = (double)bwGBs * 1024.0 * 1024.0 * 1024.0;
  } else {
    fc->totalRecvBandwidth = totalRecvBandwidth;
  }

  int64_t userMaxConcurrent = ncclParamRecvFlowControlMaxConcurrent();
  fc->maxConcurrent = (userMaxConcurrent > 0) ? (int)userMaxConcurrent : 0;  // 0 = unlimited

  int64_t quotaMB = ncclParamRecvFlowControlQuotaMB();
  fc->quotaPerRound = (quotaMB > 0) ? (ssize_t)quotaMB * 1024 * 1024 : 0;

  fc->bdpBytes = (ssize_t)ncclParamRecvFlowControlBdpBytes();
  fc->proactiveRelease = (int)ncclParamRecvFlowControlProactiveRelease();
  int64_t rttUs = ncclParamRecvFlowControlBaseRttUs();
  fc->baseRttSec = (rttUs > 0 ? (double)rttUs : 22.0) * 1e-6;

  INFO(NCCL_NET, "FlowControl rank %d: initialized, bw=%.2f GB/s, maxConcurrent=%d%s, quota=%ldMB%s, bdp=%ldB%s, proactive=%d (baseRtt=%.1fus)",
       tpRank, fc->totalRecvBandwidth / (1024.0 * 1024.0 * 1024.0),
       fc->maxConcurrent, fc->maxConcurrent == 0 ? " (unlimited)" : "",
       (long)(fc->quotaPerRound / (1024 * 1024)),
       fc->quotaPerRound > 0 ? " (DRR)" : "",
       (long)fc->bdpBytes, fc->bdpBytes > 0 ? " (bypass)" : " (no bypass)",
       fc->proactiveRelease, fc->baseRttSec * 1e6);
  return ncclSuccess;
}

void ncclFlowControlSetCallback(struct ncclRecvFlowControl* fc,
                                ncclFcRateChangeCallback_t cb, void* context) {
  fc->rateChangeCallback = cb;
  fc->rateChangeContext = context;
}

ncclResult_t ncclFlowControlAddFlow(struct ncclRecvFlowControl* fc,
                                    int senderRank, int channelId, uint64_t opCount,
                                    ssize_t totalBytes, double* authorizedRate) {
  std::lock_guard<std::mutex> lock(fc->mutex);

  // Check for duplicate
  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state != NCCL_FLOW_FREE &&
        flowIdMatch(&fc->flows[i].id, senderRank, channelId, opCount)) {
      *authorizedRate = fc->flows[i].authorizedRate;
      return ncclSuccess;
    }
  }

  // Find a free slot
  int slot = -1;
  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state == NCCL_FLOW_FREE) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    WARN("FlowControl rank %d: no free flow slots (max %d)", fc->tpRank, NCCL_FC_MAX_FLOWS);
    return ncclInternalError;
  }

  fc->flows[slot].id.senderRank = senderRank;
  fc->flows[slot].id.channelId = channelId;
  fc->flows[slot].id.opCount = opCount;
  fc->flows[slot].totalBytes = totalBytes;
  fc->flows[slot].completedBytes = 0;
  fc->flows[slot].arrivalOrder = fc->nextArrivalOrder++;
  fc->flows[slot].estRate = 0;
  fc->flows[slot].lastProgressNs = 0;
  fc->flows[slot].proactiveReleased = 0;

  // Guard-style bypass: small flows skip the fair-share denominator entirely.
  // Sender is told totalRecvBandwidth (full rate); no other flow is throttled
  // by this one. Mirrors ns-3 guard's `if (flow_size > bdp)` admission gate.
  if (fc->bdpBytes > 0 && totalBytes > 0 && totalBytes <= fc->bdpBytes) {
    fc->flows[slot].state = NCCL_FLOW_OFFLOAD;
    fc->flows[slot].authorizedRate = fc->totalRecvBandwidth;
    notifyRateChange(fc, &fc->flows[slot]);
    *authorizedRate = fc->flows[slot].authorizedRate;
    INFO(NCCL_NET, "FlowControl rank %d: BYPASS flow (sender %d ch %d op %lu), bytes=%ld ≤ bdp=%ld, full rate=%.2f GB/s",
         fc->tpRank, senderRank, channelId, opCount, totalBytes, (long)fc->bdpBytes,
         *authorizedRate / (1024.0 * 1024.0 * 1024.0));
    return ncclSuccess;
  }

  // Admission: if maxConcurrent==0 (unlimited) or under limit, admit directly
  if (fc->maxConcurrent == 0 || fc->numActive < fc->maxConcurrent) {
    fc->flows[slot].state = NCCL_FLOW_ACTIVE;
    fc->flows[slot].remainingQuota = fc->quotaPerRound;
    fc->numActive++;
    recalculateAndNotify(fc);
    *authorizedRate = fc->flows[slot].authorizedRate;
    INFO(NCCL_NET, "FlowControl rank %d: ACTIVE flow (sender %d ch %d op %lu), bytes=%ld, rate=%.2f GB/s, %d active, %d pending",
         fc->tpRank, senderRank, channelId, opCount, totalBytes,
         *authorizedRate / (1024.0 * 1024.0 * 1024.0),
         fc->numActive, fc->numPending);
  } else {
    fc->flows[slot].state = NCCL_FLOW_PENDING;
    fc->flows[slot].authorizedRate = 0;
    fc->numPending++;
    *authorizedRate = 0;
    // Notify sender: rate = 0 (queued)
    notifyRateChange(fc, &fc->flows[slot]);
    INFO(NCCL_NET, "FlowControl rank %d: PENDING flow (sender %d ch %d op %lu), bytes=%ld, %d active, %d pending",
         fc->tpRank, senderRank, channelId, opCount, totalBytes,
         fc->numActive, fc->numPending);
  }
  return ncclSuccess;
}

ncclResult_t ncclFlowControlRemoveFlow(struct ncclRecvFlowControl* fc,
                                       int senderRank, int channelId, uint64_t opCount) {
  std::lock_guard<std::mutex> lock(fc->mutex);

  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state != NCCL_FLOW_FREE &&
        flowIdMatch(&fc->flows[i].id, senderRank, channelId, opCount)) {
      removeFlowByIndex(fc, i);
      return ncclSuccess;
    }
  }
  TRACE(NCCL_NET, "FlowControl rank %d: flow (sender %d ch %d op %lu) not found for removal",
        fc->tpRank, senderRank, channelId, opCount);
  return ncclSuccess;
}

ncclResult_t ncclFlowControlReportProgress(struct ncclRecvFlowControl* fc,
                                           int senderRank, int channelId, uint64_t opCount,
                                           ssize_t additionalBytes) {
  std::lock_guard<std::mutex> lock(fc->mutex);

  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state != NCCL_FLOW_FREE &&
        flowIdMatch(&fc->flows[i].id, senderRank, channelId, opCount)) {
      fc->flows[i].completedBytes += additionalBytes;

      // Check completion
      if (fc->flows[i].completedBytes >= fc->flows[i].totalBytes) {
        INFO(NCCL_NET, "FlowControl rank %d: flow (sender %d ch %d op %lu) completed (%ld/%ld bytes)",
             fc->tpRank, senderRank, channelId, opCount,
             fc->flows[i].completedBytes, fc->flows[i].totalBytes);
        removeFlowByIndex(fc, i);
        return ncclSuccess;
      }

      // EWMA proactive release (guard-style). Only meaningful for ACTIVE flows;
      // OFFLOAD flows are already out of fair-share, PENDING ones haven't started.
      if (fc->flows[i].state == NCCL_FLOW_ACTIVE && fc->proactiveRelease) {
        uint64_t nowNs = monotonicNs();
        if (fc->flows[i].lastProgressNs != 0) {
          double interval = (double)(nowNs - fc->flows[i].lastProgressNs) * 1e-9;
          if (interval > 0) {
            double instRate = (double)additionalBytes / interval;  // bytes/sec
            fc->flows[i].estRate =
                kEwmaBeta * fc->flows[i].estRate + (1.0 - kEwmaBeta) * instRate;
          }
        }
        fc->flows[i].lastProgressNs = nowNs;

        if (!fc->flows[i].proactiveReleased) {
          ssize_t vRemain = fc->flows[i].totalBytes - fc->flows[i].completedBytes;
          double vTh = fc->flows[i].estRate * fc->baseRttSec * kReleaseGamma;
          if (vRemain > 0 && (double)vRemain < vTh) {
            proactiveReleaseFlow(fc, i);
            return ncclSuccess;  // state changed, skip DRR
          }
        }
      }

      // DRR rotation for active flows
      if (fc->flows[i].state == NCCL_FLOW_ACTIVE && fc->quotaPerRound > 0) {
        fc->flows[i].remainingQuota -= additionalBytes;
        if (fc->flows[i].remainingQuota <= 0 && fc->numPending > 0) {
          demoteActiveFlow(fc, i);
        }
      }
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclFlowControlGetRate(struct ncclRecvFlowControl* fc,
                                    int senderRank, int channelId, uint64_t opCount,
                                    double* authorizedRate) {
  std::lock_guard<std::mutex> lock(fc->mutex);

  for (int i = 0; i < NCCL_FC_MAX_FLOWS; i++) {
    if (fc->flows[i].state != NCCL_FLOW_FREE &&
        flowIdMatch(&fc->flows[i].id, senderRank, channelId, opCount)) {
      *authorizedRate = fc->flows[i].authorizedRate;
      return ncclSuccess;
    }
  }
  *authorizedRate = fc->totalRecvBandwidth;
  return ncclSuccess;
}

ncclResult_t ncclFlowControlDestroy(struct ncclRecvFlowControl* fc) {
  std::lock_guard<std::mutex> lock(fc->mutex);
  memset(fc->flows, 0, sizeof(fc->flows));
  fc->numActive = 0;
  fc->numPending = 0;
  fc->rateChangeCallback = nullptr;
  fc->rateChangeContext = nullptr;
  return ncclSuccess;
}
