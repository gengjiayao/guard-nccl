/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_FLOWCONTROL_H_
#define NCCL_FLOWCONTROL_H_

#include "nccl.h"
#include <stdint.h>
#include <sys/types.h>
#include <mutex>

/*
 * Receiver-driven flow control for NCCL network transport.
 *
 * The receiver maintains a list of active inbound flows. Each flow is
 * assigned a fair-share send rate:
 *
 *   authorizedRate = totalRecvBandwidth / numActiveFlows
 *
 * When a flow is added, removed, or completes, rates are recalculated
 * and pushed to the corresponding senders via a rate-change callback.
 * Senders enforce the authorized rate using a token bucket.
 *
 * Scope: Only applied to P2P operations (AlltoAll, Send/Recv). Collective
 *   operations (AllReduce ring/tree, etc.) are exempt because they require
 *   globally synchronized progress.
 *
 * Environment variables:
 *   NCCL_RECV_FC_BW_GBS              Total receive bandwidth in GB/s (default: 25)
 *   NCCL_RECV_FC_MAX_CONCURRENT      Max concurrent admitted flows (default: 0 = unlimited)
 *   NCCL_RECV_FC_QUOTA_MB            DRR byte quota per round in MB (default: 4, 0=disable)
 *   NCCL_RECV_FC_BDP_BYTES           Bypass threshold: flows ≤ this go full rate (default: 0 = no bypass)
 *   NCCL_RECV_FC_PROACTIVE_RELEASE   Enable EWMA-based early release (default: 0)
 *   NCCL_RECV_FC_BASE_RTT_US         Assumed base RTT for proactive release threshold (default: 22)
 */

// Maximum number of registered flows per receiver
#define NCCL_FC_MAX_FLOWS 512

// Rate update message sent from receiver proxy to sender proxy via control socket
struct ncclFcRateMsg {
  uint32_t magic;           // NCCL_FC_RATE_MSG_MAGIC
  int senderRank;           // Flow identifier: sender rank
  int channelId;            // Flow identifier: channel
  uint64_t opCount;         // Flow identifier: op count
  double authorizedRate;    // Bytes/sec authorized by receiver (0 = paused)
};

#define NCCL_FC_RATE_MSG_MAGIC 0x4E464352  // "NFCR"

// Callback invoked when a flow's authorized rate changes.
// The receiver proxy uses this to send rate updates to senders.
// Called with the flow control mutex held — implementation must be non-blocking.
typedef void (*ncclFcRateChangeCallback_t)(void* context, const struct ncclFcRateMsg* msg);

// Flow states
enum ncclFlowState {
  NCCL_FLOW_FREE     = 0,
  NCCL_FLOW_ACTIVE   = 1,  // Active flow with fair-share rate
  NCCL_FLOW_PENDING  = 2,  // Waiting for admission (rate = 0)
  NCCL_FLOW_OFFLOAD  = 3,  // Tracked but excluded from fair-share denominator.
                           //   Used for (a) small flows below BDP bypass threshold,
                           //   (b) flows past EWMA-based proactive release point.
                           //   Sender keeps its last-authorized rate (no 0 notify).
};

// Flow identifier
struct ncclFlowId {
  int senderRank;
  int channelId;
  uint64_t opCount;
};

// A single flow entry
struct ncclFlowEntry {
  struct ncclFlowId id;
  double authorizedRate;    // Bytes per second authorized by receiver
  ssize_t totalBytes;       // Total bytes for this flow
  ssize_t completedBytes;   // Bytes completed so far
  ssize_t remainingQuota;   // DRR: bytes remaining before rotation
  enum ncclFlowState state;
  uint64_t arrivalOrder;    // Monotonic counter for FIFO promotion

  // EWMA proactive-release tracking (guard-style)
  double   estRate;         // EWMA of received bytes/sec
  uint64_t lastProgressNs;  // CLOCK_MONOTONIC ns at last ReportProgress (0 = uninitialized)
  uint8_t  proactiveReleased; // Once-only flag: already transitioned ACTIVE→OFFLOAD
};

// Per-receiver flow control state
struct ncclRecvFlowControl {
  struct ncclFlowEntry flows[NCCL_FC_MAX_FLOWS];
  int numActive;             // Flows in ACTIVE state
  int numPending;            // Flows in PENDING state
  int maxConcurrent;         // Max flows admitted simultaneously (0 = unlimited)
  ssize_t quotaPerRound;     // DRR byte quota (0 = disabled)
  double totalRecvBandwidth; // Total receive bandwidth in bytes/sec
  int tpRank;
  uint64_t nextArrivalOrder;

  // Guard-style admission threshold + proactive release config
  ssize_t bdpBytes;          // Flows with totalBytes ≤ bdpBytes bypass fair-share (0 = disable)
  int     proactiveRelease;  // 1 = enable EWMA-based early release, 0 = disable
  double  baseRttSec;        // Base RTT used in v_remain < est * baseRtt threshold

  // Rate change callback — called whenever a flow's rate changes
  ncclFcRateChangeCallback_t rateChangeCallback;
  void* rateChangeContext;

  std::mutex mutex;
};

// Initialize flow control.
ncclResult_t ncclFlowControlInit(struct ncclRecvFlowControl* fc, int tpRank,
                                 double totalRecvBandwidth);

// Set the callback invoked on rate changes.
void ncclFlowControlSetCallback(struct ncclRecvFlowControl* fc,
                                ncclFcRateChangeCallback_t cb, void* context);

// Register a new inbound flow. Returns authorized rate via callback + out param.
ncclResult_t ncclFlowControlAddFlow(struct ncclRecvFlowControl* fc,
                                    int senderRank, int channelId, uint64_t opCount,
                                    ssize_t totalBytes, double* authorizedRate);

// Remove a completed flow and recalculate rates.
ncclResult_t ncclFlowControlRemoveFlow(struct ncclRecvFlowControl* fc,
                                       int senderRank, int channelId, uint64_t opCount);

// Report received bytes. Auto-removes on completion, triggers DRR rotation.
ncclResult_t ncclFlowControlReportProgress(struct ncclRecvFlowControl* fc,
                                           int senderRank, int channelId, uint64_t opCount,
                                           ssize_t additionalBytes);

// Query the authorized rate for a flow.
ncclResult_t ncclFlowControlGetRate(struct ncclRecvFlowControl* fc,
                                    int senderRank, int channelId, uint64_t opCount,
                                    double* authorizedRate);

// Destroy flow control instance.
ncclResult_t ncclFlowControlDestroy(struct ncclRecvFlowControl* fc);

#endif // NCCL_FLOWCONTROL_H_
