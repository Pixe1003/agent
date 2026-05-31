# Kubernetes 化技术方案 — Agent Infra Toolkit

> 目标：把现有 5 模块同进程 Python SDK 演进成 K8s 原生的分布式 Agent Runtime。
> 适用岗位：蚂蚁 / 字节 / 阿里 / 腾讯 Agent Infra 岗（JD 明确点名 Kubernetes / 容器编排 / 工作流编排 / 服务治理 / 任务调度 / 沙箱隔离）。
>
> **本文只是技术方案，不是实施清单**。决策点编号 D1-D14，每个有"必须 / 推荐 / 可选"标签，方便分阶段裁剪。

---

## 0. 设计原则（先定边界，再定细节）

1. **K8s 是 wrapper，不是替代**：核心 agent 逻辑零改动，只在外面套容器、Service、Helm。
2. **同进程 fast-path 保留**：unit tests / benchmark / demo 仍走 in-process import，**不能引入 100 ms 级 RPC overhead**。
3. **service 化是可选层**：通过环境变量切换 `MODE=local | grpc | mcp`，让单文件部署和集群部署共用同一份代码。
4. **State always externalizable**：所有 in-memory state（hybrid_stats / aiops alerts / working memory）都要支持"快照 → 外部存储 → 重启重建"。
5. **每个新组件必须有 metric**：Prometheus exposition 是默认要求，不是 nice-to-have。

---

## 1. 架构总览

```
┌──────────────────────────────── K8s Cluster ────────────────────────────────┐
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │ multi_agent │←gRPC│agent_memory │    │ agent_aiops │    │  agent_sft  │  │
│   │ Deployment  │    │StatefulSet  │    │ Deployment  │    │ Deployment  │  │
│   │  ×3 replica │    │ +PVC        │    │  ×1        │    │ +GPU node   │  │
│   │  +HPA       │    │  ×1         │    │             │    │  ×1        │  │
│   └──────┬──────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│          │                                                                   │
│   ┌──────▼──────────────────────────────────────────────────────────────┐   │
│   │            mcp_server  (Deployment ×2 + LoadBalancer Service)        │   │
│   │            对外 MCP / gRPC / REST 三协议 gateway                     │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌───── batch & cron ─────┐  ┌─── observability ────┐  ┌── data ──┐        │
│   │ benchmark-runner Job   │  │ Prometheus + Grafana │  │ PVC      │        │
│   │ trainer-lora Job (GPU) │  │ Jaeger / Tempo       │  │ MinIO/S3 │        │
│   │ drift-detector CronJob │  │ Fluent Bit → Loki    │  │ ConfigMap│        │
│   │ episode-compactor Cron │  │ ServiceMonitor       │  │ Secret   │        │
│   └────────────────────────┘  └──────────────────────┘  └──────────┘        │
│                                                                              │
│   ┌─── service mesh ────────┐  ┌─── workflow ─────────┐                     │
│   │ Istio sidecar (mTLS,    │  │ Argo Workflows /     │                     │
│   │  retry, circuit breaker,│  │  Tekton DAG:         │                     │
│   │  canary)                │  │  build → train →     │                     │
│   └─────────────────────────┘  │  eval → promote      │                     │
│                                └──────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 决策点 D1-D14

### D1. Pod 拆分粒度（**必须**）

| 模块 | 工作负载类型 | 副本 | 资源 request/limit | 备注 |
|---|---|---|---|---|
| multi_agent | Deployment + HPA | 3 (1-10) | 100m / 500m CPU, 256Mi / 1Gi | autoscale on QPS metric |
| agent_memory | StatefulSet | 1 | 200m / 1 CPU, 512Mi / 2Gi | PVC for episodic JSONL |
| agent_aiops | Deployment | 1 | 100m / 300m CPU, 256Mi / 512Mi | rolling window 纯内存 |
| agent_sft | Deployment | 1 (1-3) | 4 CPU, 8 Gi, **1 GPU** | nodeSelector `accelerator=nvidia` |
| mcp_server | Deployment + Service LB | 2 (2-10) | 100m / 500m CPU, 256Mi / 512Mi | 对外入口 |
| dashboard | Deployment + Ingress | 1 | 50m CPU, 128Mi | nginx 服务静态资源 |
| benchmark-runner | Job (manual / Argo) | parallelism=5 | 2 CPU, 4Gi | 一次性 |
| trainer-lora | Job (Argo) | 1 (gang) | 8 CPU, 16Gi, **A100 ×1** | RunOnce |
| drift-detector | CronJob | hourly | 100m CPU, 256Mi | 写 alert metric |
| episode-compactor | CronJob | daily | 500m CPU, 1Gi | JSONL → Parquet |

**决策原因**：multi_agent 是热点，需要水平扩；agent_memory 有状态，单实例 StatefulSet；agent_sft 占 GPU，单独节点；mcp_server 是 gateway，独立扩缩容。

### D2. agent 间通信协议（**必须**）

**选 gRPC + Protobuf**。理由：
- 跟现有 Pydantic schema 一一映射（用 `betterproto` 或 `protobuf` + 转换层）
- HTTP/2 多路复用 + 流式支持（实时 trace stream）
- 比 REST 快 3-5×，比 JSON-RPC 类型安全
- Istio 原生支持 gRPC 负载均衡 / 重试 / 熔断

**接口 proto 草案**：
```protobuf
service SchedulerService {
    rpc Schedule(ScheduleRequest) returns (ScheduleResponse);
    rpc Stats(google.protobuf.Empty) returns (StatsResponse);
    rpc TraceStream(stream TraceQuery) returns (stream TraceEvent);
}
service MemoryService {
    rpc Retrieve(RetrieveRequest) returns (RetrieveResponse);
    rpc Add(AddEpisodeRequest) returns (google.protobuf.Empty);
}
service AIOpsService {
    rpc Observe(OpsSnapshot) returns (Insight);
    rpc CurrentAlerts(google.protobuf.Empty) returns (AlertList);
}
```

**实施量**：写一个 `agent_grpc/` 模块，约 300 行 server + 200 行 stubs。可以用 `grpcio-tools` 自动生成。

### D3. 同进程 fast-path 怎么保留（**必须**）

加 `agent_common/runtime.py`：
```python
class Runtime:
    """根据 MODE env 自动选 local / grpc / mcp。"""
    @classmethod
    def get_scheduler(cls): ...  # 返回 callable
    @classmethod
    def get_memory(cls): ...
    @classmethod
    def get_aiops(cls): ...
```

multi_agent 内部如果要调 memory.retrieve，过 Runtime.get_memory() 拿；本地 mode 直接 import，grpc mode 走 gRPC stub。**业务代码零改动**。

### D4. State 持久化方案（**必须**）

| State | 方案 | 决策原因 |
|---|---|---|
| trace JSONL | PVC + EFS-style ReadWriteMany CSI | 多个 multi_agent pod 写同一个目录 |
| episodic memory | StatefulSet local PV + 备份到 S3 | 单实例热路径要快，冷备走 S3 |
| hybrid_stats | in-memory + push to Prometheus | 不需要重启恢复，重新累计即可 |
| AIOps active_alerts | Redis（共享） | 跨 multi_agent replica 看同一份 alert |
| SFT GGUF | initContainer 从 OCI image / OSS 拉 | 启动时 lazy load，model 当 image layer |
| LoRA adapter | PV snapshot + S3 sync | 每次训练写 PV + 异步同步 |

**checkpoint recovery 协议**：每个 agent 启动时调 `recover_from_trace(start_time=now-24h)`，readinessProbe 阻塞到回放完成。新增启动参数 `--replay-window=24h`。

### D5. Sandbox 隔离层级（**推荐**，呼应 JD "沙箱执行"）

| 层级 | 技术 | 用途 | 性能成本 |
|---|---|---|---|
| L1 | 默认 Linux ns + cgroups | 普通 agent | 1.0× |
| L2 | **gVisor** (runsc runtime) | agent_sft 跑外部 LLM 输出 | 1.05× |
| L3 | **Kata Containers** / Firecracker microVM | 用户自定义 Skill 沙箱 | 1.15× |

实施：节点装 gVisor，pod spec 加 `runtimeClassName: gvisor`。simulate "agent runtime sandbox" 给 JD 看。

### D6. ServiceAccount + RBAC（**必须**）

每个 agent 独立 SA，最小权限：
- multi_agent SA：read ConfigMap, write metric Service
- agent_memory SA：read/write PVC, read Redis Service
- trainer-lora SA: read/write LoRA PVC, read Secret(s3-credential), nothing else
- benchmark-runner SA：read multi_agent Service only

加 PodSecurityAdmission 标签：`restricted` for agent_sft, `baseline` for mcp_server。

### D7. NetworkPolicy（**必须**）

默认 `deny-all` policy + 显式 allow：
- mcp_server pod ← all ingress on port 8080
- multi_agent pod ← only mcp_server + benchmark-runner
- agent_memory pod ← only multi_agent + episode-compactor
- agent_sft pod ← only multi_agent on port 50053
- benchmark-runner pod → multi_agent / memory / aiops, nothing else

### D8. Helm Chart 结构（**必须**）

```
charts/agent-infra/
├── Chart.yaml                   appVersion: 0.1.0
├── values.yaml                  默认值
├── values-dev.yaml              开发环境 override
├── values-prod.yaml             生产环境 override
├── templates/
│   ├── _helpers.tpl
│   ├── multi-agent.yaml         Deployment + HPA + Service + ServiceMonitor
│   ├── agent-memory.yaml        StatefulSet + Service + PVC + ServiceMonitor
│   ├── agent-aiops.yaml         Deployment + Service + ServiceMonitor
│   ├── agent-sft.yaml           Deployment (gpu) + Service + ServiceMonitor
│   ├── mcp-server.yaml          Deployment + Service LB + Ingress
│   ├── dashboard.yaml           Deployment + Service + Ingress
│   ├── batch/
│   │   ├── benchmark-job.yaml   Job template (Argo invokes)
│   │   ├── trainer-job.yaml     GPU Job template
│   │   ├── drift-cron.yaml      CronJob
│   │   └── compactor-cron.yaml  CronJob
│   ├── configmap-skills.yaml    Skill registry 导出
│   ├── secret-template.yaml     不真正存 secret
│   ├── pvc.yaml                 PVC requests
│   ├── networkpolicy.yaml       deny-all + allow rules
│   ├── rbac.yaml                ServiceAccounts + Roles + RoleBindings
│   ├── psa.yaml                 PodSecurityAdmission labels
│   └── istio/
│       ├── virtualservice.yaml  canary routing
│       ├── destinationrule.yaml subset by version
│       └── peerauthentication.yaml  mTLS
└── README.md
```

Helm values 关键暴露：
```yaml
agent:
  multi_agent: { replicas: 3, resources: {...}, image: {...} }
  sft: { enabled: true, gpu: nvidia, model_pvc: sft-model }
storage:
  trace_pvc: { size: 50Gi, storageClass: standard }
  episode_pvc: { size: 10Gi, storageClass: fast }
observability:
  prometheus: { enabled: true, scrape_interval: 15s }
  jaeger: { enabled: true }
security:
  sandbox: gvisor    # none | gvisor | kata
  networkpolicy: strict
```

### D9. Observability stack（**必须**）

三层：

**Metrics（Prometheus）**：
- 每个 agent 加 `prometheus_client` exporter，暴露 `:9090/metrics`
- 把 hybrid_stats / aiops_stats / sft_stats 全部 export 成 Prometheus metrics（counter / histogram / gauge）
- Helm 装 prometheus-operator，ServiceMonitor CR 自动 scrape

**Tracing（OpenTelemetry）**：
- 每个 schedule_service / observe_ops_state 调用产生一个 root span
- 子 span：planner / scheduler / critic / memory.retrieve / aiops_critic_check
- 用 `opentelemetry-instrumentation-grpc` 自动注入跨 service trace
- 导出到 Jaeger / Tempo

**Logging**：
- TraceLogger 改成同时写 stdout（JSON line format）
- Fluent Bit DaemonSet 抓 stdout → Loki / ELK
- 保留 PVC trace 作为冷备

**Dashboard**：
- 现有 `dashboard/index.html` 重写为 Grafana dashboard JSON，入 Helm chart
- 关键面板：SLA violation rate（5min）/ AIOps risk_score 时序 / fast_path_ratio / inference_latency P95

### D10. Workflow Orchestration（**推荐**，呼应 JD "工作流编排"）

选 **Argo Workflows**。理由：
- K8s 原生（每个 step 是 Job）
- DAG + parameter passing + artifact passing 直接支持
- UI 好看（可演示）

Pipeline 设计：
```
build-dataset (Job) → train-lora (GPU Job) → eval-benchmark (parallel Jobs ×5 seeds) →
collect-metrics → drift-check → if metric_ok then deploy_canary → promote_after_24h
```

Workflow 文件 `argo/workflows/sft-pipeline.yaml`：约 100 行 YAML，配 ConfigMap 化的算法参数。

### D11. Service Mesh（**可选**，呼应 JD "服务治理"）

选 Istio。能力清单：
- mTLS：agent 间 gRPC 自动加密
- VirtualService：mcp_server canary（10% v2 / 90% v1）
- DestinationRule + subset：按版本分流
- RetryPolicy：default 3 retries + 50ms backoff
- CircuitBreaker：agent_sft 连续 5 失败 → 60s 熔断 → 落 balanced-fit fallback
- RateLimit：mcp_server 100 RPM / 租户

**坦白**：如果时间紧 Istio 是 nice-to-have。可以用 K8s 原生 Service + 自写 client retry 替代。

### D12. GPU 调度（**推荐 if agent_sft 上线**）

- 装 `nvidia-device-plugin` DaemonSet
- agent_sft Deployment：`resources.limits."nvidia.com/gpu": 1`
- 节点打 taint `nvidia.com/gpu=true:NoSchedule`，pod 加 toleration
- 训练 Job 用 **Volcano** 做 gang scheduling（确保所有 worker 同时 ready）
- MIG 切 A100 → 7 个 GPU instance，给小模型推理用（agent_sft 1.5B 一个 MIG 就够）

### D13. CI/CD pipeline（**推荐**）

GitHub Actions：
```yaml
on: push to main →
  test:                  pytest tests -q
  build:                 docker buildx multi-arch, push to ghcr
  package:               helm package + chart museum
  deploy-dev:            ArgoCD sync to dev cluster
  benchmark-regression:  Argo Workflow run benchmark, compare with baseline
  promote:               manual approval → ArgoCD sync to prod
```

也可以用 ArgoCD GitOps 模式：Helm values 仓库的 commit 自动同步集群。

### D14. Multi-tenancy（**可选**）

如果想模拟"多个团队共用一个 agent 平台"：
- 每个团队一个 Namespace
- ResourceQuota 限制 CPU / Memory / GPU
- NetworkPolicy 默认 deny cross-namespace
- mcp_server 在 ingress 层做 tenant 路由（JWT 中的 tenant claim → upstream namespace）

---

## 3. 实施阶段（6 phase，总工时 8 周）

| Phase | 时长 | 内容 | 简历能写的话 |
|---|---|---|---|
| **P1 容器化** | 1 周 | 写 6 个 multi-stage Dockerfile（distroless base）+ docker-compose.yml 本地多 service 跑通 | "多模块容器化部署，distroless 基础镜像" |
| **P2 Helm 上 K8s** | 2 周 | Helm chart 写好，minikube/kind 跑通，加 PVC + ConfigMap + Service + Ingress + RBAC + NetworkPolicy | "Helm chart, K8s native deployment, RBAC + NetworkPolicy 最小权限" |
| **P3 gRPC + 可观测** | 2 周 | 6 个 service 加 gRPC server / client + Prometheus exporter + ServiceMonitor + Jaeger trace + Grafana dashboard | "gRPC 服务化（proto + Pydantic 桥接），Prometheus + OpenTelemetry + Grafana 三层可观测体系" |
| **P4 Workflow** | 1 周 | Argo Workflows 把 build-dataset → train → eval → deploy 串成 DAG，配 CronJob 跑 drift detect 和 episode compactor | "Argo Workflows 编排 SFT 全流程；CronJob 跑漂移检测 + 数据压缩" |
| **P5 Service Mesh** | 1 周 | Istio sidecar + mTLS + canary + circuit breaker + rate limit | "Istio service mesh：mTLS、灰度发布、熔断、限流" |
| **P6 Sandbox + GPU** | 1 周 | gVisor runtime class + nvidia-device-plugin + Volcano gang scheduling + MIG | "gVisor 沙箱隔离 + GPU 调度 + Volcano gang scheduling" |

---

## 4. MVP 推荐：只做 P1 + P2 + 半个 P3（4 周）

**够用的最小集合**，能命中蚂蚁 JD 90% 关键词：

- ✅ 容器化（6 个 multi-stage Dockerfile）
- ✅ Helm chart（Deployment / StatefulSet / Service / Ingress / PVC / RBAC / NetworkPolicy）
- ✅ Prometheus metrics export（不强求 Jaeger trace）
- ✅ HPA + GPU nodeSelector
- ✅ minikube / kind 本地跑通的 demo

**先不做**：Istio / Argo / gVisor / canary / multi-tenant —— 这些都可以面试时口头说"如果给我两周可以加上"，反而比硬塞进个人项目更专业。

**简历能写**：

> 将 Agent Infra Toolkit 完整容器化并部署到 K8s：6 个 Service（multi_agent / memory / aiops / sft / mcp / dashboard）通过 Helm chart 统一发布；multi_agent 用 Deployment + HPA 基于 QPS 自动扩缩容，agent_memory 用 StatefulSet + PVC 持久化 episodic JSONL，agent_sft 用 nvidia-device-plugin + nodeSelector 调度 GPU；通过 ServiceAccount + 最小权限 RBAC + 默认 deny-all NetworkPolicy 实现网络隔离；每个 agent 用 prometheus-client 导出 15+ 业务指标，Grafana dashboard 实时观测 SLA / latency / fast_path_ratio。

---

## 5. 决策时刻：你接下来三个选择

| 选项 | 工时 | 简历加分 | 我的建议 |
|---|---|---|---|
| **不做 K8s** | 0 | 当前简历已经能投蚂蚁 80% 岗位 | 投递期内**不推荐**做这么大动作 |
| **MVP（P1+P2+半 P3）** | 4 周 | 直接命中蚂蚁/字节/阿里 Agent Infra 岗 90% 关键词 | 如果还有 4 周打磨期，**强烈推荐** |
| **完整 6 phase** | 8 周 | 可以投 SRE / Platform Engineer 岗 | 工时投入太大，**不推荐** |

---

## 6. 如果决定做，下一步

跟我说"MVP"或"完整"，我立刻动手：

**MVP 路径**：
1. 写 6 个 Dockerfile + docker-compose.yml（首日跑通）
2. 写 Helm chart 完整骨架（多 service + Service + PVC + RBAC + NetworkPolicy）
3. 在 multi_agent 加 prometheus_client metric exporter
4. 给 mcp_server 加 K8s liveness/readiness probe
5. 写 `docs/K8S_QUICKSTART.md` 一键 deploy 文档

**完整路径**：在上面基础上加 gRPC / Istio / Argo / gVisor，按 phase 推进。

不做也没事——这个项目当前的形态已经能撑住绝大多数 agent 应用岗位的面试。**K8s 化主要是给"Agent Infra"这个特定岗位族量身做的加分项**。
