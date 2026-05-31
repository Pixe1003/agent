# K8s Quickstart — Agent Infra Toolkit

> 从零部署到本地 kind / minikube 集群，约 10 分钟跑通。

## 一、本地 docker-compose 先验证

```powershell
# 1. 装 fastapi/uvicorn/prometheus-client (本地测试用)
pip install fastapi "uvicorn[standard]" prometheus-client

# 2. 本地单进程跑 multi_agent server (跳过 docker)
python -m multi_agent.serve
# 另开终端测一下
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
curl -X POST http://localhost:8080/schedule -H "Content-Type: application/json" `
    -d '{"servers":[[0,80,80,80],[1,70,70,70]],"service":[10,10,10],"aiops_risk_tags":[]}'

# 3. docker-compose 跑 4 个 service + prometheus + grafana
docker compose up --build
# multi_agent      → http://localhost:8080
# agent_memory     → http://localhost:8081
# agent_aiops      → http://localhost:8082
# prometheus       → http://localhost:9090   (Status → Targets 看 scrape OK)
# grafana          → http://localhost:3000   (admin / admin，加 prometheus 数据源)
docker compose down -v
```

## 二、本地 K8s 集群：kind

```powershell
# 1. 装 kind + kubectl + helm
choco install kind kubernetes-cli kubernetes-helm

# 2. 创建集群
kind create cluster --name agent-infra

# 3. 构建镜像并 load 到 kind
docker build -f docker/Dockerfile --target multi_agent -t agent/multi_agent:0.1 .
docker build -f docker/Dockerfile --target agent_memory -t agent/memory:0.1 .
docker build -f docker/Dockerfile --target agent_aiops -t agent/aiops:0.1 .
docker build -f docker/Dockerfile --target mcp_server -t agent/mcp:0.1 .

kind load docker-image agent/multi_agent:0.1 --name agent-infra
kind load docker-image agent/memory:0.1     --name agent-infra
kind load docker-image agent/aiops:0.1      --name agent-infra
kind load docker-image agent/mcp:0.1        --name agent-infra

# 4. Helm 安装
kubectl create namespace agent-infra
helm install agent-infra ./charts/agent-infra `
    --namespace agent-infra `
    --set observability.prometheus.serviceMonitor=false  # 跳过 prometheus-operator 依赖
    
# 5. 看 pod 拉起来
kubectl -n agent-infra get pods -w

# 6. 端口转发测一下
kubectl -n agent-infra port-forward svc/agent-infra-multi-agent 8080:8080
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics | findstr agent_

# 7. 清理
helm uninstall agent-infra -n agent-infra
kind delete cluster --name agent-infra
```

## 三、values.yaml 关键 override 范例

**dev (resource 拉到最小)**：
```yaml
multiAgent:
  replicas: 1
  hpa:
    enabled: false
  resources:
    requests: { cpu: 50m, memory: 128Mi }
    limits:   { cpu: 200m, memory: 256Mi }
agentSft:
  enabled: false   # 本地不跑 GPU
security:
  networkPolicy:
    enabled: false # kind 默认不装 cni policy
observability:
  prometheus:
    serviceMonitor: false
```

**prod (开 SFT + 全 observability + 严格 sec)**：
```yaml
global:
  imageRegistry: ghcr.io/your-handle/
  storageClass: fast-ssd
multiAgent:
  replicas: 5
  hpa:
    maxReplicas: 20
agentSft:
  enabled: true
  replicas: 2
  modelPVC:
    size: 20Gi
mcpServer:
  service:
    type: LoadBalancer
  ingress:
    enabled: true
    hosts:
      - host: mcp.your-domain.com
        paths: [{ path: /, pathType: Prefix }]
```

## 四、可观测性 verify

部署后 pod 上的 metrics 默认 export 在 `:8080/metrics`，关键 metric：

```
agent_requests_total{agent="multi_agent",endpoint="/schedule",status="200"}
agent_request_latency_seconds_bucket{agent="multi_agent",endpoint="/schedule",le="0.001"}
agent_inflight_requests{agent="multi_agent",endpoint="/schedule"}
agent_ready{agent="multi_agent"}
```

Grafana 推荐面板（用 Prometheus query）：
- 调度 QPS：`rate(agent_requests_total{agent="multi_agent",endpoint="/schedule"}[1m])`
- 延迟 P95：`histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m]))`
- 错误率：`sum(rate(agent_requests_total{status!~"2.."}[5m])) / sum(rate(agent_requests_total[5m]))`
- AIOps trigger ratio：`agent_inflight_requests{agent="agent_aiops"}`

## 五、故障排查

| 症状 | 排查 |
|---|---|
| pod ImagePullBackOff | `kind load docker-image` 是否漏跑；或 image tag mismatch |
| readyz 永不 ready | exec 进 pod 跑 `python -m multi_agent.serve --help`；日志看 init 卡哪 |
| memory pod 反复重启 | PVC 没 mount；检查 StorageClass 是否存在 |
| metrics 看不到 | Service annotation `prometheus.io/scrape: "true"` 是否在；NetworkPolicy 是否阻断 |
| HPA never scales | metrics-server 是否装（`kubectl top pod` 出数据） |

## 六、上线 checklist

- [ ] PSA labels enforced (`restricted` for sft, `baseline` for others)
- [ ] image scan 通过（trivy / grype）
- [ ] image 用 cosign 签名 + admission webhook 校验
- [ ] Secret 走 External Secrets / Vault，不入 Git
- [ ] BackupJob 定期备份 PVC episodes.jsonl → S3
- [ ] Liveness/Readiness 已实测重启 + 失败回滚场景
- [ ] 三 nines SLO：multi_agent endpoint p95 < 5ms（不含 LLM 路径）
