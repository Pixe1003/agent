;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             ____                                                           ;;
;;            / ___| _   _ _ __  _ __  _   _                                  ;;
;;            \___ \| | | | '_ \| '_ \| | | |                                 ;;
;;             ___) | |_| | | | | | | | |_| |                                 ;;
;;            |____/ \__,_|_| |_|_| |_|\__, |                                 ;;
;;                                     |___/    v1.04 (rc)                    ;;
;;                                                            Dapeng Dong     ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
extensions [ py ]
;;
globals
[
  ;;--Layout Related----------------------------------------------------------;;
  ;; The height of the service submission zone in the world.
  service-submission-zone-height

  ;; The height of the space for placing the SCHEDULER nodes in the world.
  svr-scheduler-placement-zone-height

  ;; The height of the delay zone between the 'service submission zone'
  ;; and the 'scheduler zone'. This zone is used for adding some random
  ;; factors to the submission of services. When services are created
  ;; in the 'service submission zone', each of which will be assigned a
  ;; default moving speed. This creates an effect that services will not
  ;; arrive at scheduler nodes at the same time.
  service-submission-delay-zone-height

  ;; The default width of the separation line in the world.
  def-sepa-line-width

  ;; The height for the gaps between different types of objects.
  def-gap-width

  ;; Used for tracing the coordinates of objects.
  current-top-cord
  current-bottom-cord

  ;; Display agents' information
  show-server-label?
  show-server-model?
  show-scheduler-label?
  show-service-label?
  show-service-attempt?


  ;;--Service Related--------------------------------------------------------;;
  sys-services-waiting

  service-method-ct-delay
  service-method-vm-delay

  service-lifetime-max
  service-lifetime-min

  service-mem-access-ratio-beta-alpha
  service-mem-access-ratio-beta-beta

  service-cpu-usage-dist-random?
  service-cpu-usage-dist-beta-alpha
  service-cpu-usage-dist-beta-beta

  service-mem-usage-dist-random?
  service-mem-usage-dist-beta-alpha
  service-mem-usage-dist-beta-beta

  service-net-usage-dist-random?
  service-net-usage-dist-beta-alpha
  service-net-usage-dist-beta-beta


  ;;--Server Related--------------------------------------------------------;;
  server-standby-factor
  server-cold-start-delay

  server-cpu-underutil-threshold
  server-cpu-overutil-threshold

  server-mem-underutil-threshold
  server-mem-overutil-threshold

  server-net-underutil-threshold
  server-net-overutil-threshold


  ;;--System Related--------------------------------------------------------;;
  sys-random-seed


  ;;--Global Reporter-------------------------------------------------------;;
  sys-current-active-servers           ;; Current active servers
  sys-current-active-svr-ops-util      ;; CPU utilization of CURRENT active servers
  sys-current-active-svr-mem-util      ;; MEM utilization of CURRENT active servers
  sys-current-active-svr-net-util      ;; NET utilization of CURRENT active servers


  sys-accumulated-service-ops-sla-vio  ;; Accumumulated CPU SLA violation
  sys-accumulated-service-mem-sla-vio  ;; Accumumulated MEM SLA violation
  sys-accumulated-service-net-sla-vio  ;; Accumumulated NET SLA violation
  sys-current-service-ops-sla-vio      ;; Current CPU SLA violation
  sys-current-service-mem-sla-vio      ;; Current MEM SLA violation
  sys-current-service-net-sla-vio      ;; Current NET SLA violation

  sys-current-service-completed

  sys-service-rejection-counter        ;; The total number of services that can't be deployed in the cloud
  sys-service-reschedule-counter       ;; The total number of services that have been re-tried for their deployment.

  sys-current-service-delay-due-to-server-cold-start
  sys-current-service-delay-due-to-deployment-method
  sys-accumulated-service-delay-due-to-server-cold-start
  sys-accumulated-service-delay-due-to-deployment-method

  sys-current-migration-event-due-to-consolidation
  sys-current-migration-event-due-to-auto-migration
  sys-accumulated-migration-event-due-to-consolidation
  sys-accumulated-migration-event-due-to-auto-migration

  sys-service-lifetime-total
  sys-power-consumption-total

  sys-accumulated-delay-from-vm
  sys-accumulated-delay-from-ct
  sys-accumulated-delay-from-svr-cold-start

  ai-usage-count    ;; 成功使用AI的次数
  total-usage-count ;; 总调用次数

]
;;
;;============================================================================;;
;;--Breeds--------------------------------------------------------------------;;
breed  [ servers      server      ]
breed  [ schedulers   scheduler   ]
breed  [ services     service     ]
breed  [ vis-agents   vis-agent   ]
;;----------------------------------------------------------------------------;;
vis-agents-own
[
  from-svr
  to-svr
  moving-speed
]
;;
services-own
[
  id
  host ;; The ID of its current hosting server
  ops-cnf   mem-cnf   net-cnf
  ops-now   mem-now   net-now
  ops-prev  mem-prev  net-prev
  ops-sla   mem-sla   net-sla
  ops-hist  mem-hist  net-hist

  method
  life-time
  access-ratio
  moving-speed

  status

  delay-counter
  attempt
]
;;
servers-own
[
  id     ;; The server index in the rack
  rack   ;; Rack id, it is the same as its corresponding Scheduler's id
  model  ;; Different models have different specifications.

  ;; Status of a server can be 'REPAIR', 'IDLE', 'READY', 'OVERLOAD, and
  ;; 'OFF'. When a server is created, its status is set to 'OFF'.
  status

  ops-phy   mem-phy   net-phy  ;; The phyically installed the resources of the server.
  ops-now   mem-now   net-now  ;; The currently used resources.
  ops-rsv   mem-rsv   net-rsv  ;; The reserved resources for incoming services.
  ops-hist  mem-hist  net-hist ;; The resource usage history on this server.

  power      ;; Current energy consumed.
  base-power ;; The energy consumption when it is ON but no running services.
]
;;
schedulers-own
[
  id         ;; Rack id
  capacity   ;; A queue size that will be used in the future development
  ops-hist  mem-hist  net-hist
]
;;
;;----------------------------------------------------------------------------;;
to setup
  clear-all
  py:setup ".\\.venv\\Scripts\\python.exe"
  (py:run
    "import numpy as np"
    ;;"from sklean.linear_model import LinearRegression"
  )

  ifelse rand-seed <= 0
  [
    set sys-random-seed new-seed
    random-seed sys-random-seed
    py:set "simu_seed" abs(sys-random-seed)
  ]
  [
    random-seed rand-seed
    py:set "simu_seed" abs(rand-seed)
  ]

  py:run "np.random.seed(simu_seed)"


  ;;--Layout Related----------------------------------------------------------;;
  set service-submission-zone-height       4
  set svr-scheduler-placement-zone-height  4
  set service-submission-delay-zone-height 3
  set def-sepa-line-width 1
  set def-gap-width 1
  set current-top-cord max-pycor
  set current-bottom-cord min-pycor

  set show-server-label?      true
  set show-server-model?      true
  set show-scheduler-label?   true
  set show-service-label?     true
  set show-service-attempt?   true


  ;;--Service Related---------------------------------------------------------;;
  ;; 5  seconds, converted to 'sumulation-time-unit'
  set service-method-ct-delay 5 / (simulation-time-unit * 60)
  ;; 15 seconds, converted to 'sumulation-time-unit'
  set service-method-vm-delay 15 / (simulation-time-unit * 60)

  set service-lifetime-max max (read-from-string service-lifetime)
  set service-lifetime-min min (read-from-string service-lifetime)

  py:set "service_mem_access_ratio_beta_alpha" (first (read-from-string mem-access-ratio))
  py:set "service_mem_access_ratio_beta_beta"  (last  (read-from-string mem-access-ratio))


  ifelse (first (read-from-string cpu-usage-dist)) != 0 or (last (read-from-string cpu-usage-dist)) = 0
  [
    set service-cpu-usage-dist-random? true
    py:set "service_cpu_usage_dist_beta_alpha" (first (read-from-string cpu-usage-dist))
    py:set "service_cpu_usage_dist_beta_beta"  (last  (read-from-string cpu-usage-dist))
  ]
  [ set service-cpu-usage-dist-random? false ]


  ifelse (first (read-from-string mem-usage-dist)) != 0 or (last (read-from-string mem-usage-dist)) != 0
  [
    set service-mem-usage-dist-random? true
    py:set "service_mem_usage_dist_beta_alpha" (first (read-from-string mem-usage-dist))
    py:set "service_mem_usage_dist_beta_beta"  (last  (read-from-string mem-usage-dist))
  ]
  [ set service-mem-usage-dist-random? false ]


  ifelse (first (read-from-string net-usage-dist)) != 0 and (last (read-from-string net-usage-dist)) != 0
  [
    set service-net-usage-dist-random? false
    py:set "service_net_usage_dist_beta_alpha" (first (read-from-string net-usage-dist))
    py:set "service_net_usage_dist_beta_beta"  (last  (read-from-string net-usage-dist))
  ]
  [ set service-net-usage-dist-random? true  ]


  ;;--Server Related----------------------------------------------------------;;
  ;; Server cold booting delay is fixed at 15 seconds, measured in 'simulation-time-unit'.
  set server-cold-start-delay 15 / (simulation-time-unit * 60)

  set server-cpu-overutil-threshold  ((max (read-from-string server-cpu-utilization-threshold)) / 100)
  set server-cpu-underutil-threshold ((min (read-from-string server-cpu-utilization-threshold)) / 100)

  set server-mem-overutil-threshold  ((max (read-from-string server-mem-utilization-threshold)) / 100)
  set server-mem-underutil-threshold ((min (read-from-string server-mem-utilization-threshold)) / 100)

  set server-net-overutil-threshold  ((max (read-from-string server-net-utilization-threshold)) / 100)
  set server-net-underutil-threshold ((min (read-from-string server-net-utilization-threshold)) / 100)


  (ifelse
    server-standby-strategy = "adaptive" [ set server-standby-factor 2   ]
    server-standby-strategy = "all-off"  [ set server-standby-factor 0   ]
    server-standby-strategy = "all-on"   [ set server-standby-factor 1   ]

    ;; Ensure at least 10% of the servers in the rack is switched ON and so on.
    server-standby-strategy = "10%-on"   [ set server-standby-factor 0.1 ]
    server-standby-strategy = "20%-on"   [ set server-standby-factor 0.2 ]
    server-standby-strategy = "30%-on"   [ set server-standby-factor 0.3 ]
    server-standby-strategy = "40%-on"   [ set server-standby-factor 0.4 ]
    server-standby-strategy = "50%-on"   [ set server-standby-factor 0.5 ]
  )

  ;;--System Related----------------------------------------------------------;;
  set sys-service-rejection-counter  0
  set sys-service-reschedule-counter 0

  set sys-current-active-servers 0
  set sys-current-active-svr-ops-util 0
  set sys-current-active-svr-mem-util 0
  set sys-current-active-svr-net-util 0

  set sys-power-consumption-total 0

  initialize-datacenter

  if total-services < service-generation-speed [ set service-generation-speed total-services ]
  generate-client-services service-generation-speed
  set sys-services-waiting total-services - service-generation-speed


  (py:run
    "import sys, os"
    "sys.path.insert(0, os.getcwd())"
    "from agent_phase1 import init_agent, schedule_service, last_decision_summary"
    "from agent_phase2 import init_agent as init_agent_phase2, schedule_service as schedule_service_phase2, last_decision_summary as last_decision_summary_phase2"
    "from agent_phase3 import init_agent as init_agent_phase3, schedule_service as schedule_service_phase3, last_decision_summary as last_decision_summary_phase3"
    "init_agent(model_name='qwen3:8b', temperature=0.1)"
    "init_agent_phase2(model_name='heuristic')"
    "init_agent_phase3(model_name='heuristic')"
  )

  reset-ticks

end
;;
;;
to go
  tick

  ;; Track how many client services are in the service pool. To maintain the
  ;; concurrency value of the service submission events, i.e., keeping the
  ;; number of services specified by 'service-generation-speed', new
  ;; services need to be continuously added to the service pool until all the
  ;; services are submitted.
  if sys-services-waiting > 0
  [
    let services-in-the-pool count services-on patches with [pcolor = blue + 1]
    if sys-services-waiting > 0 and services-in-the-pool < service-generation-speed
    [
      let more-services (service-generation-speed - services-in-the-pool)
      if more-services > sys-services-waiting [ set more-services sys-services-waiting ]
      generate-client-services more-services
      set sys-services-waiting (sys-services-waiting - more-services)
    ]
  ]

  ;; Ask all the services to perform their routines.
  update-services-status

  ;; Update schedulers' routines.
  update-scheduler-status

  ;; Server status will be updated on every tick of the simulation.
  update-servers-status

  ;; Consolidation events only occur when
  ;;   1. at least two servers are busy,
  ;;   2. the scheduled interval ('consolidation-interval') is reached, and
  ;;   3. the 'consolidation?' switch is ON.
  if (consolidation?)
  [ consolidate-servers ]

  ;; Update visual movement of service migration.
  update-vis-agents-status

  if not any? services
  [
    ask servers [ set status "OFF" set color white set power 0 reset-server self ]
    ask vis-agents [ die ]
    print-summary
    stop
  ]

end
;;
;;
;;============================================================================;;
;; During the simulation, services' status are updated on a
;; per 'tick' basis. On each update (tick), the lifetime of each service will
;; be reduced by 1. At the same time, the resource usage of each service will
;; be updated (the distribution on the resource usage follows a Beta
;; distribution).
;; In principle, each update of the resource usage should be in range
;; [0, ops/mem/net-cnf], i.e., it can't exceed the resources initially
;; configured (ops/mem/net-cnf) at the deployment time of the service,
;; neither a service can consume negative resources. However, it can become
;; complicated when multiple services are being deployed on the same server
;; at different times. For example, when Service-1 was 'SCHEDULED' to run
;; on Server-1 at Time-1, the required resources of Service-1 would first be
;; reserved on Server-1. When Service-1 arrives at Server-1, its resource
;; usage will be updated and, more importantly, the updated resource usage
;; would very likely be smaller than the initially configured resources.
;; If at Time-2 (e.g., after Service-1 has already started running on
;; Server-1), Service-2, had been scheduled to run on the same server
;; (Server-1), then during the scheduling process, the scheduler must
;; ensure that the Server-1 has sufficient resources for the deployment of
;; the Service-2, more specifically, this is determined by the condition:
;; (currently occupied resources of Server-1 + currently reserved resources of
;; Server-1 + the configured resources of Service-2) is greater or equal to
;; (the physically installed resources of Server-1).
;; In this equation, the 'currently occupied resources of Server-1' varies
;; from time to time. In this very specific example, it is only the current
;; resource usage of Service-1, since it is the only service running on the
;; server. If we further assume that (1) the total physical memory of
;; Server-1 is 2GB, (2) the current updated memory resource usage of
;; Service-1 is only a half of its requested memory (e.g., mem-now = 0.5,
;; mem-cnf = 1), and (3) the requested memory of Service-2 is 1.5GB (i.e.,
;; mem-cnf = 1.5). Based on the information assumed above, Server-1 would be
;; an eligible candidtate for the deployment of Service-2. When Service-2
;; starts running on Server-1,both services (Service 1 and 2) will update
;; their resource usage at the same time. Thus, it is possible when,
;; at Time-3, both services had updated their memory resource usage to
;; 1GB (Service-1) and 1.2GB (Service-2), respectively. However, the total
;; requested memory, at this moment, has exceeded the physical memory capacity
;; of Server-1. If this happens, the performance of the two
;; services will be degenerate. To compensate, both Service-1/2's
;; lifetime will be extended (i.e., requires more time to complete their
;; tasks). The aforementioned phenomenon will occur more frequently after
;; server consolidations are performed.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to update-services-status
  ask services
  [
    ;; When a service has reacheed the end of its lifetime, or it has been
    ;; rejected three times, it dies.
    if life-time <= 0 and status != "MIGRATING"
    [ set sys-current-service-completed sys-current-service-completed + 1
      die
    ]

    if attempt > 2
    [
      set sys-service-rejection-counter sys-service-rejection-counter + 1
      die
    ]

    ;; Update the visual movement of services during submission and scheduling
    ;; processes.
    (ifelse
      ;; When a service is on its way to the designated scheduler.
      status = "OFFLINE"
      [ ;; Check if it has arrived at the scheduler
        ifelse distance-nowrap scheduler host > 0.5
        [ face-nowrap scheduler host fd moving-speed ]
        [ ;; If arrived, change its status to "SUBMITTED". When a scheduler
          ;; sees a service with "SUBMITTED" status, the scheduler will
          ;; find a suitable server for the service.
          move-to scheduler host
          set status "SUBMITTED"
          set moving-speed 0
        ]
      ]
      ;; If a scheduler sees a service with a status 'SUBMITTED',
      ;; the scheduler will find a suitable server for the deployment of the
      ;; service, and the status of the service is then changed to 'SCHEDULED',
      ;; followed by moving the service to its designated server.
      status = "SCHEDULED"
      [
        ifelse distance-nowrap server host > 0.5
        [ face-nowrap server host fd moving-speed ]
        [ ;; If the service has arrived at its designated server,
          ;; its status will be changed to "DEPLOYED".
          move-to server host
          set status "DEPLOYED"
          set color orange
          set moving-speed 0
        ]
      ]

      ;; If a server is thought to be under- or over-utilized, all or some of
      ;; the running services will be migrated to other servers. On the same
      ;; server, if a service is identified as a migrant, the service will
      ;; create a replica. The original service will be moved to the
      ;; migrating destination server immediately, and its status will be changed
      ;; to 'DEPLOYED' with no further delays. The replica will stay on
      ;; the same server. This replica will occupy the same amount of resources
      ;; as of the migrating service, except for the net resource, which will be
      ;; set to the amount of network bandwidth required for migration (partially
      ;; determined by the memory access rate * mem-now).
      ;; Additionally, the replica will be given a lifetime according to the
      ;; migration delay. After the lifetime has reached, the replicas will die
      ;; quietly. The process reflect the real-world situation in which a service
      ;; migrating to another server, it occupies some reserouces on both
      ;; source and destinatin servers, until the migration process is completed.
      ;; In addition, to visualize the migration process i.e., the
      ;; 'show-migr-move?' = TRUE, a vis-agent agent will be created to show
      ;; the movement on the display. However, any visualization related agents
      ;; will be categorized into a diffent breed, thus they do not get
      ;; involved in any calculation of resources.
      ;; NOTE: during migration, the resource utilizations of replicas will
      ;; maintain static. The process is implemented in detail in the
      ;; 'migrate-services' procedure. Here we just updated the status of the
      ;; replicas.
      status = "MIGRATING"
      [ set life-time life-time - 1 ]

      ;; When a server receives a service, the server will change the status
      ;; of the service to "RUNNING".
      ;; Update the service status and resource usages if they are 'RUNNING'.
      ;; Note that resource usage freezes for replicas of migrants.
      status = "RUNNING"
      [
        ;; The lifetime of the service. It decreases on a per-tick basis.
        set life-time life-time - 1

        ;; During the runtime of the service, the runtime resources used may be
        ;; lower than the 'configured', but it should not be more than the
        ;; configured resources. The resource update follows either a Gaussian or
        ;; Beta distribution.
        let res-req-now 0
        ifelse service-cpu-usage-dist-random?
        [ set res-req-now (py:runresult "np.random.rand()" * ops-cnf) ]
        [ set res-req-now (py:runresult "np.random.beta(service_cpu_usage_dist_beta_alpha, service_cpu_usage_dist_beta_beta)" * ops-cnf) ]
        set ops-prev ops-now ;; Cache the previous resource usage. This will be used to calculate SLA violation.
        set ops-now round res-req-now

        ifelse service-mem-usage-dist-random?
        [ set res-req-now (py:runresult "np.random.rand()" * mem-cnf) ]
        [ set res-req-now (py:runresult "np.random.beta(service_mem_usage_dist_beta_alpha, service_mem_usage_dist_beta_beta)" * mem-cnf) ]
        set mem-prev mem-now
        set mem-now round res-req-now

        ifelse service-net-usage-dist-random?
        [ set res-req-now (py:runresult "np.random.rand()" * net-cnf) ]
        [ set res-req-now (py:runresult "np.random.beta(service_net_usage_dist_beta_alpha, service_net_usage_dist_beta_beta)" * net-cnf) ]
        set net-prev net-now
        set net-now round res-req-now

        ;; Saving the resource histories for resource forecast
        set ops-hist fput ops-now ops-hist
        set mem-hist fput mem-now mem-hist
        set net-hist fput net-now net-hist
        if (length ops-hist) > service-history-length
        [ ;; A circular list, expensive operations
          set ops-hist remove-item service-history-length ops-hist
          set mem-hist remove-item service-history-length mem-hist
          set net-hist remove-item service-history-length net-hist
        ]
      ]
    )

    ifelse show-trace?
    [ pendown ]
    [ penup ]
  ]

end
;;
;;
;;============================================================================;;
;; If there is any migrating services, its associated visualization agent,
;; i.e., vis-agent, will be updated on the display. When a vis-agent has
;; reached its destination server, it dies quietly.
;;--PARAMETERS----------------------------------------------------------------;;
;;  none
;;============================================================================;;
to update-vis-agents-status
  ;;show (word "num of vis-agents:" (count vis-agents))
  ask vis-agents [
    ifelse distance-nowrap server to-svr > 1.2
    [ face-nowrap server to-svr fd moving-speed ]
    [ die ]
  ]

end
;;
;;
;;============================================================================;;
;; Servers' status is updated every 'tick'. When calculating servers' status,
;; only the 'ops/mem/net-now' are used. Note that the 'ops/mem/net-rsv'
;; are not actually consumed resources, but reserved.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to update-servers-status
  set sys-current-service-ops-sla-vio 0
  set sys-current-service-mem-sla-vio 0
  set sys-current-service-net-sla-vio 0
  let pow-temp 0

  ;; Clear the reserved resources for the new services that are about
  ;; running on the server.
  ask servers
  [
    let newly-arrived-services services-here with [status = "DEPLOYED"]
    if count newly-arrived-services > 0
    [
      set ops-rsv ops-rsv - (sum [ops-cnf] of newly-arrived-services)
      set mem-rsv mem-rsv - (sum [mem-cnf] of newly-arrived-services)
      set net-rsv net-rsv - (sum [net-cnf] of newly-arrived-services)
      ask newly-arrived-services [ set status "RUNNING" ]
    ]

    ;; Update the available resources of the server based on the current
    ;; resource usage of the running services. If the requested resources
    ;; are greater than the physical resources installed on the server,
    ;; a violation of SLA will be recorded and the 'ops/mem/net-now' will
    ;; be capped at the maximum of the physical resources installed, i.e.,
    ;; 'ops/mem/net-phy'. In addition, services requesting for resources
    ;; that can't be allocated will be penalized by extending their
    ;; life-time, to reflect the service performance degeneration.
    ;; This will be accumulated from the calculations of all
    ;; types of resources.
    let running-services services-here with [ status = "RUNNING" or status = "MIGRATING" ]
    ifelse (count running-services) > 0
    [ ;; For ops shortage
      ;; We only care about the actual usage of the resoruces. The
      ;; 'ops-rsv' are reserved resources. They are not acutally used,
      ;; thus 'ops-rsv' is not taking part in this calculation.
      let sum-service-Rr sum [ops-now] of running-services
      let sla-component-1 ((sum-service-Rr - ops-phy) / ops-phy)
      ifelse sla-component-1 > 0
      [
        set ops-now ops-phy
        let services-res-delta []
        let res-min 999999999
        ask running-services
        [
          let res-delta (ops-now - ops-prev)
          if res-min > res-delta [ set res-min res-delta ]
          set services-res-delta lput (list who res-delta) services-res-delta
        ]

        ;; Distribution factors
        let sla-component-2 cal-sla-dist-factors services-res-delta res-min
        foreach sla-component-2
        [
          x -> ask service (first x)
          [
            set ops-sla (last x) * sla-component-1
            if use-sla-local-factor?
            [
              let sla-component-3 (1 + (ops-now - ops-prev) / ops-cnf)
              set ops-sla ops-sla * sla-component-3
            ]
            set ops-now (ops-now * (1 - (last x)))
            set life-time life-time + ops-sla
            set sys-current-service-ops-sla-vio sys-current-service-ops-sla-vio + ops-sla
            set sys-accumulated-service-ops-sla-vio sys-accumulated-service-ops-sla-vio + ops-sla
          ]
        ]
      ]
      [
        set ops-now sum-service-Rr
      ]

      ;; For mem shortage
      set sum-service-Rr sum [mem-now] of running-services
      set sla-component-1 ((sum-service-Rr - mem-phy) / mem-phy)
      ifelse sla-component-1 > 0
      [
        set mem-now mem-phy
        let services-res-delta []
        let res-min 999999999
        ask running-services
        [
          let res-delta (mem-now - mem-prev)
          if res-min > res-delta [ set res-min res-delta ]
          set services-res-delta lput (list who res-delta) services-res-delta
        ]

        ;; Distribution factors
        let sla-component-2 cal-sla-dist-factors services-res-delta res-min
        foreach sla-component-2
        [
          x -> ask service (first x)
          [
            set mem-sla (last x) * sla-component-1
            if use-sla-local-factor?
            [
              let sla-component-3 (1 + (mem-now - mem-prev) / mem-cnf)
              set mem-sla mem-sla * sla-component-3
            ]
            set mem-now (mem-now * (1 - (last x)))
            set life-time life-time + mem-sla
            set sys-current-service-mem-sla-vio sys-current-service-mem-sla-vio + mem-sla
            set sys-accumulated-service-mem-sla-vio sys-accumulated-service-mem-sla-vio + mem-sla
          ]
        ]
      ]
      [
        set mem-now sum-service-Rr
      ]

      ;; For net shortage
      set sum-service-Rr sum [net-now] of running-services
      set sla-component-1 ((sum-service-Rr - net-phy) / net-phy)
      ifelse sla-component-1 > 0
      [
        set net-now net-phy
        let services-res-delta []
        let res-min 999999999
        ask running-services
        [
          let res-delta (net-now - net-prev)
          if res-min > res-delta [ set res-min res-delta ]
          set services-res-delta lput (list who res-delta) services-res-delta
        ]

        ;; Distribution factors
        let sla-component-2 cal-sla-dist-factors services-res-delta res-min
        foreach sla-component-2
        [
          x -> ask service (first x)
          [
            set net-sla (last x) * sla-component-1
            if use-sla-local-factor?
            [
              let sla-component-3 (1 + (net-now - net-prev) / net-cnf)
              set net-sla net-sla * sla-component-3
            ]
            set net-now (net-now * (1 - (last x)))
            set life-time life-time + net-sla
            set sys-current-service-net-sla-vio sys-current-service-net-sla-vio + net-sla
            set sys-accumulated-service-net-sla-vio sys-accumulated-service-net-sla-vio + net-sla
          ]
        ]
      ]
      [
        set net-now sum-service-Rr
      ]
    ]
    [
      set ops-now 0
      set mem-now 0
      set net-now 0
    ]

    ;; Update server status indicator
    (ifelse
      (ops-now / ops-phy) > server-cpu-overutil-threshold or
      (mem-now / mem-phy) > server-mem-overutil-threshold or
      (net-now / net-phy) > server-net-overutil-threshold
      [ set status "OVERLOAD" set color red ]

      ops-now > 0 or mem-now > 0 or net-now > 0
      [ set status "ON" set color green ]

      (ops-now + ops-rsv) = 0 and
      (mem-now + mem-rsv) = 0 and
      (net-now + net-rsv) = 0 and
      (status = "ON" or status = "OVERLOAD")
      [ set status "IDLE" set color blue ]
    )

    (ifelse
      status = "REPAIR" [ set color grey  reset-server self set power 0 ]
      status = "OFF"    [ set color white reset-server self set power 0 ]
    )

    if status != "OFF" and status != "REPAIR"
    [
      set power calc-energy-consumption ops-now model
      set pow-temp pow-temp + power
    ]
  ]

  set sys-power-consumption-total sys-power-consumption-total + ((pow-temp * simulation-time-unit) / (60 * 1000))
  let actsvr servers with [status = "ON" or status = "OVERLOAD"]
  set sys-current-active-servers count actsvr
  if sys-current-active-servers > 0
  [
    set sys-current-active-svr-ops-util (sum [ops-now] of actsvr) / (sum [ops-phy] of actsvr)
    set sys-current-active-svr-mem-util (sum [mem-now] of actsvr) / (sum [mem-phy] of actsvr)
    set sys-current-active-svr-net-util (sum [net-now] of actsvr) / (sum [net-phy] of actsvr)
  ]

  if auto-migration?
  [
    ask servers with [ status = "OVERLOAD" ]
    [ auto-migrate self ]
  ]

end
;;
;;
;;============================================================================;;
;; Calculate energy consumption based on the CPU utilization.
;;--PARAMETERS------------------------------------------------------------------
;;  'the-ops-now'   : the amounts of computing power used now.
;;  'the-svr-model' : the model of the hosting server.
;;============================================================================;;
to-report calc-energy-consumption [ the-ops-now the-svr-model ]
  let energy 0
  (ifelse
    power-model-method = "stepwise-simple-linear-regression"
    [ set energy calc-power-consumption-stepwise the-ops-now  the-svr-model ]
    power-model-method = "simple-linear-regression"
    [ set energy calc-power-consumption-simple the-ops-now  the-svr-model ]
    power-model-method = "quadratic-polynomial"
    [ set energy calc-power-consumption-quadratic the-ops-now the-svr-model ]
    power-model-method = "cubic-polynomial"
    [ set energy calc-power-consumption-cubic the-ops-now  the-svr-model    ]
  )

  report energy

end
;;
;;
;;============================================================================;;
;; Generate services and initialize their status.
;;--PARAMETERS------------------------------------------------------------------
;;  'amount' : the number of services to be generated.
;;============================================================================;;
to generate-client-services [ amount ]
  ask n-of amount patches with [ pcolor = blue + 1 ]
  [
    sprout-services 1
    [
      set shape "circle"
      set color yellow
      set size 0.7

      ;; Each service has a different lifetime.
      set life-time random (service-lifetime-max - service-lifetime-min) + service-lifetime-min
      set sys-service-lifetime-total sys-service-lifetime-total + life-time
      ;; Allowing services to have different configurations. If this
      ;; is not needed, use a single value in the following lists.
      set ops-cnf one-of (list 25000 50000 100000 150000 200000);; ssj-ops
      set mem-cnf one-of (list 512 1024 2048 4096 8192 16384)   ;; MB
      set net-cnf one-of (list 10 20 50 100 200 500 1000)       ;; MBps

      ;; The 'method' specifies how a service will be depoyed.
      ;; It can be 'VM' (VIRTUAL-MACHINE), 'CT' (CONTAINER) or
      ;; 'BM' (BARE-METAL). Each deployment method is associated with a different
      ;; deployment delay.
      set method one-of (list "BM" "VM" "CT")

      ;; When a service has just started running, it is assumed that the
      ;; service will consume the same amount of resources as of the
      ;; requested resources.
      set ops-now ops-cnf
      set mem-now mem-cnf
      set net-now net-cnf

      set ops-hist []
      set ops-hist fput ops-cnf ops-hist
      set mem-hist []
      set mem-hist fput mem-cnf mem-hist
      set net-hist []
      set net-hist fput net-cnf net-hist

      ;; The memory access ratio (aka, memory dirtying rate) determines how
      ;; busy the system memory is being accessed, i.e., the amounts of system
      ;; memory that are frequently accessed. This will affect the efficiency of
      ;; service live migration.
      set access-ratio py:runresult "np.random.beta(service_mem_access_ratio_beta_alpha, service_mem_access_ratio_beta_beta)"

      ;; The strategies for sending a service to a scheduler node.
      ;; Strategy 1: send a service to its closest scheduler node.
      ;; Strategy 2: send a service to a rack on which the requested resources
      ;; match resource usage pattern of the servers. If a datacenter contains
      ;; homogeneous servers, the pattern will be calculated based on currently
      ;; available resources on the rack (aggregated).
      (ifelse
        service-submission-strategy = "closest"
        [ set host [who] of (min-one-of schedulers [distance myself]) ]
        service-submission-strategy = "resource-pattern-matching"
        [ set host (calc-resource-pattern self) ]
      )

      ;; Calculate the initial moving speed from the service pool to
      ;; the designated scheduler node.
      set moving-speed (random-float 0.6) + 0.05

      set status "OFFLINE"
      set attempt 0
      set delay-counter 0  ;; The delays caused by the server cold start.
    ]
  ]

end
;;
;;
;;############################################################################;;
;; START: Scheduler Related
;;############################################################################;;
;;
;;============================================================================;;
;; Update scheduler.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to update-scheduler-status
  ask schedulers
  [
    ;; The history of the total resources requested from the rack is recorded.
    ;; This information is used for the server standby strategy.
    ;;update-rack-resource-request-history

    ;; Apply server standby strategy selected from the global variable
    ;; 'server-standy-strategy'.
    apply-server-standby-strategy (servers with [rack = [id] of myself])

    ;; Initial service placement
    let submitted-services services-here with [status = "SUBMITTED"]
    if (count submitted-services) > 0
    [
      let the-server-set servers with [ rack = [id] of myself ]
      ask submitted-services
      [
        let candidate find-server the-server-set self
        ifelse candidate != nobody
        [
          set host ([who] of candidate)
          set status "SCHEDULED"
          set moving-speed ((distance-nowrap candidate) / 8)

          (ifelse
            method = "VM"
            [ set sys-accumulated-delay-from-vm sys-accumulated-delay-from-vm + service-method-vm-delay ]
            method = "CT"
            [ set sys-accumulated-delay-from-ct sys-accumulated-delay-from-ct + service-method-ct-delay ]
          )
        ]
        [ resubmit-service self ]
      ]
    ]
  ]

end
;;
;;
;;============================================================================;;
;; Each scheduler is responsible for controlling a number of servers and some
;; of the servers will be set in standby mode. When a server is in a standby
;; mode, it will be switched on and its status will be set to 'IDLE'. The
;; rationale behind the use of a standby strategy is that switching a server
;; from 'OFF' to 'ON' will incur a delay by the cold start process. To avoid
;; such a delay and to improve user experience, each rack should maintain a
;; number of 'IDLE' servers ready for service deployment. On the
;; other hand, although servers are in the 'IDLE' mode, they still consume
;; electricity. This somehow needs to be balanced between power consumption
;; and system response time.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-servers':
;; A list of servers belonging to the rack.
;;============================================================================;;
to apply-server-standby-strategy [ the-servers ]
  (ifelse
    server-standby-factor = 1 ;; All ON
    [ ask the-servers with [status = "OFF"] [ set status "IDLE" ] ]
    server-standby-factor = 0 ;; All OFF
    [ ask the-servers with [status = "IDLE"] [ set status "OFF" ] ]
    server-standby-factor < 1
    [
      let num-off-svrs count (the-servers with [status = "OFF"])
      let should-be-idle round (server-standby-factor * rack-space)
      let are-idle count (the-servers with [status = "IDLE"])
      let need-to-be-idle should-be-idle - are-idle

      (ifelse
        need-to-be-idle > 0 ;; Need to be IDLE
        [
          if need-to-be-idle > num-off-svrs [ set need-to-be-idle num-off-svrs ] ;; Not enought off servers left
          ask up-to-n-of need-to-be-idle the-servers with [status = "OFF"]
          [ set status "IDLE" set color blue reset-server self]

        ]
        need-to-be-idle < 0 ;; Need to be OFF
        [
          ask up-to-n-of (abs need-to-be-idle) the-servers with [status = "IDLE"]
          [ set status "OFF" set color white set power 0 reset-server self]
        ]
      )
    ]
  )

end
;;
;;
;;============================================================================;;
;; If no suitable servers can be found, resubmit the service, i.e., send it
;; back to the service pool to a random place and increate the global counter
;; 'sys-service-reschedule-counter' by 1.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-service':
;;  The service to be placed back in the service submission pool.
;;============================================================================;;
to resubmit-service [ the-service ]
  ask the-service
  [
    move-to one-of patches with [ pcolor = blue + 1 ]
    set attempt attempt + 1
    set sys-service-reschedule-counter sys-service-reschedule-counter + 1
    set status "OFFLINE"

    (ifelse
      service-submission-strategy = "closest"
      [ set host [who] of (min-one-of schedulers [distance myself]) ]
      service-submission-strategy = "resource-pattern-matching"
      [ set host (calc-resource-pattern self) ]
    )

    set moving-speed ((distance-nowrap scheduler host) / 8)
    set color yellow - attempt
    set label attempt
  ]

end
;;
;;
;;############################################################################;;
;; END: Scheduler Related
;;############################################################################;;
;;
;;
;;############################################################################;;
;; START: Placement Algorithms
;;############################################################################;;
;;
;;============================================================================;;
;; Find a server depending on the staged-evaluation selected.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : the set of servers that may be used for the deployment
;;                    of 'the-service'.
;; 'the-service'    : the service that is to be deployed.
;;============================================================================;;
to-report find-server [ the-server-set the-service ]
  (ifelse
    staged-evaluation = "1" [ report (one-staged-placement   the-server-set the-service) ]
    staged-evaluation = "2" [ report (two-staged-placement   the-server-set the-service) ]
    staged-evaluation = "3" [ report (three-staged-placement the-server-set the-service) ]
    [ report nobody ]
  )

end
;;
;;
;;============================================================================;;
;; When a service has arrived at a scheduler, the scheduler will try to
;; identify a server that is not in 'REPAIR'. It will select a server from
;; the 'the-server-set' with sufficient resources for the deployment of the
;; 'the-service'. The actual selection of the server is based on the
;; service placement algorithm specified in the 'service-placement-algorithm'
;; dropdown list. If no suitable servers can be found, the 'the-service' will
;; be sent back to the service submission pool by its corresponding scheduler.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report one-staged-placement [ the-server-set the-service ]
  let server-set the-server-set with [who != ([host] of the-service) and status != "REPAIR"]
  let candidate (find-candidate server-set the-service)

  ifelse candidate != nobody
  [
    reserve-server-resources candidate the-service
    report candidate
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; When a service has arrived at a scheduler, the scheduler will first try to
;; identify an active server, i.e., servers that are in the 'ON' or 'READY'
;; or 'IDLE' mode. It will select a server from the list of active servers with
;; sufficient resources for the deployment of the service. If no suitable
;; servers can be found, the scheduler will try with the 'OFF' servers.
;; The two-step process will improve the overall server utilization.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report two-staged-placement [ the-server-set the-service ]
  ;; Don't use servers with status = "OVERLOAD". Placing a service on an
  ;; overloaded server will incur performance degenerate.
  let server-set the-server-set with [ who != ([host] of the-service) and (status = "ON" or status = "READY" or status = "IDLE") ]
  let candidate find-candidate server-set the-service

  if candidate = nobody
  [
    set server-set the-server-set with [ who != ([host] of the-service) and status = "OFF" ]
    set candidate find-candidate server-set the-service
  ]

  ifelse candidate != nobody
  [
    reserve-server-resources candidate the-service
    report candidate
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; When a service has arrived at a scheduler, the scheduler will first try
;; with busy servers, i.e., servers that are in 'ON' or 'READY' mode. It
;; will then select a server from the busy servers with sufficient
;; resources for the deployment of the service. If no suitable
;; server can be found, the scheduler will try with the 'IDLE' servers, if
;; still unsuccessful, it will try with the 'OFF' servers.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report three-staged-placement [ the-server-set the-service ]
  let server-set the-server-set with [ who != ([host] of the-service) and (status = "ON" or status = "READY") ]
  let candidate find-candidate server-set the-service

  if candidate = nobody
  [
    set server-set the-server-set with [ who != ([host] of the-service) and status = "IDLE" ]
    set candidate find-candidate server-set the-service

    if candidate = nobody
    [
      set server-set the-server-set with [ who != ([host] of the-service) and status = "OFF" ]
      set candidate find-candidate server-set the-service
    ]
  ]

  ifelse candidate != nobody
  [
    reserve-server-resources candidate the-service
    report candidate
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; From the given server set, find a suitable server for the service
;; deployment based on the placement algorithm selected.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-candidate [ the-server-set the-service ]
  let candidate nobody
  (ifelse
    service-placement-algorithm = "random"
    [ set candidate find-random-server the-server-set the-service ]
    service-placement-algorithm = "first-fit"
    [ set candidate find-first-fit-server the-server-set the-service ]
    service-placement-algorithm = "balanced-fit"
    [ set candidate find-balanced-fit-server the-server-set the-service ]
    service-placement-algorithm = "max-utilization"
    [ set candidate find-max-utilization-server the-server-set the-service ]
    service-placement-algorithm = "min-power"
    [ set candidate find-min-power-server the-server-set the-service ]
    service-placement-algorithm = "AI"
    [
      ifelse ticks < 10
      [ set candidate find-first-fit-server the-server-set the-service ]
      [ set candidate find-AI-server the-server-set the-service ]

    ]
    service-placement-algorithm = "AI-phase1"
    [
      ifelse ticks < 10
      [ set candidate find-first-fit-server the-server-set the-service ]
      [ set candidate find-AI-server the-server-set the-service ]

    ]
    service-placement-algorithm = "AI-phase2"
    [
      ifelse ticks < 10
      [ set candidate find-first-fit-server the-server-set the-service ]
      [ set candidate find-AI-phase2-server the-server-set the-service ]

    ]
    service-placement-algorithm = "AI-phase3"
    [
      ifelse ticks < 10
      [ set candidate find-first-fit-server the-server-set the-service ]
      [ set candidate find-AI-phase3-server the-server-set the-service ]

    ]

  )

  report candidate

end




to-report find-AI-server [ the-server-set the-service ]
  ;; 总调用次数 +1
  set total-usage-count total-usage-count + 1

  let svrIDs [who] of servers
  let servers-data []
  let n min list 10 length svrIDs

  let i 0
  while [ i < n ] [
    let svr server (item i svrIDs)
    let sid [who] of svr
    let cpu-free-pct (1 - ([ops-now] of svr / [ops-phy] of svr)) * 100
    let ram-free-pct (1 - ([mem-now] of svr / [mem-phy] of svr)) * 100
    let net-free-pct (1 - ([net-now] of svr / [net-phy] of svr)) * 100
    set servers-data lput (list sid cpu-free-pct ram-free-pct net-free-pct) servers-data
    set i i + 1
  ]

  let ref-svr server (item 0 svrIDs)
  let cpu-req-pct ([ops-cnf] of the-service / [ops-phy] of ref-svr) * 100
  let ram-req-pct ([mem-cnf] of the-service / [mem-phy] of ref-svr) * 100
  let net-req-pct ([net-cnf] of the-service / [net-phy] of ref-svr) * 100
  let service-data (list cpu-req-pct ram-req-pct net-req-pct)

  py:set "servers_raw" servers-data
  py:set "service_raw" service-data
  let sid py:runresult "schedule_service(servers_raw, service_raw)"

  (ifelse
    sid >= 0 [
      set ai-usage-count ai-usage-count + 1
      report server sid
    ]
    sid = -2 [
      report nobody
    ]
    [
      report find-balanced-fit-server the-server-set the-service
    ]
  )
end


to-report find-AI-phase2-server [ the-server-set the-service ]
  report find-AI-python-server the-server-set the-service "schedule_service_phase2"
end


to-report find-AI-phase3-server [ the-server-set the-service ]
  report find-AI-python-server the-server-set the-service "schedule_service_phase3"
end


to-report find-AI-python-server [ the-server-set the-service python-scheduler-name ]
  ;; 总调用次数 +1
  set total-usage-count total-usage-count + 1

  let svrIDs [who] of servers
  let servers-data []
  let n min list 10 length svrIDs

  let i 0
  while [ i < n ] [
    let svr server (item i svrIDs)
    let sid [who] of svr
    let cpu-free-pct (1 - ([ops-now] of svr / [ops-phy] of svr)) * 100
    let ram-free-pct (1 - ([mem-now] of svr / [mem-phy] of svr)) * 100
    let net-free-pct (1 - ([net-now] of svr / [net-phy] of svr)) * 100
    set servers-data lput (list sid cpu-free-pct ram-free-pct net-free-pct) servers-data
    set i i + 1
  ]

  let ref-svr server (item 0 svrIDs)
  let cpu-req-pct ([ops-cnf] of the-service / [ops-phy] of ref-svr) * 100
  let ram-req-pct ([mem-cnf] of the-service / [mem-phy] of ref-svr) * 100
  let net-req-pct ([net-cnf] of the-service / [net-phy] of ref-svr) * 100
  let service-data (list cpu-req-pct ram-req-pct net-req-pct)

  py:set "servers_raw" servers-data
  py:set "service_raw" service-data
  let sid py:runresult (word python-scheduler-name "(servers_raw, service_raw)")

  (ifelse
    sid >= 0 [
      set ai-usage-count ai-usage-count + 1
      report server sid
    ]
    sid = -2 [
      report nobody
    ]
    [
      report find-balanced-fit-server the-server-set the-service
    ]
  )
end


to-report find-AI-server-legacy [ the-server-set the-service ]
  ;; 总调用次数 +1
  set total-usage-count total-usage-count + 1

  let svrIDs [who] of servers
  let server-results []  ; 存储服务器信息 [id cpu ram net]

  ;; 循环处理前10个服务器
  repeat 10 [
    let current-index (length server-results)
    let current-svr server (item current-index svrIDs)
    let current-id [who] of current-svr
    let current-cpu [ops-phy - ops-now] of current-svr
    let current-ram [mem-phy - mem-now] of current-svr
    let current-net [net-phy - net-now] of current-svr

    set server-results lput (list current-id current-cpu current-ram current-net) server-results
    show current-id
  ]

  ;; 服务请求部分
  let req one-of services
  let cpu-req [ops-cnf - ops-now] of req
  let ram-req [mem-cnf - mem-now] of req
  let net-req [net-cnf - net-now] of req

  ;; 调用 AI 代理
  let serverID py:runresult (word "chain.invoke({\"question\": \"if we have 10 servers with specs (Format: [id cpu ram net]" server-results " and the service(cpu" cpu-req ",ram" ram-req ",net" net-req "). Try to make full use of these 10 servers without overloading them, which server I should use to deploy the application. please return the server-id value only without anyother words.\"})")

  show (word "---------" serverID)

  ;; 根据 AI 返回结果判断
  ifelse (length serverID) = 1
  [
    let my_id (read-from-string serverID)
    ifelse my_id < 9 and my_id > 0
    [
      show "A"
      ;; 成功使用AI
      set ai-usage-count ai-usage-count + 1
      report server read-from-string serverID
    ]
    [
      show "B"
      ;; AI返回值无效，使用备选算法
      report find-balanced-fit-server the-server-set the-service
    ]
  ]
  [
    ;; AI返回值无效，使用备选算法
    show "B"
    report find-balanced-fit-server the-server-set the-service
  ]
end



;;
;;
;;============================================================================;;
;; When a service has arrived at a scheduler, the scheduler will select a
;; server from its rack, as long as the server has sufficient resources for
;; the deployment of the service.
;; The algorithm disregards the status of servers. It is not recommended but
;; only used for a simple comparison with other algorithms.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-random-server [ the-server-set the-service ]
  let candidate one-of the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  report candidate

end
;;
;;
;;============================================================================;;
;; When a service has arrived at a scheduler, the scheduler will firstly
;; sort all servers based on their Agent ID in an ascending order, then search
;; for a suitable server in the sored list in order. The same process repeats
;; for all the services and the search order is always from the smallest Agent
;; ID to the largest Agent ID, regardless of the status of the servers.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-first-fit-server [ the-server-set the-service ]
  let candidates the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  report (min-one-of candidates [ who ])

end
;;
;;
;;============================================================================;;
;; The objective of the algorithm is to find a suitable server for
;; 'the-service', so that the placement of the service will results in a
;; balanced resources across all types (CPU/MEM/NET) of the server.
;; As a greedy algorithm, it will not guarantee a global optimal
;; solution, however activities in a datacenter change overtime, finding
;; a global optimal solution for NOW in such a dynamic system will not ensure
;; the optimality of the solution LATER, thus such a greedy algorithm may be
;; preferred.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-balanced-fit-server [ the-server-set the-service ]
  let candidates the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  ifelse count candidates > 0
  [
    let svr-id -1
    let max-resource-distance 1
    ask candidates
    [
      let ops-ratio ((ops-now + ops-rsv + [ops-cnf] of the-service) / ops-phy)
      let mem-ratio ((mem-now + mem-rsv + [mem-cnf] of the-service) / mem-phy)
      let net-ratio ((net-now + net-rsv + [net-cnf] of the-service) / net-phy)

      let max-res-ratio max (list ops-ratio mem-ratio net-ratio)
      let min-res-ratio min (list ops-ratio mem-ratio net-ratio)
      let res-diff max-res-ratio - min-res-ratio
      if res-diff < max-resource-distance
      [
        set max-resource-distance res-diff
        set svr-id who
      ]
    ]

    report (server svr-id)
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; This procedure is used for all Max-Utilization-Fit family algorithms. The
;; objective of the algorihtm is to find a suitable server for 'the-service',
;; so that the placement of the service on the server will result in a balanced
;; resources across all types (cpu/mem/net), with an additional condition
;; that the residual resources of the server are minimized. As a greedy
;; algorithm, it will not guarantee a global optimal solution.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-max-utilization-server [ the-server-set the-service ]
  let candidates the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  ifelse count candidates > 0
  [
    let svr-id -1
    let min-resource-distance 3
    ask candidates
    [
      let ops-ratio ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy)
      let mem-ratio ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy)
      let net-ratio ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy)

      let sum-residual (reduce + (list (1 - ops-ratio) (1 - mem-ratio) (1 - net-ratio)))

      if sum-residual < min-resource-distance
      [
        set min-resource-distance sum-residual
        set svr-id who
      ]
    ]

    report (server svr-id)
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; Experimental, not used.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-max-utilization-with-resource-balancing-server [ the-server-set the-service ]
  let candidates the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  ifelse count candidates > 0
  [
    let svr-id -1
    let min-resource-distance 4
    ask candidates
    [
      let ops-ratio ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy)
      let mem-ratio ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy)
      let net-ratio ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy)

      let sum-residual (reduce + (list (1 - ops-ratio) (1 - mem-ratio) (1 - net-ratio)))

      let max-res-residual max (list (1 - ops-ratio) (1 - mem-ratio) (1 - net-ratio))
      let min-res-residual min (list (1 - ops-ratio) (1 - mem-ratio) (1 - net-ratio))

      let res-residual-diff max-res-residual - min-res-residual

      let res-diff sum-residual + res-residual-diff
      if res-diff < min-resource-distance
      [
        set min-resource-distance res-diff
        set svr-id who
      ]
    ]

    report (server svr-id)
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; This procedure is used for all 'Min-Power-Fit' family algorithms. The
;; objective of this procedure is to find a suitable server for 'the-service',
;; so that the placement of the service on the server will result in minimum
;; energy consumption increase. Note that each type of servers has its own
;; energy consumption model. The energy consumption of servers were modeled
;; based on the data collected from spec.org.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server-set' : a list of all possible candidate servers.
;; 'the-service'    : the service to be deployed.
;;============================================================================;;
to-report find-min-power-server [ the-server-set the-service ]
  let candidates the-server-set with
  [
    who != [host] of the-service and
    ((ops-now + ops-rsv + ([ops-cnf] of the-service)) / ops-phy) < server-cpu-overutil-threshold and
    ((mem-now + mem-rsv + ([mem-cnf] of the-service)) / mem-phy) < server-mem-overutil-threshold and
    ((net-now + net-rsv + ([net-cnf] of the-service)) / net-phy) < server-net-overutil-threshold
  ]

  ifelse count candidates > 0
  [
    let min-power 999999999
    let svr-id -1
    ask candidates
    [
      let sum-ops 0
      ifelse power-estimation-method = "configured"
      [ set sum-ops (([ops-cnf] of the-service) + ops-now) ]
      [ set sum-ops (get-resource-statistics ([ops-hist] of the-service) + ops-now) ]

      let potential-energy-consumption 0
      let now-energy-consumption 0
      (ifelse
        power-model-method = "stepwise-simple-linear-regression"
        [ ;; An in time calculation for better accuracy
          set now-energy-consumption calc-power-consumption-stepwise ops-now model
          set potential-energy-consumption calc-power-consumption-stepwise sum-ops model
        ]
        power-model-method = "simple-linear-regression"
        [
          set now-energy-consumption calc-power-consumption-simple ops-now model
          set potential-energy-consumption calc-power-consumption-simple sum-ops model
        ]
        power-model-method = "quadratic-polynomial"
        [
          set now-energy-consumption calc-power-consumption-quadratic ops-now model
          set potential-energy-consumption calc-power-consumption-quadratic sum-ops model
        ]
        power-model-method = "cubic-polynomial"
        [
          set now-energy-consumption calc-power-consumption-cubic ops-now model
          set potential-energy-consumption calc-power-consumption-cubic sum-ops model
        ]
      )

      let power-diff potential-energy-consumption - now-energy-consumption
      if power-diff < min-power
      [
        set min-power power-diff
        set svr-id who
      ]
    ]

    report (server svr-id)
  ]
  [ report nobody ]

end
;;
;;
;;============================================================================;;
;; Calculate statistics from the cached resource usage information.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-history' : a number of resource usage data points collected over time
;;  for each service or server.
;;============================================================================;;
to-report get-resource-statistics [ the-history ]
  (ifelse
    power-estimation-method = "max"
    [ report (max the-history) ]
    power-estimation-method = "mean"
    [ report (mean the-history) ]
    power-estimation-method = "median"
    [ report (median the-history) ]
    power-estimation-method = "linear-regression"
    [
      ifelse length the-history < 3
      [
        report (mean the-history)
      ]
      [
        ;; To minimize the library dependency, we leave this for later
        ;; major releases.
        report 0
      ]
    ]
  )

end
;;
;;
;;============================================================================;;
;; Once a service has been scheduled to run on a server, the scheduler will
;; first reserve the amounts of resources requested by the service, i.e., the
;; 'ops/mem/net-cnf', on the server. The resource reservation is necessary
;; and vitally important. Since services using different deployment methods
;; move to servers at a different speed, plus services are generated at different
;; time, it is possible that lately scheduled services may arrive at the server
;; sooner than the services that were scheduled earlier. To ensure the server
;; has sufficient resources for services arriving at different speeds, all
;; reservations must be done at the scheduling time.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-server'  : resources for 'the-service' to be reserved on 'the-server'.
;; 'the-service' : the service contains the information about the resources to
;;  be reserved on the server.
;;============================================================================;;
to reserve-server-resources [ the-server the-service ]
  ask the-server
  [
    if status = "OFF"
    [
      ask the-service
      [ set delay-counter delay-counter + server-cold-start-delay ]
      set sys-accumulated-delay-from-svr-cold-start sys-accumulated-delay-from-svr-cold-start + server-cold-start-delay
    ]

    ;; Setting a server in 'READY' mode will prevent the server from being
    ;; switched off by the 'server standby stragegy' (controlled by schedulers).
    if status = "IDLE" or status = "OFF"
    [ set status "READY" set color blue set power base-power ]

    set ops-rsv ops-rsv + [ops-cnf] of the-service
    set mem-rsv mem-rsv + [mem-cnf] of the-service
    set net-rsv net-rsv + [net-cnf] of the-service
  ]
end
;;
;;
;;############################################################################;;
;; END: Placement Algorithms
;;############################################################################;;
;;
;;
;;############################################################################;;
;; START: Server Consolidation
;;############################################################################;;
;;
;;============================================================================;;
;; Consolidating servers in a datacenter to minimize the total power
;; consumption and to maximize the overall resource utilization.
;; The main mechanism used for server consolidation is live-migration. In
;; general, a live-migration can be performed easily for virtual machines or
;; containers. In this simulator, live-migrations are allowed for all virtual
;; machine, container, and bare-metal deployment of services.
;; Additionally, from a technical point of view, live-migration may only be
;; possible when processor architectures of the hosting server and
;; targeting server are compatible. However, in this version of the simulator,
;; migrations between heterogeneous processor architectures are allowed.
;; NOTE: migration of services shall start from over-utlized servers first,
;; then followed by under-utilized servers, due to the following reasons:
;;  1. it is more important to maintain servers' performance than
;;     saving energy or maximizing resource utilization;
;;  2. after migrating services off of the over-utilized servers, the number
;;     of under-utilized servers could be reduced, thus avoid unnecessary
;;     calculations. Note that service migraitons are expensive operations,
;;     in terms of service performance degeneration and large network bandwidth
;;     consumption.
;;--PARAMETERS----------------------------------------------------------------;;
;;  none
;;============================================================================;;
to consolidate-servers
  if server-consolidation-strategy = "within-datacenter" or server-consolidation-strategy = "within-rack"
  [ consolidate-underutilized-servers ]

end
;;
;;
;;============================================================================;;
;; Migrating services out of under-utilized servers. A constituent
;; under-utilzed server must exhibit the following characteristics:
;;  1. the server must not be in 'OFF', 'IDLE', 'READY', or 'OVERLOAD' mode
;;  2. resource utilization of the server must under the threshold
;;     specified by the global variable
;;     'server-cpu/mem/net-utilization-threshold'
;;  3. all services on the under-utilized servers must be in the 'RUNNING'
;;     mode, i.e., if there is any service that is undergo a migration
;;     process, the server will not be considered as an under-utilized server.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to consolidate-underutilized-servers
  let under-util-svrs servers with
  [
    status = "ON" and (all? services-here [status = "RUNNING"]) and
    ((ops-now + ops-rsv) / ops-phy) <= server-cpu-underutil-threshold and
    ((mem-now + mem-rsv) / mem-phy) <= server-mem-underutil-threshold and
    ((net-now + net-rsv) / net-phy) <= server-net-underutil-threshold
  ]

  if (count under-util-svrs) > 0
  [
    foreach (list under-util-svrs)
    [
      svr -> ask svr
      [ ;; Special attention must be paid here. When migrating services out of a
        ;; server, the services could be re-placed on a server that was originally
        ;; identified as an under-utilized server, however, the placement might
        ;; make the server a 'normal' server, which should not be considered as a
        ;; candidate server for consolidation anymore, thus the re-check.
        if status = "ON" and (all? services-here [status = "RUNNING"]) and
        ((ops-now + ops-rsv) / ops-phy) <= server-cpu-overutil-threshold and
        ((mem-now + mem-rsv) / mem-phy) <= server-mem-overutil-threshold and
        ((net-now + net-rsv) / net-phy) <= server-net-overutil-threshold
        [
          ;; For each service here, find a sutiable target server.
          let migr-list []
          let service-count count services-here
          let keep-searching? true
          ask services-here
          [ ;; Identify a suitable server
            if keep-searching?
            [
              let the-server-set []
              (ifelse
                server-consolidation-strategy = "within-datacenter"
                [
                  set the-server-set servers with
                  [ ;; The 'ops/mem/net-cnf' could be changed to other values.
                    who != ([host] of myself) and
                    (any? services-here with [status = "RUNNING"]) and
                    ((ops-now + ops-rsv + [ops-cnf] of myself) / ops-phy) < server-cpu-overutil-threshold and
                    ((mem-now + mem-rsv + [mem-cnf] of myself) / mem-phy) < server-mem-overutil-threshold and
                    ((net-now + net-rsv + [net-cnf] of myself) / net-phy) < server-net-overutil-threshold
                  ]
                ]
                server-consolidation-strategy = "within-rack"
                [
                  set the-server-set servers with
                  [
                    who != ([host] of myself) and rack = ([rack] of svr) and
                    (any? services-here with [status = "RUNNING"]) and
                    ((ops-now + ops-rsv + [ops-cnf] of myself) / ops-phy) < server-cpu-overutil-threshold and
                    ((mem-now + mem-rsv + [mem-cnf] of myself) / mem-phy) < server-mem-overutil-threshold and
                    ((net-now + net-rsv + [net-cnf] of myself) / net-phy) < server-net-overutil-threshold
                  ]
                ]
              )
              ;; If no suitable server could be found for this service,
              ;; terminate the entire process for the server. Otherwise,
              ;; reserve resources on the server and mark the service for
              ;; batch migration.
              ifelse (count the-server-set) > 0
              [ ;; Reserve resources on the server.
                let candidate find-server the-server-set self
                ;; Add it to the migr-list
                set migr-list lput (list who ([who] of candidate)) migr-list
                set service-count service-count - 1
              ]
              [ set keep-searching? false ]
            ]
          ]

          ;; If not all the services could find a target server, the entire
          ;; process needs to be rolled back, otherwise, trigger the migration
          ;; process.
          ifelse service-count = 0
          [ ;; Migrate services
            foreach migr-list
            [ x ->
              ;;show (word "migr-list: " (first x) " " (last x))
              migrate-service (first x) (last x)
            ]
            set sys-current-migration-event-due-to-consolidation (length migr-list)
            set sys-accumulated-migration-event-due-to-consolidation sys-accumulated-migration-event-due-to-consolidation + sys-current-migration-event-due-to-consolidation
          ]
          [ ;; Roll back
            foreach migr-list
            [
              x -> ask server (last x)
              [
                set ops-rsv ops-rsv - ([ops-cnf] of service (first x))
                set mem-rsv mem-rsv - ([mem-cnf] of service (first x))
                set net-rsv net-rsv - ([net-cnf] of service (first x))

                ;; When reserving resources on 'IDLE' or 'OFF' servers, the status
                ;; of the server will be changed to 'READY', thus the rolling back
                ;; process here is needed.
                if (ops-rsv = 0 and mem-rsv = 0 and net-rsv = 0 and ops-now = 0 and mem-now = 0 and net-now = 0)
                [ set status "IDLE" ] ;; When reserving resources on 'IDLE' or 'OFF' servers, the status of the server will
              ]
            ]
          ]
        ]
      ]
    ]
  ]
end
;;
;;
;;============================================================================;;
;; Calculating service migration related metrics including time,
;; required bandwidth, and migration speed for visualization. This will be
;; based on the current available network bandwidth between the source and destination, and the
;; destination, and the size of the busy memory, which can be jointly
;; determined by the size of the currently used memory (i.e., mem-now) and
;; the memory access rate specified by the service attribute 'access-ratio'.
;; The is also the life time for the replicas of migrants.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-service'  : the migranting service.
;;  'the-dest-svr' : the destination server for the migranting service.
;;============================================================================;;
to-report calc-migr-metrics [ the-service the-dest-svr ]
  let mem-footprint ([mem-now * access-ratio] of the-service)
  let hosting-svr server ([host] of the-service)
  let available-bw-current-svr ([net-phy - net-now - net-rsv] of hosting-svr)
  let available-bw-dest-svr ([net-phy - net-now - net-rsv] of the-dest-svr)
  let max-bw-for-the-service ([net-cnf] of the-service)

  ;; This the maximum possible network bandwidth that can be used for the
  ;; migration of the service.
  let p-to-p-bw min (list available-bw-current-svr available-bw-dest-svr max-bw-for-the-service)
  if p-to-p-bw <= 10 [ set p-to-p-bw 10 ]
  ;; The time is in second, but converted to 'simulation-time-unit'.
  ;; The time is also the life-time of the replicas of the migrating
  ;; service.
  let time-needed ((mem-footprint / p-to-p-bw) / (simulation-time-unit * 60))

  let migr-speed 1

  if show-migr-move?
  [
    let dist-to-dest (distancexy [xcor] of the-dest-svr [ycor] of the-dest-svr)
    ;; show (word "src-dest svrs distance: " dist-to-dest)

    ;; A simplified sigmod function for computational efficiency
    let a-factor dist-to-dest * time-needed
    ifelse a-factor > 5
    [ set migr-speed 1.0 + 0.4 ]
    [ set migr-speed (1 / (1 + (exp a-factor)) + 0.4)]
  ]

  report (list time-needed p-to-p-bw migr-speed)

end
;;
;;
;;============================================================================;;
;; Migrating a single service to its destination server.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-service-id'  : the agent ID of the migrant.
;;  'the-dest-svr-id' : the agent ID of the desination server.
;;============================================================================;;
to migrate-service [ the-service-id the-dest-svr-id ]
  let the-service (service the-service-id)
  let the-dest-svr (server the-dest-svr-id)
  let the-host-svr (server ([host] of the-service))
  let migr-metrics calc-migr-metrics the-service the-dest-svr
  ;; show (word "migr metrics: " migr-metrics)

  ifelse
  (item 0 migr-metrics) <= simulation-time-unit
  [
    ;; Calculate the energy consumed by the replica on the current hosting server.
    let ops-t1 ([ops-now] of the-host-svr)
    let ops-t2 (([ops-now] of the-service) + ([ops-now] of the-host-svr))
    calc-power-with-fraction-time-unit ops-t1 ops-t2 ([model] of the-host-svr)
  ]
  [
    ask the-service
    [ ;; This hatched service will inherits all the property of 'the-service'.
      ;; It stays on the current hosting server until dies.
      hatch-services 1
      [
        set xcor ([xcor] of the-host-svr)
        set ycor ([ycor] of the-host-svr)
        set life-time (item 0 migr-metrics)
        set status "MIGRATING" ;; indicating this is a service moving out of the server.
        set net-now (item 1 migr-metrics) ;; This is the network bandwidth used to migrate the service
      ]
    ]
  ]

  ;; Move the service to the destination server
  ask the-service
  [
    if show-migr-move?
    [
      hatch-vis-agents 1
      [
        set color cyan
        set xcor ([xcor] of myself)
        set ycor ([ycor] of myself)
        set shape "circle 2"
        set color orange
        set size 1
        set to-svr the-dest-svr-id
        set from-svr [who] of the-host-svr
        set moving-speed (item 2 migr-metrics)
      ]
    ]

    move-to the-dest-svr
    set host ([who] of the-dest-svr)
    set status "DEPLOYED" ;; This ensures the server frees the reserved resources for the service.
  ]

end
;;
;;
;;============================================================================;;
;; When a service has less than one 'simulation-time-unit', the calculation
;; of the energy consumption for the service will be treated specially.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-ops-t1'   : the 'ops-now' of the server without the service.
;;  'the-ops-t2'   : the 'ops-now of the server with the service.
;;  'the-svr-model': the model of the server.
;;============================================================================;;
to calc-power-with-fraction-time-unit [ the-ops-t1 the-ops-t2 the-svr-model ]
  let power-prev calc-energy-consumption the-ops-t1 the-svr-model
  let power-post calc-energy-consumption the-ops-t2 the-svr-model
  let diff power-post - power-prev
  ;; show (word "Energy consumed by migrant replicas with a fraction of lift-time: " diff)
  set sys-power-consumption-total sys-power-consumption-total + ((diff * simulation-time-unit) / (60 * 1000))

end
;;
;;
;;============================================================================;;
;; Migrating services out of the over-utilized servers. A constituent
;; over-utilized server is a server on which any of the CPU/MEM/NET resource
;;  utilization is above a threshold specified by the global variable
;; 'server-cpu/mem/net-utilization-threshold'. In addition, the server must
;; not have any reserved resources, i.e., ops/mem/net-rsv. The reason behind
;; this is that reserved resources are for those (1) services that are
;; migrating to other servers, or (2) newly scheduled services that are on
;; their way to the server. Additionally, no services running on the server
;; should have 'MIGR-OUT' status.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-server' : a server that has status = 'OVERLOAD'.
;;============================================================================;;
to auto-migrate [ the-server ]
  ask the-server
  [
    let current-svr-ress (list ops-now mem-now net-now)
    let svr-phy-ress (list ops-phy mem-phy net-phy)

    ;; We should not consider services with status = 'MIGRATING',
    ;; as they are flowing out or in.
    let running-services services-here with [ status = "RUNNING" ]
    let sorted-services []
    (ifelse
      auto-migration-strategy = "least-migration-time"
      [ set sorted-services (sort-on [mem-now * access-ratio]  running-services) ]
      auto-migration-strategy = "least-migration-number"
      [
        let ops-ratio ops-now / ops-phy
        let net-ratio net-now / net-phy
        let mem-ratio mem-now / mem-phy
        let max-res (max (list ops-ratio net-ratio mem-ratio))
        (ifelse ;; we consider the mem the last
          ops-ratio = max-res [ set sorted-services (sort-on [ops-now] running-services) ]
          net-ratio = max-res [ set sorted-services (sort-on [net-now] running-services) ]
          mem-ratio = max-res [ set sorted-services (sort-on [mem-now] running-services) ]
        )
      ]
    )

    let utilization-thresholds (list server-cpu-overutil-threshold server-mem-overutil-threshold server-net-overutil-threshold)
    let continue? true
    let event-counter 0
    foreach sorted-services
    [
      x ->
      if continue?
      [
        let service-res-list (list ([ops-now] of x) ([mem-now] of x) ([net-now] of x))
        let now-level (map / current-svr-ress svr-phy-ress)
        let now-cmp (map < now-level utilization-thresholds) ;; pay attention to this
        ifelse (not (reduce and now-cmp))
        [
          let the-server-set []

          set the-server-set servers with
          [
            who != ([host] of x) and
            ((ops-now + ops-rsv + [ops-cnf] of x) / ops-phy) < server-cpu-overutil-threshold and
            ((mem-now + mem-rsv + [mem-cnf] of x) / mem-phy) < server-mem-overutil-threshold and
            ((net-now + net-rsv + [net-cnf] of x) / net-phy) < server-net-overutil-threshold
          ]

          if server-consolidation-strategy = "within-rack"
          [
            let rack-id [rack] of (server ([host] of x))
            set the-server-set the-server-set with [rack = rack-id]
          ]

          let candidate find-server the-server-set x
          if candidate != nobody
          [
            set current-svr-ress (map - current-svr-ress service-res-list)
            migrate-service [who] of x [who] of candidate
            set event-counter event-counter + 1
          ]
        ]
        [ set continue? false ]
      ]
    ]
    set sys-current-migration-event-due-to-auto-migration event-counter
    set sys-accumulated-migration-event-due-to-auto-migration sys-accumulated-migration-event-due-to-auto-migration + event-counter
  ]

end
;;
;;
;;############################################################################;;
;; END: Server Consolidation
;;############################################################################;;
;;
;;
;;############################################################################;;
;; START: Utility Functions
;;############################################################################;;
;;
;;============================================================================;;
;; When services requiring more resources but their requests can't be satisfied
;; by the underlying hosting server, the performance of the services will be
;; affected. In this case, all services on the server will be penalized based
;; on how much total excessive resources were requested.
;;--PARAMETERS----------------------------------------------------------------;;
;; 'the-res-diff'  : this input is a list that contains the differences
;;                   between the currently used resources (ops/mem/net-now)
;;                   and the previous resource usage (ops/mem/net-prev) of
;;                   each running service on the server.
;;  'the-ext-unit' : it is a normalized amounts of units that have exceeded
;;                   the available physical resources of the server. It is
;;                   calculated as follows: (sum(ops/mem/net) of all running
;;                   services - the resources occupied by the services who
;;                   are migrating out of the server - the total available
;;                   resources of the server) divided by the total available
;;                   resources of the server.
;; 'the-res-min'   : this input indicates the minimum value of 'the-res-diff'.
;;============================================================================;;
to-report cal-sla-dist-factors [ the-service-res-delta the-res-min ]
  let service-res-delta-prime []
  let sum-delta-prime 0
  foreach the-service-res-delta
  [
    x ->
    let delta-prime (10 + (abs the-res-min) + last x)
    set service-res-delta-prime lput (list (first x) delta-prime) service-res-delta-prime
    set sum-delta-prime sum-delta-prime + delta-prime
  ]

  let res-dist-factors []
  foreach service-res-delta-prime
  [
    x ->
    set res-dist-factors lput (list (first x) ((last x) / sum-delta-prime)) res-dist-factors
  ]

  report res-dist-factors

end



to-report apply-penalty [ the-res-diff the-ext-unit the-res-min ]
  let res-diff-scale []
  let sum-diff 0
  foreach the-res-diff
  [
    x ->
    let scaler (the-res-min + last x)
    set res-diff-scale lput (list (first x) scaler) res-diff-scale
    set sum-diff sum-diff + scaler
  ]

  let res-diff-scaled-unit []
  foreach res-diff-scale [ x -> set res-diff-scaled-unit lput (list (first x) ((last x) * the-ext-unit / sum-diff)) res-diff-scaled-unit ]

  report res-diff-scaled-unit

end
;;
;;
;;============================================================================;;
;; Each service is assigned to one of the schedulers, conditioning on the
;; configuration set out for the datacenter.
;;
;; Scenario 1:
;; The datacenter contains heterogeneous resources/configurations, specifed
;; by the global variable 'datacenter-level-heterogeneity? = ON', each
;; rack also has mixed types of servers specified by the global variable
;; 'rack-level-heterogeneity? = OFF', and the 'resource-pattern-matching'
;; strategy is selected from the 'service-submission-strategy' dropdown list,
;; then the service will be assigned to a scheduler with the best matching
;; scores. The two tuples are then the required resources of the
;; service and the available resource of the rack, respectively.
;;
;; Scenario 2:
;; The datacenter contains heterogeneous resources/configurations specified by
;; the global variable 'datacenter-level-heterogeneity? = ON', each rack has
;; only one type of servers specified by 'rack-level-heterogeneity? = OFF',
;; and the 'resource-pattern-matching' strategy is selected from the
;; 'service-submission-strategy', then the service will be assigned to a
;; scheduler with the best matching scores. The two tuples are then
;; the required resources of the service and the configured resources of one
;; of the servers in the rack, respectively.
;;
;; Scenario 3:
;; The datacenter contains homogeneous resources/configurations specified by
;; 'datacenter-level-heterogeneity? = OFF'. In this case the
;; 'rack-level-heterogeneity?'and the 'service-submission-strategy' will not
;; be evaluated. The service to scheduler assignment will fallback to the
;; 'closest' strategy, i.e., send the service to its closest scheduler measured
;; by the distance between the service and a scheduler.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-service' : the service to be assigned to one of the available
;;                  schedulers.
;;============================================================================;;
to-report calc-resource-pattern [ the-service ]
  let suggested-candidate-id -1
  (ifelse not datacenter-level-heterogeneity?
    [ set suggested-candidate-id [who] of (min-one-of schedulers [distance the-service]) ]
    [
      let min-score 1 ;; The smaller, the better
      ask schedulers
      [
        let rack-id id
        let matching-score 1
        ifelse rack-level-heterogeneity?
        [
          set matching-score calc-resource-pattern-matching-score ;; The function
          (list ([ops-cnf] of the-service) ([mem-cnf] of the-service) ([net-cnf] of the-service)) ;; The first input to the function
          calc-servers-available-resources (servers with [ rack = rack-id ]) ;; The second input to the function
        ]
        [
          let any-svr one-of servers with [ rack = rack-id ]
          set matching-score calc-resource-pattern-matching-score ;; The funciton
          (list ([ops-cnf] of the-service) ([mem-cnf] of the-service) ([net-cnf] of the-service)) ;; The first input to the function
          (list ([ops-phy] of any-svr) ([mem-phy] of any-svr) ([net-phy] of any-svr)) ;; The second input to the function
        ]

        if matching-score < min-score
        [
          set min-score matching-score
          set suggested-candidate-id who
        ]
      ]
    ]
  )

  report suggested-candidate-id

end
;;
;;
;;============================================================================;;
;; Calculate the current total available resources of a given rack.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-server-set' : a set of servers in a given rack.
;;============================================================================;;
to-report calc-servers-available-resources [ the-server-set ]
  let sum-ops 0
  let sum-mem 0
  let sum-net 0
  ask the-server-set
  [
    set sum-ops (sum-ops + (ops-phy - ops-now - ops-rsv))
    set sum-mem (sum-mem + (mem-phy - mem-now - mem-rsv))
    set sum-net (sum-net + (net-phy - net-now - net-rsv))
  ]

  report (list sum-ops sum-mem sum-net)

end
;;
;;
;;============================================================================;;
;; Reset server status when setting it to 'IDLE' or 'OFF' or first created.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-server' : a server.
;;============================================================================;;
to reset-server [ the-server ]
  ask the-server
  [
    set ops-now 0
    set mem-now 0
    set net-now 0

    set ops-rsv 0
    set mem-rsv 0
    set net-rsv 0

    set ops-hist 0
    set mem-hist 0
    set net-hist 0
  ]

end
;;
;;
;;============================================================================;;
;; This procedure quantifies the difference between different types of
;; resources with different sizes and scales.
;; Given two resource tuples 'the-service--ress (CPU, MEM, NET) and
;; the total available resources of the rack (aCPU, aMEM, aNET), the pattern
;; matching score is calculated as follows:
;; max(CPU/aCPU, MEM/aMEM, NET/aNET) - min(CPU/aCPU, MEM/aMEM, NET/aNET')
;; The rationale behind this method is that if the difference between the
;; ratio of different types of resources are close to each other, it will
;; indicate the resources of different types are used more evenly.
;; The procedure is mainly used to find a suitable scheduler for a newly
;; generated service.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-service-ress' : a list that contains the required resources of the
;;                       service.
;;  'the-rack-ress'    : a list that contains the aggregated available
;;                       resources of the rack.
;;============================================================================;;
to-report calc-resource-pattern-matching-score [ the-service-ress the-rack-ress ]
  let norm-tuple (map / the-service-ress the-rack-ress)

  report (max norm-tuple - min norm-tuple)

end
;;
;;
;;============================================================================;;
;; At each 'tick', each server will update its power consumption. The power
;; consumed by a server is determined by the total 'ops' consumed at the
;; moment, i.e., the 'ops-now'.
;; Depending on the manufacture and the model, each server may have a different
;; energy consumption pattern.
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-workload' : the total computing power used at the moment.
;;  'the-model'    : the server model.
;;============================================================================;;
to-report calc-power-consumption-stepwise [ the-workload the-model ]
  ;; Benchmark information were collected from spec.org
  let power-consumption 0
  (ifelse
    the-model = 1 ;; HP Enterprise ProLiant DL110 Gen10 Plus
    [
      (ifelse
        ;; CPU Utilization 100% ~ 90%
        the-workload >= 2996610 [ set power-consumption (0.00006841  * the-workload + 72.001)   ]
        ;; CPU Utilization 90% ~ 80%
        the-workload >= 2663920 [ set power-consumption (0.00014127  * the-workload - 146.3391) ]
        ;; CPU Utilization 80% ~ 70%
        the-workload >= 2323252 [ set power-consumption (0.00010861  * the-workload - 59.3287)  ]
        ;; CPU Utilization 70% ~ 60%
        the-workload >= 1991168 [ set power-consumption (0.000066248 * the-workload + 39.0885)  ]
        ;; CPU Utilization 60% ~ 50%
        the-workload >= 1662976 [ set power-consumption (0.000051799 * the-workload + 67.8596)  ]
        ;; CPU Utilization 50% ~ 40%
        the-workload >= 1330630 [ set power-consumption (0.000039116 * the-workload + 88.9513)  ]
        ;; CPU Utilization 40% ~ 30%
        the-workload >= 997346  [ set power-consumption (0.00003601  * the-workload + 93.0902)  ]
        ;; CPU Utilization 30% ~ 20%
        the-workload >= 668831  [ set power-consumption (0.000039572 * the-workload + 89.533 )  ]
        ;; CPU Utilization 20% ~ 10%
        the-workload >= 331877  [ set power-consumption (0.000038581 * the-workload + 90.1959 ) ]
        ;; CPU Utilization 10% ~ 0%
        the-workload >= 0       [ set power-consumption (0.000026817 * the-workload + 94.1)     ]
      )
    ]
    the-model = 2 ;; Lenovo Global Technology ThinkSystem SR655
    [
      (ifelse
        the-workload >= 5520918 [ set power-consumption (0.000053905 * the-workload - 101.6066) ]
        the-workload >= 4875441 [ set power-consumption (0.000037182 * the-workload - 9.2777)   ]
        the-workload >= 4287820 [ set power-consumption (0.000020421 * the-workload + 72.437)   ]
        the-workload >= 3672211 [ set power-consumption (0.000016244 * the-workload + 90.3483)  ]
        the-workload >= 3069128 [ set power-consumption (0.000016581 * the-workload + 89.1094)  ]
        the-workload >= 2453590 [ set power-consumption (0.000016246 * the-workload + 90.1391)  ]
        the-workload >= 1839277 [ set power-consumption (0.000016278 * the-workload + 90.0596)  ]
        the-workload >= 1229720 [ set power-consumption (0.000018046 * the-workload + 86.8086)  ]
        the-workload >= 613931  [ set power-consumption (0.000017538 * the-workload + 87.4326)  ]
        the-workload >= 0       [ set power-consumption (0.000073624 * the-workload + 53.0)     ]
      )
    ]
    the-model = 3 ;; Fujitsu PRIMERGY RX2530 M6
    [
      (ifelse
        the-workload >= 6814856 [ set power-consumption (0.00010672  * the-workload - 228.2503) ]
        the-workload >= 6047198 [ set power-consumption (0.000071646 * the-workload + 10.7395)  ]
        the-workload >= 5292424 [ set power-consumption (0.000084794 * the-workload - 68.7637)  ]
        the-workload >= 4535904 [ set power-consumption (0.000058161 * the-workload + 72.187)   ]
        the-workload >= 3781094 [ set power-consumption (0.000045044 * the-workload + 131.6827) ]
        the-workload >= 3031768 [ set power-consumption (0.000037367 * the-workload + 160.7122) ]
        the-workload >= 2279204 [ set power-consumption (0.000034549 * the-workload + 169.2568) ]
        the-workload >= 1516364 [ set power-consumption (0.000035394 * the-workload + 167.3297) ]
        the-workload >= 758504  [ set power-consumption (0.000035627 * the-workload + 166.9771) ]
        the-workload >= 0       [ set power-consumption (0.000083058 * the-workload + 131.0)    ]
      )
    ]
    the-model = 4 ;; New H3C Technologies Co. Ltd. H3C UniServer R4900 G5
    [
      (ifelse
        the-workload >= 7329609 [ set power-consumption (0.000084383 * the-workload - 77.4948)  ]
        the-workload >= 6523038 [ set power-consumption (0.0001463   * the-workload - 531.3096) ]
        the-workload >= 5705607 [ set power-consumption (0.000070954 * the-workload - 39.8356)  ]
        the-workload >= 4893456 [ set power-consumption (0.000049252 * the-workload + 83.9879)  ]
        the-workload >= 4075795 [ set power-consumption (0.000042805 * the-workload + 115.5355) ]
        the-workload >= 3255610 [ set power-consumption (0.000028042 * the-workload + 175.7047) ]
        the-workload >= 2445686 [ set power-consumption (0.000030867 * the-workload + 166.5088) ]
        the-workload >= 1634538 [ set power-consumption (0.000033286 * the-workload + 160.5925) ]
        the-workload >= 813621  [ set power-consumption (0.000038981 * the-workload + 151.2844) ]
        the-workload >= 0       [ set power-consumption (0.000078661 * the-workload + 119.0)    ]
      )
    ]
    the-model = 5 ;; Inspur Corporation Inspur NF8480M6
    [
      (ifelse
        the-workload >= 10444641 [ set power-consumption (0.0001007   * the-workload - 155.7673) ]
        the-workload >= 9274428  [ set power-consumption (0.00010425  * the-workload - 192.9011) ]
        the-workload >= 8127807  [ set power-consumption (0.00012907  * the-workload - 423.0959) ]
        the-workload >= 6973093  [ set power-consumption (0.00008487  * the-workload - 63.8029)  ]
        the-workload >= 5799377  [ set power-consumption (0.000065604 * the-workload + 70.54)    ]
        the-workload >= 4643659  [ set power-consumption (0.000058838 * the-workload + 109.7769) ]
        the-workload >= 3480205  [ set power-consumption (0.000045554 * the-workload + 171.4627) ]
        the-workload >= 2317053  [ set power-consumption (0.000039548 * the-workload + 192.3659) ]
        the-workload >= 1159866  [ set power-consumption (0.000038887 * the-workload + 193.8958) ]
        the-workload >= 0        [ set power-consumption (0.00011639  * the-workload + 104.0)    ]
      )
    ]
    the-model = 6 ;; Dell Inc. PowerEdge R6515
    [
      (ifelse
        the-workload >= 5527289 [ set power-consumption (0.000021775 * the-workload + 97.6435)  ]
        the-workload >= 4907530 [ set power-consumption (0.000017749 * the-workload + 119.8971) ]
        the-workload >= 4289422 [ set power-consumption (0.000021032 * the-workload + 103.7852) ]
        the-workload >= 3681028 [ set power-consumption (0.000021368 * the-workload + 102.3448) ]
        the-workload >= 3056053 [ set power-consumption (0.000019201 * the-workload + 110.3214) ]
        the-workload >= 2459905 [ set power-consumption (0.000013419 * the-workload + 127.9893) ]
        the-workload >= 1840829 [ set power-consumption (0.000017768 * the-workload + 117.2914) ]
        the-workload >= 1224553 [ set power-consumption (0.000017849 * the-workload + 117.1428) ]
        the-workload >= 613770  [ set power-consumption (0.000026196 * the-workload + 106.9218) ]
        the-workload >= 0       [ set power-consumption (0.00011161  * the-workload + 54.5)     ]
      )
    ]
    the-model = 7 ;; LSDtech L224S-D/F/V-1
    [
      (ifelse
        the-workload >= 2073591 [ set power-consumption (0.00011173  * the-workload + 175.317)  ]
        the-workload >= 1841179 [ set power-consumption (0.00013769  * the-workload + 121.4945) ]
        the-workload >= 1612999 [ set power-consumption (0.00018845  * the-workload + 28.034)   ]
        the-workload >= 1380996 [ set power-consumption (0.00015517  * the-workload + 81.7103)  ]
        the-workload >= 1154911 [ set power-consumption (0.00012827  * the-workload + 118.8592) ]
        the-workload >= 924044  [ set power-consumption (0.00011262  * the-workload + 136.9351) ]
        the-workload >= 692791  [ set power-consumption (0.00010811  * the-workload + 141.1046) ]
        the-workload >= 462327  [ set power-consumption (0.000091121 * the-workload + 152.8725) ]
        the-workload >= 229125  [ set power-consumption (0.000077186 * the-workload + 159.3147) ]
        the-workload >= 0       [ set power-consumption (0.00020076  * the-workload + 131.0)    ]
      )
    ]
    the-model = 8 ;; ASUSTeK Computer Inc. RS700A-E9-RS4V2
    [
      (ifelse
        the-workload >= 10583175 [ set power-consumption (0.000026811 * the-workload + 116.2591) ]
        the-workload >= 9385781  [ set power-consumption (0.000023384 * the-workload + 152.5218) ]
        the-workload >= 8227893  [ set power-consumption (0.000024182 * the-workload + 145.0334) ]
        the-workload >= 7055617  [ set power-consumption (0.00002815  * the-workload + 112.3818) ]
        the-workload >= 5878359  [ set power-consumption (0.00001444  * the-workload + 209.1145) ]
        the-workload >= 4698054  [ set power-consumption (0.000020334 * the-workload + 174.4711) ]
        the-workload >= 3528516  [ set power-consumption (0.000022231 * the-workload + 165.5576) ]
        the-workload >= 2352022  [ set power-consumption (0.00001445  * the-workload + 193.014)  ]
        the-workload >= 1176164  [ set power-consumption (0.000025513 * the-workload + 166.9922) ]
        the-workload >= 0        [ set power-consumption (0.00007737  * the-workload + 106.0)    ]
      )
    ]
    [ ;; Random Servers. Power consumption is assumed to increase with workloads linearly.
      ;; The model is based on Dell PowerEdge R6515 server using simple linear regression.
      if the-workload >= 0       [ set power-consumption (0.000023346 * the-workload + 94.5591)  ]
    ]
  )

  report power-consumption

end
;;
;;
;;============================================================================;;
;; Simple Linear Regression Model
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-workload' : the total computing power used at the moment.
;;  'the-model'    : the server model.
;;============================================================================;;
to-report calc-power-consumption-simple [ the-workload the-model ]
  ;; Benchmark information were collected from spec.org
  let power-consumption 0
  (ifelse
    the-model = 1 ;; HP Enterprise ProLiant DL110 Gen10 Plus
    [ set power-consumption (72.37 + 0.00006076 * the-workload) ]

    the-model = 2 ;; Lenovo Global Technology ThinkSystem SR655
    [ set power-consumption (70.66 + 0.00002313 * the-workload) ]

    the-model = 3 ;; Fujitsu PRIMERGY RX2530 M6
    [ set power-consumption (125.2 + 0.00005361 * the-workload) ]

    the-model = 4 ;; New H3C Technologies Co. Ltd. H3C UniServer R4900 G5
    [ set power-consumption (106.7 + 0.00005369 * the-workload) ]

    the-model = 5 ;; Inspur Corporation Inspur NF8480M6
    [ set power-consumption (85.69 + 0.00007339 * the-workload) ]

    the-model = 6 ;; Dell Inc. PowerEdge R6515
    [ set power-consumption (94.56 + 0.00002335 * the-workload) ]

    the-model = 7 ;; LSDtech L224S-D/F/V-1
    [ set power-consumption (131.0 + 0.0001285  * the-workload) ]

    the-model = 8 ;; ASUSTeK Computer Inc. RS700A-E9-RS4V2
    [ set power-consumption (149.0 + 0.00002409 * the-workload) ]

    [ ;; Random Servers. Power consumption is assumed to increase with workloads linearly.
      ;; The model is based on Dell PowerEdge R6515 server using simple linear regression.
      if the-workload >= 0  [ set power-consumption (0.000023346 * the-workload + 94.5591) ]
    ]
  )

  report power-consumption

end
;;
;;
;;============================================================================;;
;; Quadratic Model
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-workload' : the total computing power used at the moment.
;;  'the-model'    : the server model.
;;============================================================================;;
to-report calc-power-consumption-quadratic [ the-workload the-model ]
  ;; Benchmark information were collected from spec.org
  let power-consumption 0
  (ifelse
    the-model = 1 ;; HP Enterprise ProLiant DL110 Gen10 Plus
    [ set power-consumption (99.59 + 0.000006182 * the-workload + 0.00000000001642   * (the-workload ^ 2)) ]

    the-model = 2 ;; Lenovo Global Technology ThinkSystem SR655
    [ set power-consumption (72.45 + 0.00002118 * the-workload  + 0.0000000000003178 * (the-workload ^ 2)) ]

    the-model = 3 ;; Fujitsu PRIMERGY RX2530 M6
    [ set power-consumption (158.0 + 0.00002481 * the-workload  + 0.000000000003803  * (the-workload ^ 2)) ]

    the-model = 4 ;; New H3C Technologies Co. Ltd. H3C UniServer R4900 G5
    [ set power-consumption (155.0 + 0.00001405 * the-workload  + 0.000000000004869  * (the-workload ^ 2)) ]

    the-model = 5 ;; Inspur Corporation Inspur NF8480M6
    [ set power-consumption (156.5 + 0.00003258 * the-workload  + 0.000000000003526  * (the-workload ^ 2)) ]

    the-model = 6 ;; Dell Inc. PowerEdge R6515
    [ set power-consumption (79.69 + 0.00003951 * the-workload  - 0.000000000002637  * (the-workload ^ 2)) ]

    the-model = 7 ;; LSDtech L224S-D/F/V-1
    [ set power-consumption (141.6 + 0.00009763 * the-workload  + 0.00000000001344   * (the-workload ^ 2)) ]

    the-model = 8 ;; ASUSTeK Computer Inc. RS700A-E9-RS4V2
    [ set power-consumption (137.4 + 0.00003067 * the-workload  - 0.0000000000005613 * (the-workload ^ 2)) ]

    [ ;; Random Servers. Power consumption is assumed to increase with workloads linearly.
      ;; The model is based on Dell PowerEdge R6515 server using simple linear regression.
      if the-workload >= 0  [ set power-consumption (0.000023346 * the-workload + 94.5591) ]
    ]
  )

  report power-consumption

end
;;
;;
;;============================================================================;;
;; Cubic Polynomial Model
;;--PARAMETERS----------------------------------------------------------------;;
;;  'the-workload' : the total computing power used at the moment.
;;  'the-model'    : the server model.
;;============================================================================;;
to-report calc-power-consumption-cubic [ the-workload the-model ]
  ;; benchmark information were collected from spec.org
  let power-consumption 0
  (ifelse
    the-model = 1 ;; HP Enterprise ProLiant DL110 Gen10 Plus
    [ set power-consumption (94.47 + 0.00003075 * the-workload - 0.000000000003008 * (the-workload ^ 2) + 0.000000000000000003904  * (the-workload ^ 3)) ]

    the-model = 2 ;; Lenovo Global Technology ThinkSystem SR655
    [ set power-consumption (57.96 + 0.00005883 * the-workload - 0.00000000001581  * (the-workload ^ 2) + 0.000000000000000001757  * (the-workload ^ 3)) ]

    the-model = 3 ;; Fujitsu PRIMERGY RX2530 M6
    [ set power-consumption (138.5 + 0.00006557 * the-workload - 0.00000000001031  * (the-workload ^ 2) + 0.000000000000000001243  * (the-workload ^ 3)) ]

    the-model = 4 ;; New H3C Technologies Co. Ltd. H3C UniServer R4900 G5
    [ set power-consumption (124.4 + 0.00007396 * the-workload - 0.00000000001445  * (the-workload ^ 2) + 0.000000000000000001583  * (the-workload ^ 3)) ]

    the-model = 5 ;; Inspur Corporation Inspur NF8480M6
    [ set power-consumption (130.1 + 0.00006889 * the-workload - 0.000000000004712 * (the-workload ^ 2) + 0.0000000000000000004750 * (the-workload ^ 3)) ]

    the-model = 6 ;; Dell Inc. PowerEdge R6515
    [ set power-consumption (66.12 + 0.00007478 * the-workload - 0.00000000001774  * (the-workload ^ 2) + 0.000000000000000001644  * (the-workload ^ 3)) ]

    the-model = 7 ;; LSDtech L224S-D/F/V-1
    [ set power-consumption (139.7 + 0.0001108  * the-workload - 0.00000000000161  * (the-workload ^ 2) + 0.000000000000000004365  * (the-workload ^ 3)) ]

    the-model = 8 ;; ASUSTeK Computer Inc. RS700A-E9-RS4V2
    [ set power-consumption (120.1 + 0.00005418 * the-workload - 0.000000000005828 * (the-workload ^ 2) + 0.0000000000000000002999 * (the-workload ^ 3)) ]

    [ ;; Random Servers. Power consumption is assumed to increase with workloads linearly.
      ;; The model is based on Dell PowerEdge R6515 server using simple linear regression.
      if the-workload >= 0      [ set power-consumption (0.000023346 * the-workload + 94.5591) ]
    ]
  )

  report power-consumption

end
;;
;;
;;============================================================================;;
;; Show IDs of all servers.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to show-server-label
  ifelse show-server-label?
  [
    ask servers [ set label id ]
    set show-server-label? false
  ]
  [
    ask servers [ set label "" ]
    set show-server-label? true
  ]

end
;;
;;
;;============================================================================;;
;; Show the model of all servers. This is useful when heterogeneous hardware
;; are used in the simulation.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to show-server-model
  ifelse show-server-model?
  [
    ask servers [ set label model ]
    set show-server-model? false
  ]
  [
    ask servers [ set label "" ]
    set show-server-model? true
  ]

end
;;
;;
;;============================================================================;;
;; Show IDs of all schedulers.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to show-scheduler-label
  ifelse show-scheduler-label?
  [
    ask schedulers [ set label id ]
    set show-scheduler-label? false
  ]
  [
    ask schedulers [ set label "" ]
    set show-scheduler-label? true
  ]

end
;;
;;
;;============================================================================;;
;; Show IDs of all services.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to show-service-label
  ifelse show-service-label?
  [
    ask services [ set label who set label-color blue - 1 ]
    set show-service-label? false
  ]
  [
    ask services [ set label "" ]
    set show-service-label? true
  ]

end
;;
;;
;;============================================================================;;
;; Show the number of deployment attempts of all services. This is only
;; useful when the cloud is over crowded.
;;--PARAMETERS----------------------------------------------------------------;;
;;  None
;;============================================================================;;
to show-service-attempt
  ifelse show-service-attempt?
  [
    ask services [ set label attempt set label-color blue - 1 ]
    set show-service-attempt? false
  ]
  [
    ask services [ set label "" ]
    set show-service-attempt? true
  ]

end
;;
;;
;;============================================================================;;
;; At the end of the simulation, print a summary of the resutls, especially
;; for those recorded global counters.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to print-summary
  print "==Summary of Results =========================================================="
  print "|-- Applications --------------------------------------------------------------"
  print (word "| Total Number of Services                            : " total-services)
  print (word "| Average Service Lifetime (Hour)                     : " (precision ((sys-service-lifetime-total + sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio) * simulation-time-unit / (total-services * 60)) 4))
  print (word "| Accumulated Number of Migrations (Consolidation)    : " sys-accumulated-migration-event-due-to-consolidation)
  print (word "| Accumulated Number of Migrations (Auto Migration)   : " sys-accumulated-migration-event-due-to-auto-migration)
  print (word "| SLA Violation due to Computing Power Shortage       : " (precision round sys-accumulated-service-ops-sla-vio 4))
  print (word "| SLA Violation due to Memory Shortage                : " (precision round sys-accumulated-service-mem-sla-vio 4))
  print (word "| SLA Violation due to Network Bandwidth Shortage     : " (precision round sys-accumulated-service-net-sla-vio 4))
  print (word "| Total Number of Rescheduled Services                : " sys-service-reschedule-counter)
  print (word "| Total Number of Rejected Services                   : " sys-service-rejection-counter)

  print "|-- Servers -------------------------------------------------------------------"
  print (word "| Total Number of Servers                             : " (rack-space * total-racks))
  print (word "| Total Computing Power Installed (million of ssj-ops): " (precision ((sum [ops-phy] of servers) / 1000000) 4))
  print (word "| Total Memory Installed (GB)                         : " (precision ((sum [mem-phy] of servers) / 1024) 4))
  print (word "| Total Network Bandwidth Installed (GBps)            : " (precision ((sum [net-phy] of servers) / 1024) 4))
  print "|-- Systems ------------------------------------------------------------------"
  print (word "| Accumulated Power Consumption (Unit<kWh>)           : " (precision (sys-power-consumption-total) 4))
  print "==Summary of Results =========================================================="

end
;;############################################################################;;
;; END: Utility Functions
;;############################################################################;;
;;
;;
;;############################################################################;;
;; START: Datacentre Initial Configurations
;;############################################################################;;
;;============================================================================;;
;; The initialization of a datacenter is required when configurations of the
;; simulation have been changed.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to initialize-datacenter
  build-service-submission-zone
  deploy-scheduler-nodes
  deploy-server-nodes

end
;;
;;
;;============================================================================;;
;; Creating a service submission zone in the simulation world.
;; Reserving three lines of patches on the top of the world. This place will be
;; used as service submission pool. All newly generated services will be
;; initially placed in this area. The number of patches in this area
;; defines the maximum number of concurrent services that can be submitted
;; at a time. This concurrency is controlled by the global variable
;; 'service-generation-speed'.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to build-service-submission-zone
  set current-top-cord (current-top-cord - service-submission-zone-height)
  ask patches with [ pycor > current-top-cord ]
  [ set pcolor blue + 1 ]

  set current-top-cord (current-top-cord - service-submission-delay-zone-height - def-sepa-line-width)

  ask patches with [ pycor = current-top-cord ]
  [ set pcolor grey ]

  set current-top-cord (current-top-cord - def-gap-width * 2)

end
;;
;;
;;============================================================================;;
;; Creating and placing scheduler nodes. The number of schedulers is determined
;; by the number of racks, i.e., each rack will have a dedicated scheduler.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to deploy-scheduler-nodes
  let gap ((max-pxcor - 2 - total-racks * 3) / total-racks)
  let idx 1
  let x-cord (gap / 2 + 1)
  repeat total-racks
  [
    create-schedulers 1
    [
      set shape "computer server"
      set ycor current-top-cord
      set xcor x-cord
      set color green
      set size 3
      set id idx
      set capacity scheduler-queue-capacity

      set ops-hist []
      set ops-hist fput 0 ops-hist
      set mem-hist []
      set mem-hist fput 0 mem-hist
      set net-hist []
      set net-hist fput 0 net-hist
    ]
    set idx idx + 1
    set x-cord (x-cord + gap + 3)
  ]

  set current-top-cord current-top-cord - 2
  ask patches with [ pycor = current-top-cord ]
  [ set pcolor grey ]

end
;;
;;
;;============================================================================;;
;; Deploying servers in the datacenter. The number of servers are jointly
;; determined by the global variables: 'rack-space' and 'total-racks'.
;;--PARAMETERS------------------------------------------------------------------
;;  None
;;============================================================================;;
to deploy-server-nodes
  let server-models read-from-string server-model
  let svr-icon-size 2
  let h-gap ((max-pxcor - 2 - total-racks * 3) / total-racks)
  let v-gap ((current-top-cord - 2 - rack-space * 3) / rack-space)
  let svr-x-cord (h-gap / 2 + 1)
  let rack-idx 1
  let the-svr-model 1 ;; Use server model 1 as default
  if not empty? server-model [ set the-svr-model first server-models ]

  repeat total-racks
  [
    let svr-y-cord current-top-cord - 3
    if datacenter-level-heterogeneity? and (not empty? server-models)
    [ set the-svr-model one-of server-models ]

    let rack-svr-count 0
    repeat rack-space
    [
      set rack-svr-count rack-svr-count + 1
      if datacenter-level-heterogeneity? and rack-level-heterogeneity? and (not empty? server-models)
      [ set the-svr-model one-of server-models ]

      create-servers 1
      [
        set shape "container"
        set xcor svr-x-cord
        set ycor svr-y-cord
        set size svr-icon-size
        set id rack-svr-count
        set rack rack-idx
        set model the-svr-model

        set status "OFF"
        set color white
        (ifelse ;; Benchmark information were collected from spec.org
          model = 1 [ set ops-phy 3318199  set mem-phy 65536  set net-phy 50000 set base-power 94.1 ]
          model = 2 [ set ops-phy 6114552  set mem-phy 131072 set net-phy 2000  set base-power 53   ]
          model = 3 [ set ops-phy 7573884  set mem-phy 262144 set net-phy 4000  set base-power 131  ]
          model = 4 [ set ops-phy 8135458  set mem-phy 262144 set net-phy 4000  set base-power 119  ]
          model = 5 [ set ops-phy 11556864 set mem-phy 393216 set net-phy 1000  set base-power 104  ]
          model = 6 [ set ops-phy 6124305  set mem-phy 65536  set net-phy 2000  set base-power 54.5 ]
          model = 7 [ set ops-phy 2297344  set mem-phy 200704 set net-phy 2000  set base-power 131  ]
          model = 8 [ set ops-phy 11702137 set mem-phy 262144 set net-phy 2000  set base-power 106  ]
          [ set ops-phy 3318199  set mem-phy 65536  set net-phy 50000 set base-power 94.1 ]
        )
        reset-server self
      ]
      set svr-y-cord (svr-y-cord - v-gap - 3)
    ]

    set svr-x-cord (svr-x-cord + h-gap + 3)
    set rack-idx rack-idx + 1
  ]

end
;;
;;
;;############################################################################;;
;; END: Datacentre Initial Configurations
;;############################################################################;;
@#$#@#$#@
GRAPHICS-WINDOW
351
11
1649
990
-1
-1
10.0
1
9
1
1
1
0
0
0
1
0
128
0
96
0
0
1
ticks
30.0

BUTTON
135
10
237
43
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
245
11
347
44
Go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
0
50
95
83
rack-space
rack-space
8
36
10.0
2
1
NIL
HORIZONTAL

SLIDER
95
50
204
83
total-racks
total-racks
1
36
1.0
1
1
NIL
HORIZONTAL

TEXTBOX
0
35
93
53
  Global
11
0.0
1

SLIDER
0
410
179
443
scheduler-queue-capacity
scheduler-queue-capacity
10
100
50.0
1
1
NIL
HORIZONTAL

SLIDER
179
410
351
443
scheduler-history-length
scheduler-history-length
0
200
10.0
5
1
NIL
HORIZONTAL

SWITCH
0
772
255
805
datacenter-level-heterogeneity?
datacenter-level-heterogeneity?
0
1
-1000

SWITCH
0
806
193
839
rack-level-heterogeneity?
rack-level-heterogeneity?
0
1
-1000

INPUTBOX
194
503
352
563
server-model
[1 2 3 4 5 6 7 8]
1
0
String

BUTTON
0
850
113
883
Show Server Label
show-server-label
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
112
849
223
884
Show Server Model
show-server-model
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

INPUTBOX
110
85
225
147
service-lifetime
[144 576]
1
0
String

INPUTBOX
0
234
110
301
mem-access-ratio
[2 4]
1
0
String

CHOOSER
0
304
166
349
service-submission-strategy
service-submission-strategy
"closest" "resource-pattern-matching"
0

SLIDER
110
234
351
267
service-generation-speed
service-generation-speed
1
500
26.0
5
1
NIL
HORIZONTAL

INPUTBOX
0
85
110
145
total-services
100.0
1
0
Number

TEXTBOX
0
158
108
181
Service Related
11
0.0
1

INPUTBOX
0
173
110
236
cpu-usage-dist
[2 4]
1
0
String

INPUTBOX
110
173
225
236
mem-usage-dist
[2 4]
1
0
String

INPUTBOX
225
173
351
236
net-usage-dist
[2 4]
1
0
String

CHOOSER
167
304
351
349
service-placement-algorithm
service-placement-algorithm
"random" "first-fit" "balanced-fit" "max-utilization" "min-power" "AI" "AI-phase1" "AI-phase2" "AI-phase3"
5

CHOOSER
178
444
351
489
server-standby-strategy
server-standby-strategy
"adaptive" "all-off" "all-on" "10%-on" "20%-on" "30%-on" "40%-on" "50%-on"
1

INPUTBOX
0
564
195
624
server-mem-utilization-threshold
[20 90]
1
0
String

INPUTBOX
0
503
195
563
server-cpu-utilization-threshold
[20 90]
1
0
String

INPUTBOX
0
626
195
686
server-net-utilization-threshold
[20 90]
1
0
String

PLOT
2000
408
2345
544
Power Consumption
Time
kW/h
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot (sys-power-consumption-total)"

SWITCH
196
565
352
598
consolidation?
consolidation?
0
1
-1000

SLIDER
0
689
195
722
consolidation-interval
consolidation-interval
1
1440
12.0
1
1
NIL
HORIZONTAL

CHOOSER
0
726
194
771
server-consolidation-strategy
server-consolidation-strategy
"within-datacenter" "within-rack"
0

SWITCH
196
598
352
631
auto-migration?
auto-migration?
0
1
-1000

SWITCH
193
806
352
839
show-migr-move?
show-migr-move?
0
1
-1000

SLIDER
110
268
351
301
service-history-length
service-history-length
0
200
10.0
5
1
NIL
HORIZONTAL

CHOOSER
195
727
353
772
power-estimation-method
power-estimation-method
"max" "mean" "median" "configured" "linear-regression"
2

SLIDER
205
50
351
83
simulation-time-unit
simulation-time-unit
1
120
5.0
1
1
NIL
HORIZONTAL

TEXTBOX
7
394
174
412
Scheduler Related
11
0.0
1

TEXTBOX
5
488
172
507
Server Related
11
0.0
1

CHOOSER
167
351
351
396
staged-evaluation
staged-evaluation
"1" "2" "3"
2

CHOOSER
196
632
352
677
auto-migration-strategy
auto-migration-strategy
"least-migration-time" "least-migration-number"
0

INPUTBOX
225
85
351
145
rand-seed
99999.0
1
0
Number

PLOT
1652
11
1997
140
Computing Resource Usage
Time
ssj_ops
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 1 -5298144 true "" "plot sum [ops-now] of servers"

PLOT
1653
142
1997
274
Memory Resource Usage
Time
MB
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 1 -8630108 true "" "plot sum [mem-now] of servers"

PLOT
1653
277
1997
405
Networking Resource Usage
Time
MBps
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 1 -12345184 true "" "plot sum [net-now] of servers "

PLOT
2000
546
2345
696
# of Migration Events triggered by Consolidation
Time
# of Events
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot sys-current-migration-event-due-to-consolidation"

PLOT
1652
546
1997
696
# of Migration triggered by Auto Migration
Time
# of Events
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot sys-current-migration-event-due-to-auto-migration"

PLOT
2000
10
2344
140
CPU Utilization (Active Servers)
Time
Utils (%)
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"mean" 1.0 0 -2674135 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (mean [ops-now / ops-phy] of a-svrs) ]\nplot p-value * 100"
"median" 1.0 0 -8630108 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (median [ops-now / ops-phy] of a-svrs) ]\nplot p-value * 100"
"max" 1.0 0 -13791810 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (max [ops-now / ops-phy] of a-svrs) ]\nplot p-value * 100"
"min" 1.0 0 -13840069 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (min [ops-now / ops-phy] of a-svrs) ]\nplot p-value * 100"

PLOT
2000
142
2345
274
MEM Utilization (Active Servers)
Time
Utils (%)
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"mean" 1.0 0 -2674135 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (mean [mem-now / mem-phy] of a-svrs) ]\nplot p-value * 100"
"median" 1.0 0 -8630108 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (median [mem-now / mem-phy] of a-svrs) ]\nplot p-value * 100"
"max" 1.0 0 -13791810 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (max [mem-now / mem-phy] of a-svrs) ]\nplot p-value * 100"
"min" 1.0 0 -13840069 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (min [mem-now / mem-phy] of a-svrs) ]\nplot p-value * 100"

PLOT
2000
277
2345
405
NET Utilization (Active Servers)
Time
Utils (%)
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"mean" 1.0 0 -2674135 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (mean [net-now / net-phy] of a-svrs) ]\nplot p-value * 100"
"median" 1.0 0 -8630108 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (median [net-now / net-phy] of a-svrs) ]\nplot p-value * 100"
"max" 1.0 0 -13791810 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (max [net-now / net-phy] of a-svrs) ]\nplot p-value * 100"
"min" 1.0 0 -13840069 true "" "let a-svrs servers with [status = \"ON\" or status = \"OVERLOAD\"]\nlet p-value 0\nif count a-svrs > 0\n[ set p-value (min [net-now / net-phy] of a-svrs) ]\nplot p-value * 100"

PLOT
1652
408
1997
544
Server Status
Time
Count
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"OVERLOAD" 1.0 0 -2674135 true "" "plot count servers with [status = \"OVERLOAD\"]"
"NORMAL" 1.0 0 -10899396 true "" "plot count servers with [status = \"ON\"]"
"IDLE" 1.0 0 -13791810 true "" "plot count servers with [status = \"READY\" or status = \"IDLE\"]"
"OFF" 1.0 0 -16777216 true "" "plot count servers with [status = \"OFF\"]"

PLOT
1653
699
1997
849
SLA Violation (Lifetime Extended)
Time
Unit
0.0
10.0
0.0
0.01
true
true
"" ""
PENS
"CPU" 1.0 0 -2674135 true "" "plot sys-current-service-ops-sla-vio"
"MEM" 1.0 0 -8630108 true "" "plot sys-current-service-mem-sla-vio"
"NET" 1.0 0 -13791810 true "" "plot sys-current-service-net-sla-vio"

PLOT
2000
698
2347
848
Service Rejection (Accumulated)
Time
Count
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"Rescheduled" 1.0 0 -8630108 true "" "plot sys-service-reschedule-counter"
"Rejected" 1.0 0 -2674135 true "" "plot sys-service-rejection-counter"

CHOOSER
196
678
352
723
power-model-method
power-model-method
"stepwise-simple-linear-regression" "simple-linear-regression" "quadratic-polynomial" "cubic-polynomial"
0

SWITCH
221
772
352
805
show-trace?
show-trace?
1
1
-1000

BUTTON
225
850
352
883
Show Scheduler Label
show-scheduler-label
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
0
885
168
918
Show Service label
show-service-label
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
169
886
352
919
Show Service Attempt
show-service-attempt
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
1653
851
1997
991
Service Deployment Delay
TIme
Second
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"Virtual Machine" 1.0 0 -2674135 true "" "plot sys-accumulated-delay-from-vm"
"Container" 1.0 0 -10899396 true "" "plot sys-accumulated-delay-from-ct"
"Svr Cold Start" 1.0 0 -13345367 true "" "plot sys-accumulated-delay-from-svr-cold-start"

PLOT
1999
850
2346
991
Service Status
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"In Queue" 1.0 0 -13345367 true "" "plot count services with [status = \"OFFLINE\"]"
"Submitted" 1.0 0 -1184463 true "" "plot count services with [status = \"SUBMITTED\"]"
"Scheduled" 1.0 0 -10899396 true "" "plot count services with [status = \"SCHEDULED\"]"
"Deployed" 1.0 0 -2064490 true "" "plot count services with [status = \"DEPLOYED\"]"
"Running" 1.0 0 -2674135 true "" "plot count services with [status = \"RUNNING\"]"
"Completed" 1.0 0 -16777216 true "" "plot sys-current-service-completed"

SWITCH
0
351
166
384
use-sla-local-factor?
use-sla-local-factor?
0
1
-1000

@#$#@#$#@
## WHAT IS IT?

The simulation model was built for studying resource management in the clouds, focusing on how service placement strategies, service auto-migration, and server consolidation affect the overall performance of homogeneous and heterogeneous clouds, with regard to energy consumption, resource utilization, service-level agreement violation, and many other aspects that are important to cloud service providers and researchers.     


## HOW TO USE IT

### Global Configurations

"**_rack-space_**": The capacity of each server rack. The parameter determines how many servers can be installed in a rack.
RANGE: [8 - 36]; INCREMENT: [1]; DEFAULT: [12]

"**_total-racks_**": The total number of racks in the datacenter. In conjunction with the _rack-space_, the maximum number of servers can be determined by _rack-space_ * _total-racks_, maximized at 1296.
RANGE: [1 - 36]; INCREMENT: [1]; DEFAULT: [8]

"**_simulation-time-unit_**": This parameter specifies the unit of one 'tick', measured in miniute. For example, 5 indicates that every simulation _tick_ represents a 5 minutes time elapsed. More importantly, it implies that all status of the elements in the datacenter will be updated every 5 minutes. It can be configured from 1 minute to 24 hours (1440 minutes).
RANGE: [1 - 1440]; INCREMENT: [1]; DEFAULT: [5]

"**_total-services_**": A user can specify a number of services to be deployed in the datacenter. However, not all the services will be deployed at once. The number of services to be sent to the datacenter for deployment will be jointly determined with the _service-generation-speed_ parameter.
VALUE: [ POSITIVE INTEGER ]

"**_service-lifetime_**": Each service will have a lifetime randomly drawn from a Gaussian distribution in the range specified here.
RANGE: [ MIN MAX ]

"**_rand-seed_**": To ensure the resutls from a simulation are reproducible, a random seed can be specified here. If the value of the random seed is greater than zero, the simulation result will be tied to the seed while keeping all other configurations intact. If the same seed was used in another run, the "same" results would be produced. However, to also allow randomness across multiple runs, the seed can be set to zero or a negative value, so that for each run, a pseudo-random seed will be used.
VALUE: [ INTEGER ]



### Service Configurations

"**_cpu-usage-dist_**", "**_mem-usage-dist_**", "**_net-usage-dist_**": For each service, three types of resources are considered, i.e., CPU, Memory and Network Bandwidth. At runtime, the usage of each type of resource will be drawn from a Beta distribution on a per 'tick' basis. This process will be performed for each individual service.    
RANGE: [ ALPHA BETA ]

"**_mem-access-ratio_**": This parameter determines how frequently the service's memory is being accessed. This is an important factor to be considered during service migration, as the size of the _busy_ memory (jointly with the available source-to-destination bandwidth) will influence speed of service migration. For each service, the ratio can be drawn from a given Beta distribution.
RANGE: [ ALPHA BETA ]

"**_service-generation-speed_**": It determines how many services will be generated in one simulation unit, i.e., how many concurrent services need to be deployed in the datacenter. It is an important factor for the evaluation of the efficiency of schedulers and placement algorithms in terms of scalability and speed. 
RANGE: [1 - 500]; INCREMENT: [5]; DEFAULT: [50]

"**_service-history-length_**": Each running service will cache a number of historical resource usage information (CPU, MEM, and NET). The values will be used for some advanced decision-making processes. A longer list may provide better forecast accuracy but slows down the simulation process.
VALUE: [ POSITIVE INTEGER ]

"**_service-submission-strategy_**": When a service is first created, it needs to be assigned a SCHEDULER. There are two strategies implemented in the simulation so far: _closest_ and _resource-pattern-matching_.

1. _closest_ strategy: a service will be assigned a SCHEDULER who is physically closest to it.

2. _resource-pattern-matching_ strategy: This is only applied to a datacenter that contains heterogeneous servers. In a heterogeneous cloud, servers have different configurations. The goal of this strategy is to maximize the overall resource utilization of the datacenter. Let R(a) denotes a tuple representing the three types of resource of a SERVICE, a; and R(s) denotes a tuple representing the available resources of the rack under the management of a SCHEDULER, s, the best match is determined by:
  [ min{ max{R(a)/R(s)} - min{R(a)/R(s)} } ]


"**_service-placement-algorithm_**": The service placement algorithms are the core in studying resource optimization in the cloud. There are currently five placement algorithms implemented: _random_, _first-fit_, _balanced-fit_, _max-utilization_, and _min-power_. Depending on the objectives set out for the cloud, algorithms shall be selected accordingly.

1. _random_: When a SERVICE has arrived at a SCHEDULER, the scheduler will find a random SERVER for the deployment of the service, with the following constraints: (1) the hosting server has sufficient resources for the service; (2) the deployment of the service will not make the server entering the "_OVERLOAD_" mode. Technically, a blind random placement is never encouraged in real environments, it is thus only provided as a baseline for comparison with other thoughtful algorithms.  

2. _first-fit_: the _first-fit) algorithm is one of the simplest algorithms used in resource optimization and application scheduling. The _first-fit_ in the simulator is a three-step process:
    (i)   _sort servers in the rack in an ascending order_
    (ii)  _place service(s) on the first server with sufficient resources in the list,     if the  server could not meet the conditions, move on to the next one in the list,     and so on._
    (iii) _if a server has been found, ask the service to move to the server, otherwise,         place the service back to the 'service submission zone'._  

3. _balanced_fit_: One of the important factors in measuring resource utilization is the resource fragmentation. In a cloud environment, servers often have different configurations, as well as services whose configurations vary largely depending on their types and/or user preferences. A careless placement might create small resource fragments that could not be used further. For instance, if two servers have the configurations Rc(s1) = {1000, 800, 100} and Rc(s2) = { 1000, 1000, 500}, i.e., the installed resources of the server (CPU, MEM and NET); and a service with the configuration Rc(a) = {500, 500, 100} to be deployed. If _a_ was deployed on _s1_, the remaining resources on _s1_ would be R'c(s1) = {500, 300, 0}; and for _s2_, R'c(s2) = {500, 500, 400}. Obviously, the former deployment makes _s1_ unavailable for future service placement as it has no bandwidth resources left. In comparison, the latter placement makes both the _s1_ and _s2_ available for future service deployment. The steps involved in the _balanced fit_ are outlined below:
    (i) _Since different types of resources are often measured in different units,     comparing different types of resources makes no sense. Comparing normalized             values or ratios would be reasonable. Furthermore, normalization can be             challenging as there is no single reference value across heterogeneous servers             and types of services. Thus, to calculate ratios, the resource tuples for the             service (requested resources) and candidate server (currently available             resources), Rc(a) and (Rc(s) - Ro(s)), need to be firstly collected. Note that Ro(.)     indicates the currently occupied resources._
    (ii)  _For each candidate server, calculate the ratios of the resources, i.e.,             Rc(a)/(Rc(s) - Ro(s))._
    (iii) _Identify the smallest distance between the ratios of the resources of all      the candidate servers, i.e., min{ max{ Rc(a)/(Rc(s) - Ro(s) } - min{ Rc(a)/(Rc(s) -         Ro(s) } }._
    (iv)  _Deploy the service on the server, which has the smallest resource distance,         otherwise, resubmit the service to the 'service submission zone' if no servers could be found._

4. _max-utilization_: This placement algorithm minimizes the residual resources of the server after the service deployment. I.e., given the physical and currently occupied resources of server, Rc(s) and Ro(s), and the requested resources from the service, Rc(a) that is to be deployed, the algorithm tries to find a server with minimum residual as follows:
min{ sum(1 - (Ro(s) + Rc(a))/Rc(s)) }, for all s in S.

5. _min-power_: This algorithm tries to place a service to a server so that the resulting placement will have a minimum increase of energy consumption in the datacenter.

"**_staged-evaluation_**": This parameter is used in conjunction with the _service-placement-algorithm_. When a scheduler evaluates servers for a service placement, it may find any servers in the rack (or datacenter) that can satisfy the resource requirements of the service, hence the 1-staged evaluation, or it can first evaluate all _active_ servers, i.e., the servers with status = _ON_, _READY_, or _IDLE_. If no satisfactory servers can be found, it will find a server with status = _OFF_. In this case, a fixed penalty point (the 'server-cold-start-delay') will be given to the service who is the first one using the server, hence the 2-staged evaluation. Lastly, a scheduler can also evaluate all the _running_ servers (i.e., status = _ON_ or _READY_) at first, then moves on to _IDLE_ servers (i.e., status = _IDLE_), and lastly tries with the _OFF_ servers, hence the 3-staged evaluation.



### Scheduler Configurations

"**_scheduler-queue-capacity_**": Each scheduler has its capacity specified by the parameter. It is reserved to study queuing effects in the cloud. 
RANGE: [10 - 100]; INCREMENT: [1]; DEFAULT: [50]

"**_scheduler-history-length_**": This value indicates how much historical information a scheduler will cache, so that some decisions could be made based on the statistics collected here. Caching more information, i.e., the CPU, MEM and NET usage, can potentially support a more accurate decision making, but on the other hand, it may slow down the simulation significantly and result in a much bigger memory footprint.
RANG: [0, 200]; INCREMENT: [5]; DEFAULT; [5]

"**_server-standby-strategy_**": A SCHEDULER is responsible for managing servers in the rack. A scheduler decides when a server should be switched off to save energy or switched on for an upcoming service deployment. The current implementation maintains a fixed number of standby servers, _all-on_ or _all-off_. Adaptive strategies are still in the TO-DO list.




### Server Configurations

"**_server-cpu-utilization-threadhold_**", "**_server-mem-utilization-threadhold_**", "**_server-net-utilization-threadhold_**": These parameters specify the under-utilization and over-utilization thresholds, which will be used for determining whether a server consolidation and/or an auto-migration process will be triggered.

"**_server-model_**": In general, different models of servers may have different default OEM configurations and energy consumption patterns. To simulate a heterogeneous datacenter environment, several pre-built servers (coded from 0 to 8) can be selected. The model of the server can be specified in the _server-model_ global variable. For example, using servers of Model 1, 2, 3 in the simulation can be specified as: [1 2 3]. The detailed server specifications and their associated code can be found in the **Server Specifications** section.

"**_datacenter-level-heterogeneity?_**": This switch indicates whether multiple server models will be allowed in the cloud. If it's switched off, the default server (coded 1) will be used, regardless of the servers specified in the _server-model_. If a specific server is needed in the simulation, enable this parameter and specify the specific server in the _server_model_. For example, if the cloud contains only the Dell PowerEdge R6515 servers, this can be configured as: _datacenter-level-heterogeneity? = ON_ and _server-model = [6]_.

"**_rack-level-heterogeneity?_**": If this switch is on and the _server-model_ contains multiple servers, each rack will contain servers with different configurations. This switch only works when the _datacenter-level-heterogeneity?_ is switched on.

"**_consolidation?_**": This switch tells the simulator whether under-utilized servers should be consolidated, i.e., migrate all services out of the servers, so that the server can be switched off to save energy.

"**_consolidation-interval_**": Consolidation is an expensive process. It is not recommended to perform server consolidation frequently. Each tick in this sliding bar indicates a 'simulation-time-unit'.
RANGE: [1 - 1440]; INCREMENT: [1]; DEFAULT: [12]

"**_server-consolidation-strategy_**": When trying to consolidate a server, all services will be migrated out or not at all. When migrating services out of a server, target servers must first be identified. The target server can be a server in the same rack (_within-rack_) or on other racks in the datacenter (_within-datacenter_). Technically, migrating within a rack would be preferred as the network traffic would be kept local. On the other side, there may not be many servers that can be used in the rack. It is about balancing between better optimization and minimizing burden on cloud network.


"**_power-estimation-method_**": This parameter is mainly used with the _min-power_ placement algorithm. It is used to estimate energy consumption of a server if a given service was placed on it. The estimation can be based on the following statistics:

1. _max_: The max _ops_ value the service has experienced in its cached history.
2. _mean_: The average _ops_ of the service in its cached history.
3. _median_: The median _ops_ of the service in its cached history.
4. _configured_: The initially configured _ops_ of the service, i.e., 'ops-cnf'.
5. _linear-regression_: Reserved


"**_show-migr-move?_**": If enabled, service migration movement will be displayed. This visualization will not affect the accuracy of any calculations. The moving objects are a different breed of agents that only for visualization purpose. 

"**_auto-migration?_**: When a server enters the _OVERLOAD_ mode, the server will migrate some services out automatically, if this switch is enabled.  

"**_auto-migration-strategy_**": When migrating services out of a server due to over-utilization, it is only necessary to migrate some services out until the server is back to a _normal_ status. Thus, which services should be migrated depends on the objectives of the system. In the simulation, two strategies have been implemented:

1. _least migration time_: A service migration time is generally determined by the P2P network bandwidth, memory footprint, and _memory dirtying rate_ (i.e., how frequent the service's memory is being accessed). The preference list for services to be migrated out is a list of services sorted in ascending order according the values in the above factors. The service in the first place of the list will be migrated out. If the server is still over-utilized, then the second service will be moved out, so on and so forth.

2. _leas migration number_: In this case, the service with the most aggressive resource demand will be migrated out first.


"**_power-model-method_**: As mentioned above, servers with different specifications may exhabit different patterns of energy consumption. Thus, we need to build an energy consumption model for each _model_ of servers. Additionally, to evaluate the accuracy of different models, a _stepwise-simple-linear-regression_, _simple_linear_regression_, _Quadratic_Polynomial_, and _Cubic-Polynomial_ energy consumption models have been built for each _model_ of servers, based on the data collected from spac.org. The accuracy and other statistics associated with the models can be found in the publication.
 
#### Server Specifications
0. Random Server
[ CODE: 0; CPU: Rand(2M ~ 10M ops);                  RAM: Rand(64 ~ 512GB) ]

1. HP ProLiant DL110 Gen10 Plus
[ CODE: 1; CPU: Intel Xeon Gold 6314U @2.30GHz;      RAM: 64GB             ]

2. Lenovo ThinkSystem SR655
[ CODE: 2; CPU: AMD EPYC 7763 @2.45GHz;              RAM: 128GB            ]

3. Fujitsu PRIMERGY RX2530 M6
[ CODE: 3; CPU: Intel Xeon Platinum 8380 @2.30GHz;   RAM: 256GB            ]

4. New H3C Technologies H3C UniServer R4900 G5
[ CODE: 4; CPU: Intel Xeon Platinum 8380 @2.30GHz;   RAM: 256GB            ]

5. Inspur Corporation Inspur NF8480M6
[ CODE: 5; CPU: Intel Xeon Platinum 8380HL @2.90GHz; RAM: 384GB            ]

6. Dell Inc. PowerEdge R6515
[ CODE: 6; CPU: AMD EPYC 7702P @2.00GHz;             RAM: 64GB             ]

7. LSDtech L224S-D/F/V-1
[ CODE: 7; CPU: Intel Xeon Gold 6136 @3.00GHz;       RAM: 196GB            ]

8. ASUSTeK Computer Inc. RS700A-E9-RS4V2
[ CODE: 8; CPU: AMD EPYC 7742 @2.25GHz;              RAM: 256GB            ]


### Display Related

"**_Show Trace_**": Display the service migration path in the display.

"**_Show Server Label_**": Display the server ID (the server index in the rack).

"**_Show Server Model_**: Display the model code of servers.

"**_Show Scheduler Label_**: Display the ID of schedulers.

"**_Show Service Label_**: Display the agent ID of services.

"**_Show Service Attempt_**: Display how many times a service has been tried to deploy in the cloud.


## CREDITS AND REFERENCES
If you have found this software helpful in your research, please consider citing the preprint published at TechRxiv, IEEE.

Dong, D. Agent-based cloud simulation model for resource management. J Cloud Comp 12, 156 (2023). https://doi.org/10.1186/s13677-023-00540-5
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

computer server
false
0
Rectangle -7500403 true true 75 30 225 270
Line -16777216 false 210 30 210 195
Line -16777216 false 90 30 90 195
Line -16777216 false 90 195 210 195
Rectangle -10899396 true false 184 34 200 40
Rectangle -10899396 true false 184 47 200 53
Rectangle -10899396 true false 184 63 200 69
Line -16777216 false 90 210 90 255
Line -16777216 false 105 210 105 255
Line -16777216 false 120 210 120 255
Line -16777216 false 135 210 135 255
Line -16777216 false 165 210 165 255
Line -16777216 false 180 210 180 255
Line -16777216 false 195 210 195 255
Line -16777216 false 210 210 210 255
Rectangle -7500403 true true 84 232 219 236
Rectangle -16777216 false false 101 172 112 184

computer workstation
false
0
Rectangle -7500403 true true 60 45 240 180
Polygon -7500403 true true 90 180 105 195 135 195 135 210 165 210 165 195 195 195 210 180
Rectangle -16777216 true false 75 60 225 165
Rectangle -7500403 true true 45 210 255 255
Rectangle -10899396 true false 249 223 237 217
Line -16777216 false 60 225 120 225

container
false
0
Rectangle -7500403 false false 0 75 300 225
Rectangle -7500403 true true 0 75 300 225
Line -16777216 false 0 210 300 210
Line -16777216 false 0 90 300 90
Line -16777216 false 150 90 150 210
Line -16777216 false 120 90 120 210
Line -16777216 false 90 90 90 210
Line -16777216 false 240 90 240 210
Line -16777216 false 270 90 270 210
Line -16777216 false 30 90 30 210
Line -16777216 false 60 90 60 210
Line -16777216 false 210 90 210 210
Line -16777216 false 180 90 180 210

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

net-switch
true
0
Polygon -13791810 true false 90 90 15 165 15 210 225 210 225 165 285 90 90 90 285 90 285 135 225 210 225 165 90 90
Polygon -13791810 true false 75 60 270 60 75 60 15 135
Polygon -13791810 true false 15 165 90 90 285 90 225 165
Line -1 false 15 165 225 165
Line -1 false 225 165 285 90
Line -1 false 15 135 15 135
Line -1 false 15 165 15 210
Line -1 false 15 210 225 210
Line -1 false 225 165 225 210
Line -1 false 225 210 285 135
Line -1 false 285 135 285 90
Line -1 false 90 90 285 90
Line -1 false 90 90 15 165
Line -1 false 150 105 240 105
Line -1 false 225 90 240 105
Line -1 false 240 105 225 120
Line -1 false 180 120 90 120
Line -1 false 210 135 120 135
Line -1 false 150 150 60 150
Line -1 false 90 120 105 105
Line -1 false 90 120 105 135
Line -1 false 195 120 210 135
Line -1 false 210 135 195 150
Line -1 false 60 150 75 135
Line -1 false 60 150 75 165

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.3.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="energy-hete-123staged-noAutomigr-noConsol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;1&quot;"/>
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-homo-123staged-noAutomigr-noConsol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;1&quot;"/>
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-hete-123staged-automigr-noConsol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;1&quot;"/>
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-homo-123staged-automigr-noConsol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;1&quot;"/>
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-hete-23staged-automigr-consol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-homo-23staged-automigr-consol-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;2&quot;"/>
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="energy-homo-3staged-automigr-consol-energymodel-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
      <value value="&quot;simple-linear-regression&quot;"/>
      <value value="&quot;quadratic-polynomial&quot;"/>
      <value value="&quot;cubic-polynomial&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-hete-3staged-noAutomigr-noConsol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-homo-3staged-noAutomigr-noConsol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-hete-3staged-automigr-noConsol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-homo-3staged-automigr-noConsol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-hete-3staged-automigr-consol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="util-homo-3staged-automigr-consol-100" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[300 300]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;random&quot;"/>
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="sbro-hete-2staged-tfAutomigr-tfConsol-psm-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <metric>(sys-accumulated-migration-event-due-to-consolidation)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="1000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[144 576]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;2&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="sbro-homo-2staged-tfAutomigr-tfConsol-psm-100" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-power-consumption-total)</metric>
    <metric>(sys-accumulated-service-ops-sla-vio + sys-accumulated-service-mem-sla-vio + sys-accumulated-service-net-sla-vio)</metric>
    <metric>(sys-accumulated-migration-event-due-to-auto-migration)</metric>
    <metric>(sys-accumulated-migration-event-due-to-consolidation)</metric>
    <enumeratedValueSet variable="rack-space">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="1000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[144 576]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;first-fit&quot;"/>
      <value value="&quot;balanced-fit&quot;"/>
      <value value="&quot;max-utilization&quot;"/>
      <value value="&quot;min-power&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;2&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="78389459"/>
      <value value="33723175"/>
      <value value="84604581"/>
      <value value="57807322"/>
      <value value="33047754"/>
      <value value="5673057"/>
      <value value="71458564"/>
      <value value="38906898"/>
      <value value="81943259"/>
      <value value="90518310"/>
      <value value="50082228"/>
      <value value="79666492"/>
      <value value="44009356"/>
      <value value="85705203"/>
      <value value="20750924"/>
      <value value="62713447"/>
      <value value="96878016"/>
      <value value="66317258"/>
      <value value="8799021"/>
      <value value="30171548"/>
      <value value="85132824"/>
      <value value="13137777"/>
      <value value="3703550"/>
      <value value="82165898"/>
      <value value="34455035"/>
      <value value="34116410"/>
      <value value="25164538"/>
      <value value="12983870"/>
      <value value="10706892"/>
      <value value="83669664"/>
      <value value="74559435"/>
      <value value="38385999"/>
      <value value="13382862"/>
      <value value="26894479"/>
      <value value="96180561"/>
      <value value="42106246"/>
      <value value="68704114"/>
      <value value="67152249"/>
      <value value="97704838"/>
      <value value="62518999"/>
      <value value="45211588"/>
      <value value="48786139"/>
      <value value="91512188"/>
      <value value="18458633"/>
      <value value="60071955"/>
      <value value="77415892"/>
      <value value="23540181"/>
      <value value="57736950"/>
      <value value="93300039"/>
      <value value="84024880"/>
      <value value="283561"/>
      <value value="63643656"/>
      <value value="32499730"/>
      <value value="8017168"/>
      <value value="62051834"/>
      <value value="83708485"/>
      <value value="43615649"/>
      <value value="5439560"/>
      <value value="99229147"/>
      <value value="33501380"/>
      <value value="45735931"/>
      <value value="21930036"/>
      <value value="82497244"/>
      <value value="17466412"/>
      <value value="84150569"/>
      <value value="53518097"/>
      <value value="17464782"/>
      <value value="4123795"/>
      <value value="71902779"/>
      <value value="6745597"/>
      <value value="5328325"/>
      <value value="40336474"/>
      <value value="10232059"/>
      <value value="87472079"/>
      <value value="12105566"/>
      <value value="9347638"/>
      <value value="70176465"/>
      <value value="11286042"/>
      <value value="11424025"/>
      <value value="82591386"/>
      <value value="93095962"/>
      <value value="65457498"/>
      <value value="27829389"/>
      <value value="35523835"/>
      <value value="41542298"/>
      <value value="4021761"/>
      <value value="80952905"/>
      <value value="27271376"/>
      <value value="36206992"/>
      <value value="49840011"/>
      <value value="52943301"/>
      <value value="92940563"/>
      <value value="23281369"/>
      <value value="86348689"/>
      <value value="58929518"/>
      <value value="58682913"/>
      <value value="98990485"/>
      <value value="63428092"/>
      <value value="46467304"/>
      <value value="59966231"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="11" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>(sys-current-active-servers)</metric>
    <metric>(sys-current-active-svr-ops-util)</metric>
    <metric>(sys-current-active-svr-mem-util)</metric>
    <metric>(sys-current-active-svr-net-util)</metric>
    <enumeratedValueSet variable="datacenter-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-cpu-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-submission-strategy">
      <value value="&quot;closest&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-history-length">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="net-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-model">
      <value value="&quot;[1 2 3 4 5 6 7 8]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-migr-move?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-trace?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-standby-strategy">
      <value value="&quot;all-off&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="use-sla-local-factor?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="consolidation-interval">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-services">
      <value value="400"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration-strategy">
      <value value="&quot;least-migration-time&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-history-length">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cpu-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-model-method">
      <value value="&quot;stepwise-simple-linear-regression&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-generation-speed">
      <value value="16"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scheduler-queue-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-mem-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="auto-migration?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="power-estimation-method">
      <value value="&quot;median&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-space">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time-unit">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-placement-algorithm">
      <value value="&quot;first-fit&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-consolidation-strategy">
      <value value="&quot;within-datacenter&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rack-level-heterogeneity?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="staged-evaluation">
      <value value="&quot;3&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-access-ratio">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="total-racks">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="service-lifetime">
      <value value="&quot;[144 576]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="server-net-utilization-threshold">
      <value value="&quot;[20 90]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mem-usage-dist">
      <value value="&quot;[2 4]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rand-seed">
      <value value="33723175"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
