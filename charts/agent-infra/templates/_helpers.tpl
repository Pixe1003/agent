{{/* Common helpers */}}

{{- define "agent-infra.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

{{- define "agent-infra.componentLabels" -}}
{{ include "agent-infra.labels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "agent-infra.image" -}}
{{- $registry := .root.Values.global.imageRegistry -}}
{{- $repo := .repo -}}
{{- $tag := default .root.Values.global.imageTag .tag -}}
{{- printf "%s%s:%s" $registry $repo $tag -}}
{{- end }}

{{- define "agent-infra.podSecurityContext" -}}
{{- if .Values.security.podSecurityContext }}
securityContext:
  {{- toYaml .Values.security.podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}
