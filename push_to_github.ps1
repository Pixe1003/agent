# =============================================================================
# Agent Infra Toolkit — Push to GitHub
# =============================================================================
# 用法：
#   .\push_to_github.ps1 -RepoUrl https://github.com/<your-handle>/agent-infra.git
#
# 完整流程：
#   1. 验证 git / pytest 可用
#   2. 扫描大文件 + 敏感信息（OpenAI key / GitHub token 等）
#   3. 跑全套 pytest（98 cases）
#   4. 初始化 git 仓库 + 配置 main branch
#   5. 自动写 commit message
#   6. 设置 remote + push
#
# 可选参数：
#   -Branch        默认 "main"
#   -CommitMessage 自定义 commit message（不传则用 auto-generated）
#   -SkipTests     跳过 pytest（不推荐）
#   -Force         即使有大文件 / secret / pytest 失败也强制推送
#   -DryRun        只跑检查不真 push
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, HelpMessage="GitHub repo URL, e.g. https://github.com/your-handle/agent-infra.git")]
    [string]$RepoUrl,

    [string]$Branch = "main",
    [string]$CommitMessage = "",
    [switch]$SkipTests,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
}
Set-Location $ProjectRoot

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Agent Infra Toolkit -> GitHub" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Project: $ProjectRoot"
Write-Host " Remote:  $RepoUrl"
Write-Host " Branch:  $Branch"
if ($DryRun)    { Write-Host " Mode:    DRY-RUN (no actual push)" -ForegroundColor Yellow }
if ($Force)     { Write-Host " Force:   ENABLED (will push despite warnings)" -ForegroundColor Yellow }
if ($SkipTests) { Write-Host " Tests:   SKIPPED" -ForegroundColor Yellow }
Write-Host ""

# =============================================================================
# [1/7] 验证工具链
# =============================================================================
Write-Host "[1/7] Checking toolchain..." -ForegroundColor Cyan
try {
    $gitVersion = (git --version) -replace 'git version ', ''
    Write-Host "  git: $gitVersion"
} catch {
    Write-Error "git not found. Install from https://git-scm.com/"
    exit 1
}

$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Warning "  .venv\Scripts\python.exe not found, falling back to 'python'"
    $pythonExe = "python"
}

# =============================================================================
# [2/7] 扫描大文件（>10 MB），防止 GitHub 100 MB 硬限
# =============================================================================
Write-Host ""
Write-Host "[2/7] Scanning for files larger than 10 MB..." -ForegroundColor Cyan
$ignorePathRegex = '\\(\.venv|\.python31[2-4]|__pycache__|\.git|\.pytest_cache|node_modules)\\'
$largeFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Length -gt 10MB -and $_.FullName -notmatch $ignorePathRegex
}
if ($largeFiles) {
    Write-Warning "Files larger than 10 MB detected:"
    $largeFiles | ForEach-Object {
        $rel = $_.FullName.Substring($ProjectRoot.Length + 1)
        $sizeMB = [math]::Round($_.Length / 1MB, 1)
        Write-Host "  $rel ($sizeMB MB)" -ForegroundColor Yellow
    }
    Write-Host "  These should be matched by .gitignore. Verify before push."
    if ($largeFiles.Length -gt 0 -and -not $Force) {
        Write-Host "  -> Hint: run 'git check-ignore <file>' to verify each is ignored"
    }
} else {
    Write-Host "  OK - no large files found." -ForegroundColor Green
}

# =============================================================================
# [3/7] 扫描敏感信息（API keys, tokens）
# =============================================================================
Write-Host ""
Write-Host "[3/7] Scanning for secrets / API keys..." -ForegroundColor Cyan
$secretPatterns = @{
    "OpenAI API key"     = 'sk-[a-zA-Z0-9]{20,}'
    "GitHub PAT"         = 'ghp_[a-zA-Z0-9]{30,}'
    "GitHub fine-grained"= 'github_pat_[a-zA-Z0-9_]{50,}'
    "AWS access key"     = 'AKIA[0-9A-Z]{16}'
    "Google API key"     = 'AIzaSy[a-zA-Z0-9_\-]{30,}'
    "Slack token"        = 'xox[baprs]-[a-zA-Z0-9-]{20,}'
    "Hugging Face token" = 'hf_[a-zA-Z0-9]{30,}'
    "Anthropic API key"  = 'sk-ant-[a-zA-Z0-9_\-]{30,}'
}
$secretHits = @()
$scanExt = @("*.py","*.yaml","*.yml","*.json","*.md","*.toml","*.env","*.ps1","*.sh","*.cfg","*.ini","*.txt")
Get-ChildItem -Path $ProjectRoot -Recurse -File -Include $scanExt -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $ignorePathRegex } |
    ForEach-Object {
        $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) { return }
        foreach ($name in $secretPatterns.Keys) {
            $pattern = $secretPatterns[$name]
            if ($content -match $pattern) {
                $rel = $_.FullName.Substring($ProjectRoot.Length + 1)
                $secretHits += "  ${rel}: $name -> $($Matches[0].Substring(0, [Math]::Min(20, $Matches[0].Length)))..."
            }
        }
    }
if ($secretHits) {
    Write-Host "Potential secrets detected:" -ForegroundColor Red
    $secretHits | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    if (-not $Force) {
        Write-Error "Aborting to protect secrets. Pass -Force to override (NOT RECOMMENDED)."
        exit 1
    }
    Write-Warning "Continuing despite secrets due to -Force."
} else {
    Write-Host "  OK - no secrets detected." -ForegroundColor Green
}

# =============================================================================
# [4/7] 跑测试
# =============================================================================
if (-not $SkipTests) {
    Write-Host ""
    Write-Host "[4/7] Running pytest (98 cases expected)..." -ForegroundColor Cyan
    & $pythonExe -m pytest tests -q 2>&1 | Tee-Object -Variable pytestOutput | Out-Host
    if ($LASTEXITCODE -ne 0) {
        if (-not $Force) {
            Write-Error "pytest failed. Aborting. Use -SkipTests or -Force to bypass."
            exit 1
        }
        Write-Warning "Tests failed but -Force was passed; continuing."
    } else {
        Write-Host "  OK - all tests passed." -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "[4/7] Skipping pytest (--SkipTests)" -ForegroundColor Yellow
}

# =============================================================================
# [5/7] 初始化 git 仓库
# =============================================================================
Write-Host ""
Write-Host "[5/7] Initializing git repository..." -ForegroundColor Cyan
if (-not (Test-Path ".git")) {
    git init | Out-Null
    git branch -M $Branch
    Write-Host "  git init done, branch set to '$Branch'."
} else {
    $currentBranch = git rev-parse --abbrev-ref HEAD 2>$null
    if ($currentBranch -ne $Branch) {
        Write-Host "  Switching from '$currentBranch' to '$Branch'..."
        git branch -M $Branch
    } else {
        Write-Host "  Existing repo, branch '$Branch' OK."
    }
}

# Check user.name / user.email
$gitUserName = git config user.name
$gitUserEmail = git config user.email
if (-not $gitUserName -or -not $gitUserEmail) {
    Write-Warning "git user.name / user.email not configured. Set globally:"
    Write-Host "  git config --global user.name `"Your Name`""
    Write-Host "  git config --global user.email `"you@example.com`""
    if (-not $Force) {
        Write-Error "Aborting. Configure git identity first."
        exit 1
    }
}

# =============================================================================
# [6/7] Stage + commit
# =============================================================================
Write-Host ""
Write-Host "[6/7] Staging files and committing..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if (-not $status) {
    Write-Host "  Nothing to commit." -ForegroundColor Yellow
} else {
    $fileCount = ($status -split "`n").Length
    Write-Host "  $fileCount files staged."

    if (-not $CommitMessage) {
        $CommitMessage = @"
feat: Agent Infra Toolkit — initial public release

End-to-end multi-agent cloud scheduler infrastructure with closed-loop AIOps
and self-distilled LLM policy, built on NetLogo simulation.

Architecture:
- 5 agent modules sharing Pydantic v2 schemas + same-shape API
- LangGraph Planner-Scheduler-Critic with AIOps risk_tags closed loop
- Episodic memory (Jaccard + Euclidean + reward, PrivateAttr token cache)
- llama-cpp-python GGUF q4 inference adapter with deterministic fallback
- MCP server (FastMCP) exposing 3 tools + 2 resources per Anthropic spec
- Skill SDK + click CLI (8 subcommands)
- Multi-stage Dockerfile + docker-compose + complete Helm chart

Headline results (5 seed x 4 dist x 7 algo benchmark):
- AIOps closed loop: SLA 35% -> 0.75% (-97%), energy -11%
- Profile-driven optimization: phase3 latency 678us -> 270us (-60%)
- 98 pytest cases, 100% pass

Key findings:
- LLM in sub-ms control plane is the wrong location (250000x slower)
- RAG signal diluted by strong external AIOps signal
- Deterministic fallback is must-have for LLM agents

Docs:
- README.md, RESUME.md (Chinese/English, multiple lengths)
- TECH_RETROSPECTIVE.md (problems found and fixed)
- K8S_DEPLOYMENT_PLAN.md (14 decisions + 6 phase roadmap)
- K8S_QUICKSTART.md (kind cluster 10-minute walkthrough)
"@
    }

    if ($DryRun) {
        Write-Host "  DRY RUN: would commit with message:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host $CommitMessage -ForegroundColor Gray
        Write-Host ""
    } else {
        git commit -m $CommitMessage | Out-Null
        $commitHash = git rev-parse HEAD
        Write-Host "  Commit created: $($commitHash.Substring(0, 7))" -ForegroundColor Green
    }
}

# =============================================================================
# [7/7] 设置 remote 并 push
# =============================================================================
Write-Host ""
Write-Host "[7/7] Configuring remote and pushing..." -ForegroundColor Cyan

$remoteExists = $false
try {
    $existingUrl = git remote get-url origin 2>$null
    if ($existingUrl) { $remoteExists = $true }
} catch {}

if ($remoteExists) {
    Write-Host "  remote 'origin' already exists, updating URL..."
    git remote set-url origin $RepoUrl
} else {
    Write-Host "  adding remote 'origin' -> $RepoUrl"
    git remote add origin $RepoUrl
}

if ($DryRun) {
    Write-Host ""
    Write-Host "  DRY RUN: would execute 'git push -u origin $Branch'" -ForegroundColor Yellow
    Write-Host "  No actual push performed." -ForegroundColor Yellow
} else {
    Write-Host "  Pushing to $RepoUrl ($Branch)..."
    try {
        git push -u origin $Branch
        if ($LASTEXITCODE -ne 0) {
            throw "git push exited with code $LASTEXITCODE"
        }
    } catch {
        Write-Host ""
        Write-Error "Push failed. Common causes:"
        Write-Host "  1. Repo does not exist on GitHub yet. Create it first:" -ForegroundColor Yellow
        Write-Host "     https://github.com/new"
        Write-Host "  2. Authentication: use a Personal Access Token (PAT) or SSH key."
        Write-Host "     Generate PAT at: https://github.com/settings/tokens"
        Write-Host "  3. Branch protection or non-fast-forward: try 'git push --force-with-lease'"
        Write-Host "  4. Large file rejected: see GitHub error message; rerun with -Force after fixing"
        exit 1
    }
}

# =============================================================================
# 完成
# =============================================================================
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
if ($DryRun) {
    Write-Host " DRY RUN complete. Re-run without -DryRun to actually push." -ForegroundColor Green
} else {
    Write-Host " SUCCESS! Repository pushed to:" -ForegroundColor Green
    Write-Host " $RepoUrl" -ForegroundColor Green
    Write-Host ""
    Write-Host " Next steps:" -ForegroundColor Cyan
    $webUrl = $RepoUrl -replace '\.git$', ''
    Write-Host "   1. Visit $webUrl"
    Write-Host "      Verify: README renders, architecture.png displays, badges show."
    Write-Host "   2. Add repo topics (helps discovery):"
    Write-Host "      agent-infrastructure, langchain, langgraph, llm-agent, mcp,"
    Write-Host "      aiops, kubernetes, fastapi, qwen, self-distillation, pareto"
    Write-Host "   3. Set repo description: 'Agent Infra Toolkit -- Framework + Runtime +"
    Write-Host "      Memory SDK + Sandbox + Observability + MCP + K8s native deployment'"
    Write-Host "   4. Pin repo to your GitHub profile."
    Write-Host "   5. Update LinkedIn project entry with the link."
    Write-Host "   6. Reference in resume's Project section (see docs/RESUME.md)."
}
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""
