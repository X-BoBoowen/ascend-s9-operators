$ErrorActionPreference = 'Stop'

$packageDir = 'D:\29722\Desktop\GCC\提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$inputCsv = Join-Path $packageDir '00_材料索引\外部链接总表_初始.csv'
$cloneRoot = Join-Path $packageDir '03_往届冠军仓库快照'
$statusRoot = Join-Path $packageDir '00_材料索引\仓库拉取状态'
$git = 'C:\Users\29722\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'

New-Item -ItemType Directory -Path $cloneRoot -Force | Out-Null
New-Item -ItemType Directory -Path $statusRoot -Force | Out-Null

$repositories = Import-Csv -LiteralPath $inputCsv |
    Where-Object Category -eq 'Repository' |
    Sort-Object NormalizedCloneUrl -Unique

$jobs = for ($i = 0; $i -lt $repositories.Count; $i++) {
    $row = $repositories[$i]
    $uri = [Uri]$row.NormalizedCloneUrl
    $segments = $uri.AbsolutePath.Trim('/').Split('/')
    $owner = $segments[0] -replace '[^A-Za-z0-9_.-]', '_'
    $repo = ($segments[1] -replace '\.git$', '') -replace '[^A-Za-z0-9_.-]', '_'
    [pscustomobject]@{
        Index = $i + 1
        OriginalUrl = $row.OriginalUrl
        CloneUrl = $row.NormalizedCloneUrl
        Directory = ('R{0:D3}_{1}_{2}_{3}' -f ($i + 1), $uri.Host.Split('.')[0], $owner, $repo)
    }
}

$jobs | Export-Csv -LiteralPath (Join-Path $packageDir '00_材料索引\仓库目录映射.csv') -NoTypeInformation -Encoding utf8BOM

$jobs | ForEach-Object -Parallel {
    $job = $_
    $cloneRoot = $using:cloneRoot
    $statusRoot = $using:statusRoot
    $git = $using:git
    $destination = Join-Path $cloneRoot $job.Directory
    $statusFile = Join-Path $statusRoot ('R{0:D3}.json' -f [int]$job.Index)
    $env:GIT_TERMINAL_PROMPT = '0'
    $env:GCM_INTERACTIVE = 'Never'
    $started = Get-Date
    $status = 'Failed'
    $message = ''
    $head = ''
    $branch = ''
    $commitDate = ''
    try {
        if (Test-Path -LiteralPath (Join-Path $destination '.git')) {
            $status = 'AlreadyPresent'
        } else {
            $output = & $git -c core.longpaths=true -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 clone --quiet -- $job.CloneUrl $destination 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw (($output | Out-String).Trim())
            }
            $status = 'Cloned'
        }
        if (Test-Path -LiteralPath (Join-Path $destination '.git')) {
            $head = (& $git -C $destination rev-parse HEAD 2>$null | Select-Object -First 1)
            $branch = (& $git -C $destination branch --show-current 2>$null | Select-Object -First 1)
            $commitDate = (& $git -C $destination show -s --format=%cI HEAD 2>$null | Select-Object -First 1)
        }
    } catch {
        $message = $_.Exception.Message
        if (Test-Path -LiteralPath $destination) {
            $message += ' | Partial directory retained for evidence.'
        }
    }
    $finished = Get-Date
    [pscustomobject]@{
        Index = $job.Index
        OriginalUrl = $job.OriginalUrl
        CloneUrl = $job.CloneUrl
        Directory = $job.Directory
        Status = $status
        Head = $head
        Branch = $branch
        CommitDate = $commitDate
        Started = $started.ToString('yyyy-MM-dd HH:mm:ss')
        Finished = $finished.ToString('yyyy-MM-dd HH:mm:ss')
        Seconds = [math]::Round(($finished - $started).TotalSeconds, 1)
        Message = $message
    } | ConvertTo-Json -Compress | Set-Content -LiteralPath $statusFile -Encoding utf8
} -ThrottleLimit 8

$results = Get-ChildItem -LiteralPath $statusRoot -Filter 'R*.json' -File |
    Sort-Object Name |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json }
$results | Export-Csv -LiteralPath (Join-Path $packageDir '00_材料索引\仓库拉取结果.csv') -NoTypeInformation -Encoding utf8BOM
$results | Group-Object Status | Select-Object Name, Count | ConvertTo-Json
