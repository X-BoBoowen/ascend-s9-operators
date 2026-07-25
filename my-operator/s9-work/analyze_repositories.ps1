$ErrorActionPreference = 'Stop'

$packageDir = 'D:\29722\Desktop\GCC\提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$cloneRoot = Join-Path $packageDir '03_往届冠军仓库快照'
$resultCsv = Join-Path $packageDir '04_往届赛题与代码分析\已拉取仓库代码审读矩阵.csv'
$statusCsv = Join-Path $packageDir '00_材料索引\仓库拉取结果.csv'

$status = Import-Csv -LiteralPath $statusCsv | Where-Object Status -in @('Cloned', 'AlreadyPresent')
$patterns = [ordered]@{
    DataCopy = '\bDataCopy\s*\('
    DataCopyPad = '\bDataCopyPad\s*\('
    DataCopyExtParams = '\bDataCopyExtParams\b'
    TPipe = '\bTPipe\b'
    TQue = '\bTQue\b'
    TBuf = '\bTBuf\b'
    PipeBarrier = '\bPipeBarrier\s*\('
    SetAtomicAdd = '\bSetAtomicAdd\s*\('
    SetVectorMask = '\bSetVectorMask\b'
    WholeReduceSum = '\bWholeReduceSum\s*\('
    ReduceSum = '\bReduceSum\s*\('
    Compare = '\bCompare\s*\('
    Select = '\bSelect\s*\('
    Gather = '\bGather(?:Mask)?\s*\('
    Scatter = '\bScatter\s*\('
    Transpose = '\bTranspose\s*\('
    Cast = '\bCast\s*\('
    Matmul = '\bMatmul\b|\bMmad\s*\('
    Tiling = '\bTilingData\b|\bTilingFunc\b|REGISTER_TILING_DATA_CLASS'
    Template = '\btemplate\s*<'
    BufferNum = '\bBUFFER_NUM\b|\bbufferNum\b'
    BlockIndex = '\bGetBlockIdx\s*\('
    Alignment = '\bALIGN(?:MENT)?\b|\balign(?:ed|ment)?\b'
}

$rows = foreach ($repo in $status) {
    $repoPath = Join-Path $cloneRoot $repo.Directory
    $files = Get-ChildItem -LiteralPath $repoPath -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notmatch '\\\.git\\' -and
            $_.FullName -notmatch '\\(build|build_out|output|out|install|cmake-build[^\\]*)\\' -and
            $_.Extension -in @('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.py', '.md', '.txt', '.json')
        }
    $codeFiles = $files | Where-Object Extension -in @('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp')
    $text = foreach ($file in $codeFiles) {
        try { Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop } catch { '' }
    }
    $joined = $text -join "`n"
    $readme = $files | Where-Object { $_.Name -match '^README(?:\..+)?$' } | Select-Object -First 1
    $readmeTitle = ''
    if ($readme) {
        $readmeText = Get-Content -LiteralPath $readme.FullName -ErrorAction SilentlyContinue
        $readmeTitle = ($readmeText | Where-Object { $_ -match '\S' } | Select-Object -First 1) -replace '^#+\s*', ''
    }
    $topLevel = Get-ChildItem -LiteralPath $repoPath -Force -ErrorAction SilentlyContinue |
        Where-Object Name -ne '.git' |
        Select-Object -ExpandProperty Name
    $operatorHints = $files |
        Where-Object { $_.FullName -match '\\(op_host|op_kernel|kernel|host)\\' } |
        ForEach-Object {
            $relative = $_.FullName.Substring($repoPath.Length).TrimStart('\')
            $parts = $relative.Split('\')
            $marker = [Array]::FindIndex($parts, [Predicate[string]]{ param($x) $x -in @('op_host', 'op_kernel', 'kernel', 'host') })
            if ($marker -gt 0) { $parts[$marker - 1] }
        } |
        Where-Object { $_ -and $_ -notmatch '^(src|operator|operators|ascendc_op|custom_op)$' } |
        Sort-Object -Unique
    $record = [ordered]@{
        Index = $repo.Index
        CloneUrl = $repo.CloneUrl
        LocalDirectory = $repo.Directory
        Branch = $repo.Branch
        Head = $repo.Head
        CommitDate = $repo.CommitDate
        READMEFirstLine = $readmeTitle
        TopLevelEntries = ($topLevel -join ' | ')
        OperatorHints = ($operatorHints -join ' | ')
        IndexedFiles = @($files).Count
        CodeFiles = @($codeFiles).Count
        CodeBytes = [int64](($codeFiles | Measure-Object Length -Sum).Sum)
    }
    foreach ($name in $patterns.Keys) {
        $record[$name] = [regex]::Matches($joined, $patterns[$name], 'IgnoreCase').Count
    }
    [pscustomobject]$record
}

$rows | Sort-Object {[int]$_.Index} | Export-Csv -LiteralPath $resultCsv -NoTypeInformation -Encoding utf8BOM
[pscustomobject]@{
    RepositoriesAnalyzed = @($rows).Count
    IndexedFiles = ($rows | Measure-Object IndexedFiles -Sum).Sum
    CodeFiles = ($rows | Measure-Object CodeFiles -Sum).Sum
    RepositoriesUsingDataCopyPad = @($rows | Where-Object {[int]$_.DataCopyPad -gt 0}).Count
    RepositoriesUsingAtomicAdd = @($rows | Where-Object {[int]$_.SetAtomicAdd -gt 0}).Count
    RepositoriesUsingReduction = @($rows | Where-Object {[int]$_.ReduceSum -gt 0 -or [int]$_.WholeReduceSum -gt 0}).Count
    RepositoriesUsingTemplates = @($rows | Where-Object {[int]$_.Template -gt 0}).Count
} | ConvertTo-Json
