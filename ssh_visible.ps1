param(
    [Parameter(Mandatory = $true)]
    [string]$Command,
    [int]$ConnectTimeoutSeconds = 15
)

$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$artifactDir = [IO.Path]::GetFullPath(
    (Join-Path $projectRoot 'artifact'))
$expectedArtifactDir = $projectRoot.TrimEnd(
    [IO.Path]::DirectorySeparatorChar) +
    [IO.Path]::DirectorySeparatorChar
if (-not $artifactDir.StartsWith(
        $expectedArtifactDir,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use artifact directory outside project root: $artifactDir"
}

New-Item -ItemType Directory -Path $artifactDir -Force |
    Out-Null
Get-ChildItem -LiteralPath $artifactDir -Force |
    Remove-Item -Recurse -Force

$runLogPath = Join-Path $artifactDir 'run.log'
$resultPath = Join-Path $artifactDir 'result.json'
$runLogTemp = Join-Path $artifactDir (
    '.run.log.{0}.{1}.tmp' -f $PID, [Guid]::NewGuid().ToString('N'))
$resultTemp = Join-Path $artifactDir (
    '.result.json.{0}.{1}.tmp' -f $PID, [Guid]::NewGuid().ToString('N'))
$startedAt = [DateTimeOffset]::Now

$ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
$key = 'D:\29722\Desktop\GCC\GCC.pem'
$hostName = 'dev-modelarts.cn-southwest-2.huaweicloud.com'
$port = 32375
$user = 'ma-user'
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command))
$remote = @"
mkdir -p /home/ma-user/work/s9
log=/home/ma-user/work/s9/codex-visible-terminal.log
cmd=`$(printf '%s' '$encoded' | base64 -d)
printf '\n[%s] `$ %s\n' "`$(date '+%F %T %z')" "`$cmd" | tee -a "`$log"
bash -lc "`$cmd" 2>&1 | tee -a "`$log"
exit `${PIPESTATUS[0]}
"@

$outputLines = @(
    & $ssh `
        -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new `
        -o "ConnectTimeout=$ConnectTimeoutSeconds" `
        -i $key `
        -p $port `
        "$user@$hostName" `
        $remote 2>&1
)
$exitCode = $LASTEXITCODE
$finishedAt = [DateTimeOffset]::Now

$logText = if ($outputLines.Count -eq 0) {
    ''
} else {
    ($outputLines | ForEach-Object { "$_".TrimEnd() }) -join
        [Environment]::NewLine
}
if ($logText.Length -gt 0) {
    $logText += [Environment]::NewLine
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($runLogTemp, $logText, $utf8)
Move-Item -LiteralPath $runLogTemp -Destination $runLogPath -Force

$result = [ordered]@{
    command = $Command
    remote = "$user@$hostName`:$port"
    started_at = $startedAt.ToString('o')
    finished_at = $finishedAt.ToString('o')
    duration_ms = [Math]::Round(
        ($finishedAt - $startedAt).TotalMilliseconds,
        3)
    exit_code = $exitCode
    success = ($exitCode -eq 0)
    log = 'artifact/run.log'
}
$json = $result | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText(
    $resultTemp,
    $json + [Environment]::NewLine,
    $utf8)
Move-Item -LiteralPath $resultTemp -Destination $resultPath -Force

$outputLines | ForEach-Object { Write-Output $_ }
exit $exitCode
