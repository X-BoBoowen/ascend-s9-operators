param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('s03m', 's03n', 's03o')]
    [string]$Stage
)

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$temporaryRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.codex_tmp'))
$expectedPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) +
    [IO.Path]::DirectorySeparatorChar
if (-not $temporaryRoot.StartsWith(
        $expectedPrefix,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Temporary directory escaped the repository'
}

$token = [Guid]::NewGuid().ToString('N')
$download = Join-Path $temporaryRoot "artifact-$Stage-$token"
$backup = Join-Path $temporaryRoot "artifact-backup-$token"
$artifact = Join-Path $repoRoot 'artifact'
New-Item -ItemType Directory -Path $download -Force | Out-Null

$scp = (Get-Command scp.exe -ErrorAction Stop).Source
$key = 'D:\29722\Desktop\GCC\GCC.pem'
$remote = 'ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com'
$remoteArtifact = '/home/ma-user/work/s9/repository/ascend-s9-operators/artifact/.'

try {
    & $scp `
        -q `
        -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new `
        -P 32375 `
        -i $key `
        -r `
        "${remote}:$remoteArtifact" `
        $download
    if ($LASTEXITCODE -ne 0) {
        throw "Remote artifact download failed with exit code $LASTEXITCODE"
    }

    $resultPath = Join-Path $download 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw 'Downloaded artifact does not contain result.json'
    }
    $result = Get-Content -LiteralPath $resultPath -Raw |
        ConvertFrom-Json
    if ($result.stage -ne $Stage) {
        throw "Downloaded stage '$($result.stage)' does not match '$Stage'"
    }

    if (Test-Path -LiteralPath $artifact) {
        Move-Item -LiteralPath $artifact -Destination $backup
    }
    try {
        Move-Item -LiteralPath $download -Destination $artifact
    } catch {
        if (Test-Path -LiteralPath $backup) {
            Move-Item -LiteralPath $backup -Destination $artifact
        }
        throw
    }
    if (Test-Path -LiteralPath $backup) {
        Remove-Item -LiteralPath $backup -Recurse -Force
    }
    Write-Output "Latest $Stage artifact: $artifact"
} finally {
    if (Test-Path -LiteralPath $download) {
        Remove-Item -LiteralPath $download -Recurse -Force
    }
}
