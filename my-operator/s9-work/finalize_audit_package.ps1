$ErrorActionPreference = 'Stop'

$package = 'D:\29722\Desktop\GCC\提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$index = Join-Path $package '00_材料索引'
$complete = Import-Csv -LiteralPath (Join-Path $index '外部链接总表_完整.csv')
$web = @()
$web += Import-Csv -LiteralPath (Join-Path $index '网页获取结果.csv')
$web += Import-Csv -LiteralPath (Join-Path $index '网页获取结果_补充.csv')
$web += [pscustomobject]@{
    OriginalUrl = 'https://docs.pytorch.org/docs/2.5/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_'
    Status = 'Archived'
    HttpStatus = '200'
    ArchiveFile = 'S046_torch.Tensor.index_add_.html'
    FinalUrl = 'https://docs.pytorch.org/docs/2.5/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_'
    Title = 'torch.Tensor.index_add_ — PyTorch 2.5 documentation'
    Message = ''
}
$repos = Import-Csv -LiteralPath (Join-Path $index '仓库拉取结果.csv')

$webByUrl = @{}
foreach ($row in $web) { $webByUrl[$row.OriginalUrl] = $row }
$repoByUrl = @{}
foreach ($row in $repos) { $repoByUrl[$row.CloneUrl] = $row }

$final = foreach ($row in $complete) {
    $webRow = $webByUrl[$row.OriginalUrl]
    $repoRow = if ($row.NormalizedCloneUrl) { $repoByUrl[$row.NormalizedCloneUrl] } else { $null }
    $webStatus = if ($webRow) { $webRow.Status } else { 'NotApplicable' }
    $repoStatus = if ($repoRow) { $repoRow.Status } else { 'NotApplicable' }
    $overall = if ($repoStatus -in @('Cloned', 'AlreadyPresent')) {
        'RepositoryCloned'
    } elseif ($row.NormalizedCloneUrl -and $webStatus -eq 'Archived') {
        'PageArchived_CloneFailed'
    } elseif ($webStatus -eq 'Archived') {
        'PageArchived'
    } else {
        'Failed'
    }
    [pscustomobject]@{
        OriginalUrl = $row.OriginalUrl
        Sources = $row.Sources
        Category = $row.Category
        DiscoveryMethods = $row.DiscoveryMethods
        WebStatus = $webStatus
        HttpStatus = if ($webRow) { $webRow.HttpStatus } else { '' }
        WebArchiveFile = if ($webRow) { $webRow.ArchiveFile } else { '' }
        RepositoryStatus = $repoStatus
        RepositoryDirectory = if ($repoRow) { $repoRow.Directory } else { '' }
        RepositoryHead = if ($repoRow) { $repoRow.Head } else { '' }
        OverallStatus = $overall
        ErrorOrNote = (@(
            if ($webRow) { $webRow.Message }
            if ($repoRow) { $repoRow.Message }
        ) | Where-Object { $_ }) -join ' | '
    }
}
$final | Export-Csv -LiteralPath (Join-Path $index '外部链接总表_最终.csv') -NoTypeInformation -Encoding utf8BOM

$manifest = Get-ChildItem -LiteralPath $package -Recurse -Force -File |
    Where-Object { $_.FullName -notlike '*\00_材料索引\最终资料包文件清单.csv' } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [pscustomobject]@{
            RelativePath = $_.FullName.Substring($package.Length + 1)
            Bytes = $_.Length
            SHA256 = $hash.Hash
        }
    }
$manifest | Export-Csv -LiteralPath (Join-Path $index '最终资料包文件清单.csv') -NoTypeInformation -Encoding utf8BOM

$git = 'C:\Users\29722\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
$cloneRoot = Join-Path $package '03_往届冠军仓库快照'
$integrity = Get-ChildItem -LiteralPath $cloneRoot -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName '.git') } |
    ForEach-Object -Parallel {
        $repo = $_
        $git = $using:git
        $output = & $git -C $repo.FullName fsck --connectivity-only --no-progress 2>&1
        [pscustomobject]@{
            Directory = $repo.Name
            Status = if ($LASTEXITCODE -eq 0) { 'Pass' } else { 'Fail' }
            Message = ($output | Out-String).Trim()
        }
    } -ThrottleLimit 8
$integrity | Sort-Object Directory | Export-Csv -LiteralPath (Join-Path $index '仓库完整性检查.csv') -NoTypeInformation -Encoding utf8BOM

$checks = [ordered]@{
    UniqueLinks = $final.Count
    PageArchived = @($final | Where-Object WebStatus -eq 'Archived').Count
    PageFailed = @($final | Where-Object WebStatus -eq 'Failed').Count
    RepositoriesCloned = @($repos | Where-Object Status -in @('Cloned', 'AlreadyPresent')).Count
    RepositoriesFailed = @($repos | Where-Object Status -eq 'Failed').Count
    RepositoryIntegrityPass = @($integrity | Where-Object Status -eq 'Pass').Count
    RepositoryIntegrityFail = @($integrity | Where-Object Status -eq 'Fail').Count
    ManifestFiles = $manifest.Count
    SensitivePemFilesIncluded = @(Get-ChildItem -LiteralPath $package -Recurse -Force -File -Filter '*.pem').Count
    MainReportPdfExists = Test-Path -LiteralPath (Join-Path $package '06_报告与验收清单\S9资料全量审读与外部链接归档报告.pdf')
    MainReportDocxExists = Test-Path -LiteralPath (Join-Path $package '06_报告与验收清单\S9资料全量审读与外部链接归档报告.docx')
    AuditWorkbookExists = Test-Path -LiteralPath (Join-Path $package '06_报告与验收清单\S9外部链接与往届仓库审计表.xlsx')
}
$checks | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package '06_报告与验收清单\最终验收结果.json') -Encoding utf8
$checks | ConvertTo-Json
