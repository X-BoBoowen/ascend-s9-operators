$ErrorActionPreference = 'Stop'

$workspace = 'D:\29722\Desktop\GCC'
$referenceDir = Join-Path $workspace '参考材料（往届优秀代码）'
$packageDir = Join-Path $workspace '提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$indexDir = Join-Path $packageDir '00_材料索引'
$workbookDir = Join-Path $packageDir '04_往届赛题与代码分析\赛题工作簿'

New-Item -ItemType Directory -Path $indexDir -Force | Out-Null
New-Item -ItemType Directory -Path $workbookDir -Force | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-DocxText {
    param([Parameter(Mandatory)][string]$Path)
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $entry = $archive.GetEntry('word/document.xml')
        if ($null -eq $entry) { return '' }
        $reader = [IO.StreamReader]::new($entry.Open())
        try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $ns = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
        $ns.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
        $lines = foreach ($paragraph in $xml.SelectNodes('//w:p', $ns)) {
            (($paragraph.SelectNodes('.//w:t', $ns) | ForEach-Object { $_.InnerText }) -join '')
        }
        return ($lines -join "`n")
    } finally {
        $archive.Dispose()
    }
}

function Normalize-Url {
    param([Parameter(Mandatory)][string]$Url)
    $value = $Url.Replace(([char]0x00A0).ToString(), '').Replace(([char]0x2002).ToString(), '').Trim()
    if ($value -match '^github\.com/') { $value = "https://$value" }
    return $value.TrimEnd('。', '，', ',', ';', ')', ']', '}', ' ')
}

function Get-CloneUrl {
    param([Parameter(Mandatory)][string]$Url)
    $value = Normalize-Url $Url
    if ($value -notmatch '^https://(github\.com|gitee\.com|gitcode\.com)/') { return $null }
    $uri = [Uri]$value
    $segments = $uri.AbsolutePath.Trim('/').Split('/')
    if ($segments.Count -lt 2) { return $null }
    $repo = $segments[1] -replace '\.git$', ''
    if ([string]::IsNullOrWhiteSpace($repo)) { return $null }
    return "$($uri.Scheme)://$($uri.Host)/$($segments[0])/$repo.git"
}

$sourceFiles = @()
$sourceFiles += Get-ChildItem -LiteralPath $referenceDir -Recurse -Force -File
$sourceFiles += Get-ChildItem -LiteralPath (Join-Path $workspace '榜单') -Recurse -Force -File
$sourceFiles += Get-Item -LiteralPath (Join-Path $workspace '华为云Ascend C算子开发环境搭建手册.docx')
$sourceFiles += Get-Item -LiteralPath (Join-Path $workspace '算子挑战赛（S9赛季）云资源代金券申请指南.pdf')
$sourceFiles += Get-Item -LiteralPath (Join-Path $workspace '算子挑战赛S9赛题.zip')
$sourceFiles += Get-Item -LiteralPath (Join-Path $workspace '提交相关材料\提交说明文档.docx')

$inventory = foreach ($file in ($sourceFiles | Sort-Object FullName -Unique)) {
    $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
    [pscustomobject]@{
        SourcePath = $file.FullName
        FileName = $file.Name
        Extension = $file.Extension
        Bytes = $file.Length
        LastWriteTime = $file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        SHA256 = $hash.Hash
        IncludedInPackage = if ($file.Name -eq 'GCC.pem') { 'No-sensitive' } else { 'Referenced-or-copied' }
    }
}
$inventory | Export-Csv -LiteralPath (Join-Path $indexDir '源材料文件清单.csv') -NoTypeInformation -Encoding utf8BOM

$winnerDoc = Join-Path $referenceDir '往届冠军开源项目.docx'
$winnerText = Get-DocxText -Path $winnerDoc
$urlPattern = '(?i)(?:https?://|github\.com/)[^\s]+'
$winnerUrls = [regex]::Matches($winnerText, $urlPattern) |
    ForEach-Object { Normalize-Url $_.Value } |
    Sort-Object -Unique

$urlRows = foreach ($url in $winnerUrls) {
    $cloneUrl = Get-CloneUrl -Url $url
    [pscustomobject]@{
        Source = '往届冠军开源项目.docx'
        OriginalUrl = $url
        Category = if ($cloneUrl) { 'Repository' } else { 'WebPage' }
        NormalizedCloneUrl = $cloneUrl
        Status = 'Pending'
        Note = ''
    }
}

$s9Urls = @(
    'https://docs.pytorch.org/docs/2.5/generated/torch.cat.html#torch.cat',
    'https://docs.pytorch.org/docs/2.5/generated/torch.gt.html#torch.gt',
    'https://docs.pytorch.org/docs/2.5/generated/torch.index_add.html#torch-index-add',
    'https://docs.pytorch.org/docs/2.5/generated/torch.permute.html#torch-permute',
    'https://docs.pytorch.org/docs/2.5/generated/torch.sum.html#torch.sum',
    'https://docs.pytorch.org/docs/2.5/generated/torch.square.html#torch-square'
)
$urlRows += foreach ($url in $s9Urls) {
    [pscustomobject]@{
        Source = 'S9挑战性能赛题说明.xlsx'
        OriginalUrl = $url
        Category = 'Specification'
        NormalizedCloneUrl = ''
        Status = 'Pending'
        Note = ''
    }
}
$urlRows | Sort-Object Source, OriginalUrl | Export-Csv -LiteralPath (Join-Path $indexDir '外部链接总表_初始.csv') -NoTypeInformation -Encoding utf8BOM

$archives = @()
$archives += Get-Item -LiteralPath (Join-Path $workspace '算子挑战赛S9赛题.zip')
$archives += Get-ChildItem -LiteralPath $referenceDir -Filter *.zip -File

$workbooks = foreach ($archiveFile in $archives) {
    $archive = [IO.Compression.ZipFile]::OpenRead($archiveFile.FullName)
    try {
        $xlsxEntries = $archive.Entries | Where-Object { $_.FullName -match '\.xlsx$' }
        $sequence = 0
        foreach ($entry in $xlsxEntries) {
            $sequence += 1
            $safeBase = [IO.Path]::GetFileNameWithoutExtension($archiveFile.Name)
            $target = Join-Path $workbookDir ("{0}_{1:D2}.xlsx" -f $safeBase, $sequence)
            $inputStream = $entry.Open()
            $outputStream = [IO.File]::Create($target)
            try { $inputStream.CopyTo($outputStream) } finally {
                $inputStream.Dispose()
                $outputStream.Dispose()
            }
            [pscustomobject]@{
                Archive = $archiveFile.FullName
                Entry = $entry.FullName
                ExtractedWorkbook = $target
                Bytes = $entry.Length
            }
        }
    } finally {
        $archive.Dispose()
    }
}
$workbooks | Export-Csv -LiteralPath (Join-Path $indexDir '赛题工作簿来源映射.csv') -NoTypeInformation -Encoding utf8BOM

[pscustomobject]@{
    SourceFiles = $inventory.Count
    WinnerDocumentUrls = $winnerUrls.Count
    RepositoryUrls = @($urlRows | Where-Object Category -eq 'Repository').Count
    WebPageUrls = @($urlRows | Where-Object Category -ne 'Repository').Count
    ExtractedWorkbooks = @($workbooks).Count
} | ConvertTo-Json
