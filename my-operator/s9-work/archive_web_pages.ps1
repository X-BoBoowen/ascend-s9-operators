$ErrorActionPreference = 'Stop'

$packageDir = 'D:\29722\Desktop\GCC\提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$inputCsv = Join-Path $packageDir '00_材料索引\外部链接总表_初始.csv'
$archiveRoot = Join-Path $packageDir '02_网页归档\原始网页'
$statusRoot = Join-Path $packageDir '00_材料索引\网页获取状态'
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
New-Item -ItemType Directory -Path $statusRoot -Force | Out-Null

$rows = Import-Csv -LiteralPath $inputCsv | Sort-Object Source, OriginalUrl
$jobs = for ($i = 0; $i -lt $rows.Count; $i++) {
    $row = $rows[$i]
    [pscustomobject]@{
        Index = $i + 1
        Source = $row.Source
        Category = $row.Category
        Url = $row.OriginalUrl
        FileName = ('W{0:D3}.html' -f ($i + 1))
    }
}
$jobs | Export-Csv -LiteralPath (Join-Path $packageDir '00_材料索引\网页归档目录映射.csv') -NoTypeInformation -Encoding utf8BOM

$jobs | ForEach-Object -Parallel {
    $job = $_
    $archiveRoot = $using:archiveRoot
    $statusRoot = $using:statusRoot
    $outputPath = Join-Path $archiveRoot $job.FileName
    $statusPath = Join-Path $statusRoot ('W{0:D3}.json' -f [int]$job.Index)
    $started = Get-Date
    $status = 'Failed'
    $httpStatus = ''
    $finalUrl = ''
    $title = ''
    $bytes = 0
    $message = ''
    try {
        $response = Invoke-WebRequest -Uri $job.Url -MaximumRedirection 8 -TimeoutSec 45 -Headers @{
            'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0 Safari/537.36'
            'Accept-Language' = 'zh-CN,zh;q=0.9,en;q=0.8'
        } -OutFile $outputPath -PassThru
        $httpStatus = [int]$response.StatusCode
        $finalUrl = $response.BaseResponse.RequestMessage.RequestUri.AbsoluteUri
        $bytes = (Get-Item -LiteralPath $outputPath).Length
        $html = Get-Content -LiteralPath $outputPath -Raw -ErrorAction SilentlyContinue
        if ($html -match '(?is)<title[^>]*>(.*?)</title>') {
            $title = [System.Net.WebUtility]::HtmlDecode(($matches[1] -replace '\s+', ' ').Trim())
        }
        $status = 'Archived'
    } catch {
        $message = $_.Exception.Message
        if ($_.Exception.Response) {
            try { $httpStatus = [int]$_.Exception.Response.StatusCode } catch {}
            try { $finalUrl = $_.Exception.Response.RequestMessage.RequestUri.AbsoluteUri } catch {}
        }
        if (Test-Path -LiteralPath $outputPath) {
            $bytes = (Get-Item -LiteralPath $outputPath).Length
            $message += ' | Partial response retained.'
        }
    }
    $finished = Get-Date
    [pscustomobject]@{
        Index = $job.Index
        Source = $job.Source
        Category = $job.Category
        OriginalUrl = $job.Url
        ArchiveFile = $job.FileName
        Status = $status
        HttpStatus = $httpStatus
        FinalUrl = $finalUrl
        Title = $title
        Bytes = $bytes
        RetrievedAt = $finished.ToString('yyyy-MM-dd HH:mm:ss')
        Seconds = [math]::Round(($finished - $started).TotalSeconds, 1)
        Message = $message
    } | ConvertTo-Json -Compress | Set-Content -LiteralPath $statusPath -Encoding utf8
} -ThrottleLimit 8

$results = Get-ChildItem -LiteralPath $statusRoot -Filter 'W*.json' -File |
    Sort-Object Name |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json }
$results | Export-Csv -LiteralPath (Join-Path $packageDir '00_材料索引\网页获取结果.csv') -NoTypeInformation -Encoding utf8BOM
$results | Group-Object Status | Select-Object Name, Count | ConvertTo-Json
