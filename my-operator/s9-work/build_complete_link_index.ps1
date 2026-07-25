$ErrorActionPreference = 'Stop'

$workspace = 'D:\29722\Desktop\GCC'
$packageDir = Join-Path $workspace '提交相关材料\S9全量资料审计与网页仓库归档_20260725'
$initialCsv = Join-Path $packageDir '00_材料索引\外部链接总表_初始.csv'
$workbookIndex = Join-Path $packageDir '04_往届赛题与代码分析\工作簿审读结果\工作簿审读索引.json'

$rows = [System.Collections.Generic.List[object]]::new()
foreach ($row in (Import-Csv -LiteralPath $initialCsv)) {
    $rows.Add([pscustomobject]@{
        Source = $row.Source
        OriginalUrl = $row.OriginalUrl
        Category = $row.Category
        NormalizedCloneUrl = $row.NormalizedCloneUrl
        DiscoveryMethod = 'Visible text / S9 workbook'
    })
}

$supplemental = @(
    @('评分规则.docx；提交说明文档.docx', 'https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/resource/s4%E6%96%87%E6%A1%A3/zip_op.sh', 'SubmissionTool', 'Word HYPERLINK field'),
    @('评分规则.docx；提交说明文档.docx', 'https://www.hiascend.com/developer/contests/details/_blank', 'ContestPage', 'Word HYPERLINK field'),
    @('往届冠军开源项目.docx', 'https://www.hiascend.com/developer/contests/details/201bfd6701c14f3bb40ae53e11abd6e8', 'ContestPage', 'Word HYPERLINK field'),
    @('往届冠军开源项目.docx', 'https://www.hiascend.com/developer/contests/details/4252afe2506e4f6eaa5cdf6696afadbf', 'ContestPage', 'Word HYPERLINK field'),
    @('往届冠军开源项目.docx', 'https://www.hiascend.com/developer/group/01102182523402175016/_blank', 'CommunityPage', 'Word HYPERLINK field'),
    @('华为云Ascend C算子开发环境搭建手册.docx', 'http://e.huawei.com/', 'WebPage', 'Visible text / Word HYPERLINK field'),
    @('华为云Ascend C算子开发环境搭建手册.docx', 'https://console.huaweicloud.com/modelarts/?region=cn-north-4', 'CloudConsole', 'Word HYPERLINK field'),
    @('华为云Ascend C算子开发环境搭建手册.docx', 'https://www.huaweicloud.com/product/modelarts.html', 'WebPage', 'Visible text / Word HYPERLINK field'),
    @('算子挑战赛（S9赛季）云资源代金券申请指南.pdf', 'https://edu.hicomputing.huawei.com/', 'VoucherPage', 'PDF text/annotation'),
    @('算子挑战赛（S9赛季）云资源代金券申请指南.pdf', 'https://console.huaweicloud.com/iam/?region=cn-southwest-2&locale=zh-cn#/mine/apiCredential', 'CloudConsole', 'PDF text/annotation'),
    @('算子挑战赛（S9赛季）云资源代金券申请指南.pdf', 'https://edu.hicomputing.huawei.com/teaching/voucher-details/invite/202603273nttxWIz', 'VoucherPage', 'PDF annotation')
)
foreach ($item in $supplemental) {
    $rows.Add([pscustomobject]@{
        Source = $item[0]
        OriginalUrl = $item[1]
        Category = $item[2]
        NormalizedCloneUrl = ''
        DiscoveryMethod = $item[3]
    })
}

$workbooks = Get-Content -LiteralPath $workbookIndex -Raw | ConvertFrom-Json
foreach ($workbook in $workbooks) {
    foreach ($url in $workbook.urls) {
        $rows.Add([pscustomobject]@{
            Source = $workbook.fileName
            OriginalUrl = $url
            Category = 'Specification'
            NormalizedCloneUrl = ''
            DiscoveryMethod = 'artifact-tool workbook inspection'
        })
    }
}

$complete = $rows |
    Group-Object OriginalUrl |
    ForEach-Object {
        $group = $_.Group
        [pscustomobject]@{
            OriginalUrl = $_.Name
            Sources = ($group.Source | Sort-Object -Unique) -join '；'
            Category = ($group.Category | Sort-Object -Unique) -join '；'
            NormalizedCloneUrl = ($group.NormalizedCloneUrl | Where-Object { $_ } | Sort-Object -Unique) -join '；'
            DiscoveryMethods = ($group.DiscoveryMethod | Sort-Object -Unique) -join '；'
            AuditStatus = 'Pending merge with retrieval results'
        }
    } |
    Sort-Object OriginalUrl

$complete | Export-Csv -LiteralPath (Join-Path $packageDir '00_材料索引\外部链接总表_完整.csv') -NoTypeInformation -Encoding utf8BOM
[pscustomobject]@{
    UniqueLinks = $complete.Count
    Repositories = @($complete | Where-Object NormalizedCloneUrl).Count
    SpecificationLinks = @($complete | Where-Object Category -match 'Specification').Count
    OtherWebLinks = @($complete | Where-Object { -not $_.NormalizedCloneUrl -and $_.Category -notmatch 'Specification' }).Count
} | ConvertTo-Json
