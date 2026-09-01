[CmdletBinding()]
param(
    [string]$Ref = "v1.0.0",
    [string]$OutputParent = "E:\Qt",
    [string]$OutputRoot,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $scriptDirectory
$packageName = "HT-Detector-GUI-YOLO-v1.0.0"
$manifestPath = Join-Path $scriptDirectory "manifest.txt"
$emptyDirectoriesPath = Join-Path $scriptDirectory "empty-directories.txt"

function Get-ManifestEntries {
    Get-Content -LiteralPath $manifestPath -Encoding UTF8 |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") }
}

function Assert-SafeRelativePath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path) -or $Path -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "Unsafe manifest path: $Path"
    }
}

function Assert-NoForbiddenFiles([string]$Root) {
    $forbidden = Get-ChildItem -LiteralPath $Root -Force -Recurse | Where-Object {
        $_.Name -eq ".git" -or
        $_.Name -eq ".venv" -or
        $_.Name -eq "__pycache__" -or
        $_.Name -eq ".pytest_cache" -or
        $_.Name -eq ".vscode" -or
        $_.Name -eq "Run HT-Detector.txt" -or
        $_.Name -like "*.pyc" -or
        $_.Name -like "MySKILL*.txt" -or
        $_.FullName -match '[\\/](runs|results|Paper1-Integrated)([\\/]|$)' -or
        ($_.Extension -eq ".pt" -and $_.FullName -notlike "*\HT-Detector_Peng\weights\cuvette_Peng\yolov8n_train\weights\best.pt")
    }
    if ($forbidden) {
        throw "Forbidden release content:`n$($forbidden.FullName -join "`n")"
    }
}

$manifestEntries = @(Get-ManifestEntries)
if (-not $manifestEntries) { throw "Release manifest is empty." }
$manifestEntries | ForEach-Object { Assert-SafeRelativePath $_ }

Push-Location -LiteralPath $repositoryRoot
try {
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & git rev-parse --verify "$Ref^{commit}" 2>$null | Out-Null
    $refExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($refExitCode -ne 0) { throw "Git ref does not resolve to a commit: $Ref" }

    if ($DryRun) {
        if ($Ref -ne "HEAD") { throw "Dry-run builds must use -Ref HEAD." }
        if (-not $OutputRoot) { throw "Dry-run builds require an explicit -OutputRoot under the system temporary directory." }
        $tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
        $destination = [IO.Path]::GetFullPath($OutputRoot)
        if (-not ($destination + '\').StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Dry-run output must be under the system temporary directory: $tempRoot"
        }
    } else {
        if ($Ref -ne "v1.0.0") { throw "Formal builds must use the v1.0.0 tag." }
        & git show-ref --verify --quiet refs/tags/v1.0.0
        if ($LASTEXITCODE -ne 0) { throw "The required v1.0.0 tag does not exist." }
        $trackedStatus = & git status --porcelain --untracked-files=no
        if ($trackedStatus) { throw "Formal builds require a clean tracked worktree." }
        $destination = Join-Path ([IO.Path]::GetFullPath($OutputParent)) $packageName
        $zipPath = "$destination.zip"
        if (Test-Path -LiteralPath $zipPath) { throw "Refusing to overwrite existing ZIP: $zipPath" }
    }

    if (Test-Path -LiteralPath $destination) { throw "Refusing to overwrite existing directory: $destination" }

    if ($DryRun) {
        New-Item -ItemType Directory -Path $destination | Out-Null
        $files = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($entry in $manifestEntries) {
            $listed = @(& git ls-files --cached --others --exclude-standard -- $entry)
            foreach ($file in $listed) { [void]$files.Add($file) }
        }
        foreach ($file in $files) {
            $source = Join-Path $repositoryRoot $file
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Manifest source file is missing: $file" }
            $target = Join-Path $destination $file
            $targetParent = Split-Path -Parent $target
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
            Copy-Item -LiteralPath $source -Destination $target
        }
    } else {
        $temporaryArchive = Join-Path $env:TEMP ("ht-detector-release-" + [guid]::NewGuid().ToString("N") + ".tar")
        try {
            & git archive --format=tar --output=$temporaryArchive $Ref -- @manifestEntries
            if ($LASTEXITCODE -ne 0) { throw "git archive failed." }
            New-Item -ItemType Directory -Path $destination | Out-Null
            & tar -xf $temporaryArchive -C $destination
            if ($LASTEXITCODE -ne 0) { throw "Failed to extract the release archive." }
        } finally {
            if (Test-Path -LiteralPath $temporaryArchive) { Remove-Item -LiteralPath $temporaryArchive -Force }
        }
    }

    foreach ($emptyDirectory in Get-Content -LiteralPath $emptyDirectoriesPath -Encoding UTF8) {
        $emptyDirectory = $emptyDirectory.Trim()
        if (-not $emptyDirectory -or $emptyDirectory.StartsWith("#")) { continue }
        Assert-SafeRelativePath $emptyDirectory
        New-Item -ItemType Directory -Path (Join-Path $destination $emptyDirectory) -Force | Out-Null
    }

    Assert-NoForbiddenFiles $destination

    $checksumPath = Join-Path $destination "SHA256SUMS.txt"
    $checksumLines = Get-ChildItem -LiteralPath $destination -File -Recurse |
        Where-Object { $_.FullName -ne $checksumPath } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($destination.TrimEnd('\').Length + 1).Replace('\', '/')
            "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash, $relative
        }
    Set-Content -LiteralPath $checksumPath -Value $checksumLines -Encoding UTF8
    Assert-NoForbiddenFiles $destination

    if (-not $DryRun) {
        Compress-Archive -Path (Join-Path $destination '*') -DestinationPath $zipPath
        $verificationRoot = Join-Path $env:TEMP ("ht-detector-zip-verify-" + [guid]::NewGuid().ToString("N"))
        try {
            Expand-Archive -LiteralPath $zipPath -DestinationPath $verificationRoot
            $directoryFiles = @(Get-ChildItem -LiteralPath $destination -File -Recurse)
            $zipFiles = @(Get-ChildItem -LiteralPath $verificationRoot -File -Recurse)
            if ($directoryFiles.Count -ne $zipFiles.Count) {
                throw "ZIP content count does not match the release directory."
            }
            foreach ($directoryFile in $directoryFiles) {
                $relative = $directoryFile.FullName.Substring($destination.TrimEnd('\').Length + 1)
                $zipFile = Join-Path $verificationRoot $relative
                if (-not (Test-Path -LiteralPath $zipFile -PathType Leaf)) {
                    throw "ZIP is missing: $relative"
                }
                if ((Get-FileHash -LiteralPath $directoryFile.FullName -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $zipFile -Algorithm SHA256).Hash) {
                    throw "ZIP content differs from the release directory: $relative"
                }
            }
        } finally {
            if (Test-Path -LiteralPath $verificationRoot) {
                Remove-Item -LiteralPath $verificationRoot -Recurse -Force
            }
        }
        Write-Host "Release directory: $destination"
        Write-Host "Release ZIP: $zipPath"
    } else {
        Write-Host "Dry-run release directory: $destination"
    }
} finally {
    Pop-Location
}
