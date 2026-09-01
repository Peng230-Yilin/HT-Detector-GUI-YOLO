[CmdletBinding()]
param(
    [string]$PythonExecutable
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-VerifiedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$PrefixArguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    # Avoid string literals so Windows native-command argument quoting cannot
    # alter the validation program when the executable path contains spaces.
    $probe = "import platform,struct,sys;print(sys.executable);print(platform.python_implementation());print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46));print(struct.calcsize(chr(80))*8)"

    try {
        $probeOutput = @(& $Command @PrefixArguments -B -c $probe 2>$null)
        $probeExitCode = $LASTEXITCODE
    }
    catch {
        throw "$Description could not be executed: $($_.Exception.Message)"
    }

    if ($probeExitCode -ne 0 -or $probeOutput.Count -ne 4) {
        throw "$Description could not execute the Python validation probe."
    }

    $reportedExecutable = [string]$probeOutput[0]
    $implementation = [string]$probeOutput[1]
    $version = [string]$probeOutput[2]
    $bits = [string]$probeOutput[3]
    if (-not [IO.Path]::IsPathRooted($reportedExecutable) -or
        -not (Test-Path -LiteralPath $reportedExecutable -PathType Leaf)) {
        throw "$Description did not return a valid absolute sys.executable path."
    }

    $resolvedExecutable = (Resolve-Path -LiteralPath $reportedExecutable).Path
    if ($implementation -ne "CPython" -or $version -ne "3.10.11" -or $bits -ne "64") {
        throw "$Description is not the required CPython 3.10.11 64-bit interpreter (found $implementation $version, $bits-bit)."
    }

    return $resolvedExecutable
}

$python = $null
if ($PSBoundParameters.ContainsKey("PythonExecutable")) {
    if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
        throw "-PythonExecutable must specify a Python executable path."
    }
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        throw "The specified -PythonExecutable does not exist or is not a file: $PythonExecutable"
    }
    $explicitPath = (Resolve-Path -LiteralPath $PythonExecutable).Path
    $python = Resolve-VerifiedPython -Command $explicitPath -Description "The specified -PythonExecutable"
}
else {
    $failures = [System.Collections.Generic.List[string]]::new()
    $candidates = @(
        @{ Name = "Python Launcher (py -3.10)"; Command = "py"; Arguments = @("-3.10") },
        @{ Name = "PATH python"; Command = "python"; Arguments = @() },
        @{ Name = "PATH python3.10"; Command = "python3.10"; Arguments = @() }
    )
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -CommandType Application -ErrorAction SilentlyContinue)) {
            $failures.Add("$($candidate.Name): command not found")
            continue
        }
        try {
            $python = Resolve-VerifiedPython `
                -Command $candidate.Command `
                -PrefixArguments $candidate.Arguments `
                -Description $candidate.Name
            break
        }
        catch {
            $failures.Add("$($candidate.Name): $($_.Exception.Message)")
        }
    }
    if (-not $python) {
        $details = $failures -join [Environment]::NewLine
        throw @"
No suitable CPython 3.10.11 64-bit interpreter was found.
$details
Specify the interpreter explicitly, for example:
.\install-v1.0.0.ps1 -PythonExecutable "C:\Path\To\Python310\python.exe"
"@
    }
}

Push-Location -LiteralPath $repositoryRoot
try {
    if (Test-Path -LiteralPath ".venv") {
        throw ".venv already exists. Refusing to replace an existing environment."
    }

    & $python -B -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python 3.10.11 virtual environment." }

    $venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
    & $venvPython -B -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare pip." }

    & $venvPython -B -m pip install -r requirements-torch-cpu.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the verified CPU Torch packages." }

    & $venvPython -B -m pip install -r requirements-runtime.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the verified GUI runtime packages." }

    # Install the repository copy last and without dependency resolution so PyPI cannot replace it.
    & $venvPython -B -m pip install --no-deps --editable .\HT-Detector_Peng
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the repository Ultralytics source." }

    & $venvPython -B -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check reported an inconsistent environment." }

    Write-Host "Installation complete. Start with:"
    Write-Host ".venv\Scripts\python.exe -B Peng1.0_GUI\main.py"
}
finally {
    Pop-Location
}
