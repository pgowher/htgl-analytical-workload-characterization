#requires -Version 5.1

<#
.SYNOPSIS
Builds a locked Python environment and publishes one complete HTGL analytical
workload experiment and its figures.

.DESCRIPTION
This runner is fail-closed and Windows PowerShell 5.1 compatible. It validates
an explicitly supported 64-bit CPython runtime without injecting multiline
source through `python -c`, creates an isolated environment, resolves the exact
dependency closure from requirements.txt, executes both self-test suites,
publishes the simulation, and plots only the committed simulation run produced
by this invocation.
#>

[CmdletBinding()]
param(
    [string] $PythonCommand = "",
    [string] $VirtualEnvironment = ".venv-htgl",
    [string] $OutputDirectory = "htgl_analytical_data",
    [string] $ArtifactDirectory = "analytical_artifacts",
    [string] $FigureDirectory = "htgl_analytical_figures",
    [string] $Config = "",

    [ValidateRange(1, 1000000)]
    [int] $Replications = 30,

    [ValidateRange(2, 1000000)]
    [int] $DaysPerReplication = 365,

    [ValidateRange(0, 9223372036854775807)]
    [long] $Seed = 20260722,

    [ValidateRange(300, 2400)]
    [int] $Dpi = 450,

    [ValidateSet("pdf", "png")]
    [string[]] $Formats = @("pdf", "png"),

    [switch] $NoDailySamples,
    [switch] $PrimaryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ContractVersion = "htgl-analytical-release-contract-v1"
$RunnerVersion = "run-experimentV2-supported-cpython-3.12-3.13-2026-07-23"
$SupportedPythonMinors = @("3.13", "3.12")
$PinnedToolchain = [ordered]@{
    pip = "26.0.1"
    setuptools = "82.0.1"
    wheel = "0.47.0"
}

function Resolve-AnchoredPath {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $BaseDirectory
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath(
        (Join-Path -Path $BaseDirectory -ChildPath $Path)
    )
}

function Resolve-Executable {
    param([Parameter(Mandatory = $true)][string] $Name)
    $command = Get-Command -Name $Name -CommandType Application -ErrorAction Stop
    if (-not [string]::IsNullOrWhiteSpace([string] $command.Path)) {
        return [string] $command.Path
    }
    if (-not [string]::IsNullOrWhiteSpace([string] $command.Source)) {
        return [string] $command.Source
    }
    throw "Unable to resolve executable '$Name'."
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string] $Executable,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]] $Arguments,
        [Parameter(Mandatory = $true)][string] $Description
    )
    Write-Host "`n==> $Description"
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

function Invoke-CapturedNative {
    param(
        [Parameter(Mandatory = $true)][string] $Executable,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]] $Arguments,
        [Parameter(Mandatory = $true)][string] $Description
    )
    $lines = @(& $Executable @Arguments 2>&1 | ForEach-Object { [string] $_ })
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $detail = ($lines -join [Environment]::NewLine).Trim()
        throw "$Description failed with exit code $exitCode. $detail"
    }
    return ($lines -join [Environment]::NewLine).Trim()
}

function Resolve-PyLauncherInterpreter {
    param([Parameter(Mandatory = $true)][string] $Executable)
    $failures = New-Object System.Collections.Generic.List[string]
    foreach ($minor in $SupportedPythonMinors) {
        $prefix = @("-$minor")
        try {
            $version = Invoke-CapturedNative -Executable $Executable `
                -Arguments ($prefix + @("--version")) `
                -Description "Probe CPython $minor through the Python launcher"
            if ($version -match "^Python $([regex]::Escape($minor))\.\d+([\s\r\n]|$)") {
                return [PSCustomObject]@{
                    Executable = $Executable
                    Prefix = $prefix
                }
            }
            $failures.Add("${minor}: unexpected response '$version'")
        }
        catch {
            $failures.Add("${minor}: $($_.Exception.Message)")
        }
    }
    throw (
        "The Python launcher does not expose a supported 64-bit CPython " +
        "runtime ($($SupportedPythonMinors -join ' or ')). " +
        "Probe results: $($failures -join '; ')"
    )
}

function Resolve-BootstrapPython {
    if (-not [string]::IsNullOrWhiteSpace($PythonCommand)) {
        $executable = Resolve-Executable -Name $PythonCommand
        $leaf = [System.IO.Path]::GetFileName($executable).ToLowerInvariant()
        if ($leaf -eq "py" -or $leaf -eq "py.exe") {
            $selection = Resolve-PyLauncherInterpreter -Executable $executable
            return $selection
        }
        return [PSCustomObject]@{ Executable = $executable; Prefix = @() }
    }

    $launcher = Get-Command -Name "py.exe" -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        try {
            $selection = Resolve-PyLauncherInterpreter -Executable ([string] $launcher.Path)
            return $selection
        }
        catch {
            Write-Host (
                "Python launcher has no supported runtime; probing the " +
                "default 'python' executable."
            )
        }
    }
    return [PSCustomObject]@{
        Executable = Resolve-Executable -Name "python"
        Prefix = @()
    }
}

function Assert-PythonContract {
    param(
        [Parameter(Mandatory = $true)][string] $Executable,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]] $Prefix,
        [Parameter(Mandatory = $true)][string] $Label
    )
    $version = Invoke-CapturedNative -Executable $Executable `
        -Arguments ($Prefix + @("--version")) `
        -Description "$Label version query"
    if ($version -notmatch '^Python (?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)([\s\r\n]|$)') {
        throw "$Label returned an unparseable version string: '$version'."
    }
    $major = [int] $Matches.major
    $minor = [int] $Matches.minor
    $patch = [int] $Matches.patch
    $majorMinor = "$major.$minor"
    if ($SupportedPythonMinors -notcontains $majorMinor) {
        throw (
            "$Label must be 64-bit CPython $($SupportedPythonMinors -join ' or '); " +
            "observed '$version'."
        )
    }

    $implementation = Invoke-CapturedNative -Executable $Executable `
        -Arguments ($Prefix + @("-c", "import platform; print(platform.python_implementation())")) `
        -Description "$Label implementation query"
    if ($implementation.Trim() -ne "CPython") {
        throw "$Label must use CPython; observed '$implementation'."
    }

    $bitness = Invoke-CapturedNative -Executable $Executable `
        -Arguments ($Prefix + @("-c", "import sys; print(64 if sys.maxsize > 2**32 else 32)")) `
        -Description "$Label architecture query"
    if ($bitness.Trim() -ne "64") {
        throw "$Label must be 64-bit; observed '$bitness-bit'."
    }
    Write-Host "Validated ${Label}: $version, $implementation, $bitness-bit"
    return [PSCustomObject]@{
        Version = "$major.$minor.$patch"
        MajorMinor = $majorMinor
        Implementation = $implementation.Trim()
        Bitness = [int] $bitness.Trim()
    }
}

function ConvertTo-CanonicalPackageName {
    param([Parameter(Mandatory = $true)][string] $Name)
    return (($Name.Trim().ToLowerInvariant()) -replace '[-_.]+', '-')
}

function Read-ExactRequirementLock {
    param([Parameter(Mandatory = $true)][string] $Path)
    $expected = @{}
    $lineNumber = 0
    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $lineNumber += 1
        $line = $rawLine.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            continue
        }
        if ($line -notmatch '^(?<name>[A-Za-z0-9_.-]+)==(?<version>[^\s;]+)$') {
            throw "Unpinned or unsupported requirement at ${Path}:${lineNumber}: '$line'."
        }
        $name = ConvertTo-CanonicalPackageName -Name $Matches.name
        if ($expected.ContainsKey($name)) {
            throw "Duplicate requirement '$name' in $Path."
        }
        $expected[$name] = [string] $Matches.version
    }
    if ($expected.Count -eq 0) {
        throw "Dependency lock is empty: $Path"
    }
    foreach ($entry in $PinnedToolchain.GetEnumerator()) {
        $expected[(ConvertTo-CanonicalPackageName -Name $entry.Key)] = $entry.Value
    }
    return $expected
}

function Assert-ExactInstalledClosure {
    param(
        [Parameter(Mandatory = $true)][string] $Python,
        [Parameter(Mandatory = $true)][hashtable] $Expected
    )
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $inventoryJson = Invoke-CapturedNative -Executable $Python `
        -Arguments @(
            "-m", "pip", "--isolated", "--disable-pip-version-check",
            "list", "--format=json"
        ) `
        -Description "Inventory resolved dependency closure"
    try {
        $packages = $inventoryJson | ConvertFrom-Json
    }
    catch {
        throw "pip returned invalid package-inventory JSON: $($_.Exception.Message)"
    }

    $installed = @{}
    foreach ($package in @($packages)) {
        $name = ConvertTo-CanonicalPackageName -Name ([string] $package.name)
        if ($installed.ContainsKey($name)) {
            throw "Installed package inventory contains duplicate '$name'."
        }
        $installed[$name] = [string] $package.version
    }

    $problems = New-Object System.Collections.Generic.List[string]
    foreach ($name in ($Expected.Keys | Sort-Object)) {
        if (-not $installed.ContainsKey($name)) {
            $problems.Add("$name is missing; expected $($Expected[$name])")
        }
        elseif ($installed[$name] -ne $Expected[$name]) {
            $problems.Add("$name is $($installed[$name]); expected $($Expected[$name])")
        }
    }
    foreach ($name in ($installed.Keys | Sort-Object)) {
        if (-not $Expected.ContainsKey($name)) {
            $problems.Add("unexpected distribution $name==$($installed[$name])")
        }
    }
    if ($problems.Count -gt 0) {
        throw "Resolved environment does not match the exact lock: $($problems -join '; ')."
    }
    Write-Host "Exact dependency closure verified: $($Expected.Count) distributions"
}

function Read-JsonObject {
    param([Parameter(Mandatory = $true)][string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required JSON file was not published: $Path"
    }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "Invalid JSON in '$Path': $($_.Exception.Message)"
    }
}

function Assert-CommittedRelease {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $ExpectedContract,
        [string] $ExpectedInputRunId = ""
    )
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $pointerPath = Join-Path $rootFull "latest.json"
    $pointer = Read-JsonObject -Path $pointerPath
    $pointerFields = @($pointer.PSObject.Properties.Name)
    foreach ($field in @("contract_version", "run_id", "relative_run_directory", "release_commit_sha256")) {
        if (-not ($pointerFields -contains $field)) {
            throw "Release pointer '$pointerPath' omits '$field'."
        }
    }
    if ([string] $pointer.contract_version -ne $ExpectedContract) {
        throw "Unsupported release contract in '$pointerPath'."
    }

    $runDirectory = [System.IO.Path]::GetFullPath(
        (Join-Path $rootFull ([string] $pointer.relative_run_directory))
    )
    $rootPrefix = $rootFull.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $runDirectory.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Release pointer escapes its publication root."
    }
    if ((Split-Path $runDirectory -Leaf) -ne [string] $pointer.run_id) {
        throw "Release pointer run ID and immutable directory disagree."
    }
    if (-not (Test-Path -LiteralPath $runDirectory -PathType Container)) {
        throw "Committed run directory is absent: $runDirectory"
    }

    $commitPath = Join-Path $runDirectory "RELEASE_COMMIT.json"
    $commit = Read-JsonObject -Path $commitPath
    $commitFields = @($commit.PSObject.Properties.Name)
    if ($commit.complete -ne $true -or [string] $commit.contract_version -ne $ExpectedContract) {
        throw "Release commit is incomplete or unsupported: $commitPath"
    }
    if ([string] $commit.run_id -ne [string] $pointer.run_id) {
        throw "Release pointer and commit disagree on the run ID."
    }
    $commitHash = (Get-FileHash -LiteralPath $commitPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($commitHash -ne ([string] $pointer.release_commit_sha256).ToLowerInvariant()) {
        throw "Release pointer does not bind RELEASE_COMMIT.json."
    }

    $isData = $commitFields -contains "release_manifest_sha256"
    $isFigure = $commitFields -contains "figure_release_manifest_sha256"
    if ($isData -eq $isFigure) {
        throw "Release commit must bind exactly one recognized manifest."
    }
    if ($isData) {
        $manifestName = "release_manifest.json"
        $expectedManifestHash = [string] $commit.release_manifest_sha256
        $requiredEvidence = @(
            "statistical_validation_metrics.csv",
            "verification_checks.csv",
            "claim_matrix.csv",
            "claim_matrix_validation.csv"
        )
    }
    else {
        $manifestName = "figure_release_manifest.json"
        $expectedManifestHash = [string] $commit.figure_release_manifest_sha256
        $requiredEvidence = @("figure_claim_matrix.csv")
    }
    $manifestPath = Join-Path $runDirectory $manifestName
    $null = Read-JsonObject -Path $manifestPath
    $actualManifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualManifestHash -ne $expectedManifestHash.ToLowerInvariant()) {
        throw "Release commit does not bind '$manifestName'."
    }
    foreach ($name in $requiredEvidence) {
        $path = Join-Path $runDirectory $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Committed release omits required evidence: $path"
        }
    }

    if (-not ($commitFields -contains "publication_gates")) {
        throw "Release commit omits publication gates."
    }
    $failedGates = @(
        $commit.publication_gates.PSObject.Properties |
            Where-Object { $_.Value -ne $true } |
            ForEach-Object { $_.Name }
    )
    if ($failedGates.Count -gt 0) {
        throw "Release has failed publication gates: $($failedGates -join ', ')"
    }

    if (-not [string]::IsNullOrWhiteSpace($ExpectedInputRunId)) {
        if ([string] $pointer.input_data_run_id -ne $ExpectedInputRunId -or
            [string] $commit.input_data_run_id -ne $ExpectedInputRunId) {
            throw "Figure release is not bound to this invocation's simulation run."
        }
    }
    return [PSCustomObject]@{
        RunId = [string] $pointer.run_id
        Directory = $runDirectory
    }
}

try {
    $scriptRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
    $requirementsPath = Join-Path $scriptRoot "requirements.txt"
    $simulationScript = Join-Path $scriptRoot "simUp.py"
    $plotScript = Join-Path $scriptRoot "plots.py"
    foreach ($path in @($requirementsPath, $simulationScript, $plotScript)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required release file is absent: $path"
        }
    }

    $venvPath = Resolve-AnchoredPath -Path $VirtualEnvironment -BaseDirectory $scriptRoot
    $dataRoot = Resolve-AnchoredPath -Path $OutputDirectory -BaseDirectory $scriptRoot
    $artifactRoot = Resolve-AnchoredPath -Path $ArtifactDirectory -BaseDirectory $scriptRoot
    $figureRoot = Resolve-AnchoredPath -Path $FigureDirectory -BaseDirectory $scriptRoot
    $bootstrap = Resolve-BootstrapPython
    $bootstrapInfo = Assert-PythonContract -Executable $bootstrap.Executable -Prefix @($bootstrap.Prefix) `
        -Label "bootstrap interpreter"
    Write-Host "Runner contract     : $RunnerVersion"

    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $venvPath) {
            throw "Virtual-environment path exists but is incomplete: $venvPath"
        }
        Invoke-CheckedNative -Executable $bootstrap.Executable `
            -Arguments (@($bootstrap.Prefix) + @("-m", "venv", $venvPath)) `
            -Description "Create isolated CPython $($bootstrapInfo.MajorMinor) environment"
    }
    $venvInfo = Assert-PythonContract -Executable $venvPython -Prefix @() `
        -Label "virtual-environment interpreter"
    if ($venvInfo.MajorMinor -ne $bootstrapInfo.MajorMinor) {
        throw (
            "Existing virtual environment uses CPython $($venvInfo.MajorMinor), " +
            "but the selected bootstrap interpreter is CPython " +
            "$($bootstrapInfo.MajorMinor). The runner will not silently replace " +
            "an environment. Remove only '$venvPath' or choose a new " +
            "-VirtualEnvironment path, then rerun."
        )
    }

    Invoke-CheckedNative -Executable $venvPython -Arguments @(
        "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
        "--no-input", "--upgrade",
        "pip==$($PinnedToolchain.pip)",
        "setuptools==$($PinnedToolchain.setuptools)",
        "wheel==$($PinnedToolchain.wheel)"
    ) -Description "Install pinned packaging toolchain"

    Invoke-CheckedNative -Executable $venvPython -Arguments @(
        "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
        "--no-input", "--upgrade", "--only-binary=:all:",
        "--requirement", $requirementsPath
    ) -Description "Resolve exact runtime dependency lock"

    Invoke-CheckedNative -Executable $venvPython `
        -Arguments @("-m", "pip", "--isolated", "check") `
        -Description "Verify dependency consistency"
    $expectedPackages = Read-ExactRequirementLock -Path $requirementsPath
    Assert-ExactInstalledClosure -Python $venvPython -Expected $expectedPackages

    Invoke-CheckedNative -Executable $venvPython `
        -Arguments @($simulationScript, "--self-test") `
        -Description "Run simulation release-contract self-tests"
    Invoke-CheckedNative -Executable $venvPython `
        -Arguments @($plotScript, "--self-test") `
        -Description "Run plotting release-contract self-tests"

    $simulationArguments = @(
        $simulationScript,
        "--output-dir", $dataRoot,
        "--artifact-dir", $artifactRoot
    )
    if (-not [string]::IsNullOrWhiteSpace($Config)) {
        $configPath = Resolve-AnchoredPath -Path $Config -BaseDirectory $scriptRoot
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            throw "Experiment configuration does not exist: $configPath"
        }
        $simulationArguments += @("--config", $configPath)
    }
    if ([string]::IsNullOrWhiteSpace($Config) -or $PSBoundParameters.ContainsKey("Replications")) {
        $simulationArguments += @("--replications", [string] $Replications)
    }
    if ([string]::IsNullOrWhiteSpace($Config) -or $PSBoundParameters.ContainsKey("DaysPerReplication")) {
        $simulationArguments += @("--days-per-replication", [string] $DaysPerReplication)
    }
    if ([string]::IsNullOrWhiteSpace($Config) -or $PSBoundParameters.ContainsKey("Seed")) {
        $simulationArguments += @("--seed", [string] $Seed)
    }
    if ($NoDailySamples) {
        $simulationArguments += "--no-daily-samples"
    }

    Invoke-CheckedNative -Executable $venvPython `
        -Arguments $simulationArguments `
        -Description "Publish HTGL analytical workload experiment"
    $dataRelease = Assert-CommittedRelease -Root $dataRoot `
        -ExpectedContract $ContractVersion

    $plotArguments = @(
        $plotScript,
        "--data-dir", $dataRoot,
        "--figure-dir", $figureRoot,
        "--formats"
    ) + $Formats + @("--dpi", [string] $Dpi, "--clean")
    if ($PrimaryOnly) {
        $plotArguments += "--primary-only"
    }

    Invoke-CheckedNative -Executable $venvPython `
        -Arguments $plotArguments `
        -Description "Publish figures from the committed experiment"
    $figureRelease = Assert-CommittedRelease -Root $figureRoot `
        -ExpectedContract $ContractVersion `
        -ExpectedInputRunId $dataRelease.RunId

    Write-Host "`nHTGL analytical workload release completed successfully." -ForegroundColor Green
    Write-Host "Simulation run ID : $($dataRelease.RunId)"
    Write-Host "Simulation release: $($dataRelease.Directory)"
    Write-Host "Figure run ID     : $($figureRelease.RunId)"
    Write-Host "Figure release    : $($figureRelease.Directory)"
    exit 0
}
catch {
    [Console]::Error.WriteLine(
        "HTGL analytical workload release failed: $($_.Exception.Message)"
    )
    exit 1
}
