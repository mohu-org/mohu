param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs = @('test', '-p', 'mohu-io', '--target', 'x86_64-pc-windows-gnu')
)

$ErrorActionPreference = 'Stop'

$repoRoot = Join-Path $PSScriptRoot '..'
Set-Location $repoRoot

$packageRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
$llvmPackage = Get-ChildItem $packageRoot -Directory |
    Where-Object { $_.Name -like 'MartinStorsjo.LLVM-MinGW.UCRT_*' } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $llvmPackage) {
    throw 'LLVM-MinGW UCRT is not installed. Install MartinStorsjo.LLVM-MinGW.UCRT with winget first.'
}

$llvmDlltool = Get-ChildItem $llvmPackage.FullName -Recurse -Filter dlltool.exe -File |
    Select-Object -First 1

if (-not $llvmDlltool) {
    throw "Could not find dlltool.exe under $($llvmPackage.FullName)"
}

$llvmBin = Split-Path $llvmDlltool.FullName -Parent
$selfContained = Join-Path $env:USERPROFILE '.rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained'
$rustLld = Join-Path $env:USERPROFILE '.rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\rust-lld.exe'

if (-not (Test-Path $rustLld)) {
    throw "Could not find rust-lld.exe at $rustLld"
}

$env:Path = "$llvmBin;$selfContained;$env:Path"
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = $rustLld

Write-Host "dlltool => $((Get-Command dlltool.exe).Source)"
Write-Host "gcc => $((Get-Command gcc.exe).Source)"

& (Join-Path $env:USERPROFILE '.cargo\bin\rustup.exe') run stable-x86_64-pc-windows-gnu `
    (Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe') @CargoArgs