# Windows helper

Use `use-llvm-mingw.ps1` to run `cargo` with the LLVM-MinGW toolchain that
works on this workspace’s Windows setup.

Example:

```powershell
.\scripts\use-llvm-mingw.ps1
```

You can pass extra cargo args after the script name:

```powershell
.\scripts\use-llvm-mingw.ps1 test -p mohu-io --target x86_64-pc-windows-gnu
```