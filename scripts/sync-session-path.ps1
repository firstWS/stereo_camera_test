# Rebuild process PATH from Machine + User registry (fixes incomplete PATH in some IDE terminals).
$m = [Environment]::GetEnvironmentVariable("Path", "Machine")
$u = [Environment]::GetEnvironmentVariable("Path", "User")
if ([string]::IsNullOrWhiteSpace($m) -and [string]::IsNullOrWhiteSpace($u)) {
    return
}
if ([string]::IsNullOrWhiteSpace($m)) {
    $env:Path = $u
}
elseif ([string]::IsNullOrWhiteSpace($u)) {
    $env:Path = $m
}
else {
    $env:Path = "$m;$u"
}
