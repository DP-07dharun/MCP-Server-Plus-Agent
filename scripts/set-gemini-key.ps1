# Asks for your Gemini API key (input hidden) and writes it to .env at the repo root.
# .env is git-ignored, so the key never gets committed.
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"

$secure = Read-Host "Paste your Gemini API key (it will not be shown)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
if (-not $key) { Write-Host "No key entered, nothing saved."; exit 1 }

$model = "gemini-3.5-flash-lite"
[IO.File]::WriteAllText($envFile, "GEMINI_API_KEY=$key`nGEMINI_MODEL=$model`n")
Write-Host "Saved to $envFile (key length: $($key.Length) characters)."
