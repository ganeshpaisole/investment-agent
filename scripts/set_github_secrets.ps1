param(
    [string]$Repo = "ganeshpaisole/investment-agent"
)

function Set-SecretInteractive([string]$name) {
    Write-Host "Setting secret: $name"
    $sec = Read-Host -AsSecureString "Enter value for $name (input hidden)"
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    gh secret set $name -R $Repo --body $plain
    Write-Host "$name set for repo $Repo"
}

Write-Host "Ensure you are logged in: gh auth status"
Set-SecretInteractive -name 'FMP_API_KEY'
Set-SecretInteractive -name 'GOOGLE_STUDIO_API_KEY'
Set-SecretInteractive -name 'OPENAI_API_KEY'
Write-Host "If you need to set KUBECONFIG, run:`n gh secret set KUBECONFIG -R $Repo --body \"$(Get-Content -Raw <path-to-kubeconfig>)\"`"
