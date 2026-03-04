Notifications configuration and secure secrets

This project supports sending notifications (webhook POSTs and SMTP email) when calibration requests are created or approved.

Configuration files
- `proposal_agent/data/notifications.json` contains the default configuration. By default it is disabled (`"enabled": false`).

Secure environment variable overrides (recommended for secrets)
- `PROPOSAL_NOTIFY_ENABLED` : `true` or `false` to enable notifications.
- `PROPOSAL_NOTIFY_WEBHOOKS` : comma-separated list of webhook URLs.
- `PROPOSAL_NOTIFY_SMTP_SERVER` : SMTP server host.
- `PROPOSAL_NOTIFY_SMTP_PORT` : SMTP server port (e.g., `587`).
- `PROPOSAL_NOTIFY_SMTP_FROM` : From address used when sending email.
- `PROPOSAL_NOTIFY_SMTP_TO` : Comma-separated list of recipient addresses.
- `PROPOSAL_NOTIFY_SMTP_USERNAME` : SMTP username (use env var rather than storing in file).
- `PROPOSAL_NOTIFY_SMTP_PASSWORD` : SMTP password (use env var rather than storing in file).
- `PROPOSAL_NOTIFY_USE_TLS` : `true` to use STARTTLS.

Examples (PowerShell):

$env:PROPOSAL_NOTIFY_ENABLED = 'true'
$env:PROPOSAL_NOTIFY_SMTP_SERVER = 'smtp.example.com'
$env:PROPOSAL_NOTIFY_SMTP_PORT = '587'
$env:PROPOSAL_NOTIFY_SMTP_FROM = 'no-reply@example.com'
$env:PROPOSAL_NOTIFY_SMTP_TO = 'ops@example.com'
$env:PROPOSAL_NOTIFY_SMTP_USERNAME = 'smtp-user'
$env:PROPOSAL_NOTIFY_SMTP_PASSWORD = 'secret'
$env:PROPOSAL_NOTIFY_USE_TLS = 'true'

Then run the safe test script (it will actually send only when enabled is true):

python proposal_agent/tools/test_notifications.py

Security guidance
- Prefer storing credentials in an OS secret store or CI secrets rather than plaintext files.
- Do not commit secrets to source control.

If you want, I can wire the dashboard to read secrets from a specific secret manager (Azure Key Vault, AWS Secrets Manager, or HashiCorp Vault)."}