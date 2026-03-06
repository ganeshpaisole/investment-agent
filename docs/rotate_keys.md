# Key rotation and GitHub Secrets

This document shows recommended steps to rotate keys that were removed
from the repository and to add new secrets to GitHub.

1) Immediate: rotate exposed provider keys

- FMP (Financial Modeling Prep)
  - Log into your FMP account dashboard and create/regenerate an API key.
  - If unsure, create a new key and revoke the old one.

- Google APIs / Google Studio
  - Prefer creating a service account with the minimum roles required and
    use a JSON key for CI. Steps:
    - Console: IAM & Admin → Service Accounts → Create Service Account
    - Grant roles needed (e.g., Kubernetes Engine Admin, Storage Admin,
      Cloud SQL Admin) — prefer the least privilege possible.
    - Create key (JSON) and download it. Keep it secure.
    - Revoke old keys via the service account page or delete the old service account.

- OpenAI (if you had an `sk-` key)
  - Go to https://platform.openai.com/account/api-keys
  - Create a new API key, update your apps, and then revoke the old key.

2) Add rotated keys to GitHub Secrets (recommended)

Use `gh` to set repository secrets (requires GH CLI auth):

```bash
gh secret set FMP_API_KEY -R ganeshpaisole/investment-agent --body "<NEW_FMP_KEY>"
gh secret set GOOGLE_STUDIO_API_KEY -R ganeshpaisole/investment-agent --body "<NEW_GOOGLE_KEY_OR_PLACEHOLDER>"
gh secret set OPENAI_API_KEY -R ganeshpaisole/investment-agent --body "<NEW_OPENAI_KEY>"
```

For the GCP service account JSON (CI deployment):

```bash
# from repo root, where sa-key.json is the JSON key produced by gcloud
gh secret set GCP_SA_KEY -R ganeshpaisole/investment-agent --body "$(cat sa-key.json)"
```

3) Verify CI/Deploy

- Once secrets are set, update the GitHub Actions workflow to consume
  `GCP_SA_KEY` (or use GitHub OIDC for short-lived credentials). Run the
  CI pipeline on a test branch and verify that provisioning/deploy steps
  can authenticate to GCP.

4) Remove local backups securely

- If you backed up old keys to `removed_secrets_20260306/`, delete that
  directory securely once rotation is complete.

5) Audit & monitoring

- Monitor logs for any failed auth attempts using the old keys. Rotate
  again if there is any suspicion of compromise.
