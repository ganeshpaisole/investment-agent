# Deployment and staging notes

This document contains a minimal, opinionated rollout plan to run the `nse_agent`
and `proposal_agent` in staging and production. It is intentionally conservative —
follow your cloud provider best practices when provisioning real infrastructure.

1) Rotate exposed keys (immediately)
   - Any keys you removed from the repository must be rotated at their issuing
     provider (FMP, Google APIs, etc.). Do NOT reuse the old keys.

2) Add secrets to GitHub
   - Use GitHub Secrets for CI/CD and runtime credentials. Example:

```
gh secret set FMP_API_KEY -R <owner>/investment-agent --body "<new-key>"
gh secret set KUBECONFIG -R <owner>/investment-agent --body "$(cat kubeconfig)"
```

3) Local staging
   - Use `docker-compose.staging.yml` to run Redis, Postgres, MinIO and both agents:

```
docker compose -f docker-compose.staging.yml up --build
```

4) CI/CD
   - The workflow at `.github/workflows/ci-cd.yml` runs tests, pip-audit, builds
     and pushes images to GitHub Container Registry, and (optionally) applies
     the `deploy/k8s` manifests to the cluster referenced by `KUBECONFIG`.

5) Staging cluster
   - Apply the manifests in `deploy/k8s` (replace `REPLACE_OWNER` in image
     references with your GitHub organization/user) and ensure `KUBECONFIG` is
     stored as a secret for the workflow.

6) Observability
   - Provide Sentry DSN, Prometheus scrape config, and log aggregation details
     via environment variables or a config map. Do not add secrets to the repo.

7) Purging secrets from history (optional but strongly recommended)
   - Use `git filter-repo` or BFG to remove plaintext tokens from history. This
     operation rewrites history — coordinate with collaborators and ask them to
     `git fetch --all` and re-clone after the rewrite.

8) Rollout
   - Test in staging, run smoke tests and a load test. Use canary or blue/green
     deploy strategies for production cutover.

If you want, I can scaffold Terraform to provision managed Postgres/Redis/S3,
or create Helm charts for the `deploy/k8s` manifests — tell me which cloud
provider you'd like to target (GCP, AWS, Azure, or generic Kubernetes).
