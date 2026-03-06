#!/usr/bin/env bash
set -euo pipefail

# GCP provisioning helper (gcloud-based)
# Usage (example):
#   export PROJECT=my-gcp-project
#   export REGION=us-central1
#   export ZONE=us-central1-a
#   export CLUSTER_NAME=agents-cluster
#   export DB_INSTANCE_NAME=agents-db
#   export DB_PASSWORD=$(openssl rand -base64 16)
#   export REDIS_INSTANCE_NAME=agents-redis
#   export BUCKET_NAME=agents-artifacts-$(date +%s)
#   ./scripts/gcp_provision.sh

if [ -z "${PROJECT:-}" ]; then
  echo "ERROR: set PROJECT environment variable (GCP project id)"
  exit 1
fi

: ${REGION:=us-central1}
: ${ZONE:=${REGION}-a}
: ${CLUSTER_NAME:=agents-cluster}
: ${DB_INSTANCE_NAME:=agents-db}
: ${DB_PASSWORD:=changeme}
: ${REDIS_INSTANCE_NAME:=agents-redis}
: ${BUCKET_NAME:=agents-artifacts-$PROJECT}

echo "Enabling required APIs..."
gcloud services enable container.googleapis.com sqladmin.googleapis.com redis.googleapis.com storage.googleapis.com iam.googleapis.com --project "$PROJECT"

echo "Creating GKE cluster: $CLUSTER_NAME (zone: $ZONE)"
gcloud container clusters create "$CLUSTER_NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --num-nodes 1 \
  --machine-type e2-medium \
  --enable-ip-alias

echo "Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT"

echo "Creating Cloud SQL (Postgres) instance: $DB_INSTANCE_NAME"
gcloud sql instances create "$DB_INSTANCE_NAME" \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region="$REGION" \
  --project="$PROJECT"

echo "Setting postgres password..."
gcloud sql users set-password postgres --instance="$DB_INSTANCE_NAME" --password="$DB_PASSWORD" --project="$PROJECT"

echo "Creating Memorystore (Redis) instance: $REDIS_INSTANCE_NAME"
gcloud redis instances create "$REDIS_INSTANCE_NAME" \
  --size=1 --region="$REGION" --project="$PROJECT"

echo "Creating GCS bucket: $BUCKET_NAME"
gsutil mb -l "$REGION" "gs://$BUCKET_NAME"

echo "Creating CI service account 'ci-deployer'"
gcloud iam service-accounts create ci-deployer --display-name "CI Deployer" --project="$PROJECT"
SA_EMAIL="ci-deployer@$PROJECT.iam.gserviceaccount.com"

echo "Granting roles/editor to $SA_EMAIL (recommend tightening permissions later)"
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "roles/editor"

echo "Creating service account key: sa-key.json (store securely)"
gcloud iam service-accounts keys create sa-key.json --iam-account="$SA_EMAIL" --project="$PROJECT"

echo "Provisioning complete. Next steps:" 
echo " - Securely store sa-key.json and add to GitHub Secrets as GCP_SA_KEY (or use GitHub Actions OIDC)."
echo " - Add runtime secrets: FMP_API_KEY, GOOGLE_STUDIO_API_KEY, OPENAI_API_KEY"
echo " - Update CI to use GCP_SA_KEY for terraform/gcloud operations or configure OIDC for least-privilege auth."
