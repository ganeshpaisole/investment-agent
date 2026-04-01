#!/usr/bin/env bash
set -euo pipefail

# Tear down resources created by `gcp_provision.sh`.
# Usage example:
#   export PROJECT=my-gcp-project
#   export REGION=us-central1
#   export ZONE=us-central1-a
#   export CLUSTER_NAME=agents-cluster
#   export DB_INSTANCE_NAME=agents-db
#   export REDIS_INSTANCE_NAME=agents-redis
#   export BUCKET_NAME=agents-artifacts-12345
#   ./scripts/gcp_teardown.sh

: ${PROJECT:?Set PROJECT}
: ${REGION:=us-central1}
: ${ZONE:=${REGION}-a}
: ${CLUSTER_NAME:=agents-cluster}
: ${DB_INSTANCE_NAME:=agents-db}
: ${REDIS_INSTANCE_NAME:=agents-redis}
: ${BUCKET_NAME:=agents-artifacts-$PROJECT}
SA_EMAIL="ci-deployer@$PROJECT.iam.gserviceaccount.com"

echo "Deleting GKE cluster: $CLUSTER_NAME"
gcloud container clusters delete "$CLUSTER_NAME" --zone="$ZONE" --quiet --project "$PROJECT" || true

echo "Deleting Cloud SQL instance: $DB_INSTANCE_NAME"
gcloud sql instances delete "$DB_INSTANCE_NAME" --quiet --project "$PROJECT" || true

echo "Deleting Redis instance: $REDIS_INSTANCE_NAME"
gcloud redis instances delete "$REDIS_INSTANCE_NAME" --region="$REGION" --quiet --project "$PROJECT" || true

echo "Deleting GCS bucket: $BUCKET_NAME"
gsutil -m rm -r "gs://$BUCKET_NAME" || true

echo "Removing service account keys and service account: $SA_EMAIL"
gcloud iam service-accounts keys list --iam-account="$SA_EMAIL" --project="$PROJECT" --format="value(name)" | \
  xargs -r -n1 -I{} gcloud iam service-accounts keys delete {} --iam-account="$SA_EMAIL" --project="$PROJECT" --quiet || true

gcloud iam service-accounts delete "$SA_EMAIL" --project="$PROJECT" --quiet || true

echo "Teardown complete."
