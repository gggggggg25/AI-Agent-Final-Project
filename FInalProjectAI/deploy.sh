#!/usr/bin/env bash
# One-command deploy to Cloud Run.  Usage:  ./deploy.sh [PROJECT_ID]
# Optional overrides:  REGION=europe-west1 MODEL=gemini-3.5-flash SERVICE=my-service ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="${1:-${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}}"
REGION="${REGION:-us-central1}"          # where the service runs
SERVICE="${SERVICE:-ml-report-agents}"
MODEL="${MODEL:-gemini-3.8-flash}"       # Gemini 2.5 was retired Oct 2026
GEMINI_LOCATION="${GEMINI_LOCATION:-global}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "No project. Run: gcloud config set project YOUR_PROJECT_ID  (or ./deploy.sh YOUR_PROJECT_ID)"; exit 1
fi
BUILD_STAMP="$(date -u +%Y%m%d-%H%M%S)"
BUCKET="${DATA_BUCKET:-${PROJECT_ID}-ml-datasets}"   # upload your datasets here
echo ">> Deploying '$SERVICE' to project '$PROJECT_ID' ($REGION), model $MODEL @ $GEMINI_LOCATION"

# sanity check: the dependencies must be in requirements.txt or the container will crash on start
for pkg in google-adk pandas scikit-learn matplotlib; do
  grep -qi "^$pkg" requirements.txt || { echo "requirements.txt is missing $pkg"; exit 1; }
done

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com --project "$PROJECT_ID"

# dataset bucket: create if missing, let the service read it
gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --location "$REGION" --project "$PROJECT_ID"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role roles/storage.objectViewer >/dev/null

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT_ID" --region "$REGION" \
  --port 8080 --memory 2Gi --cpu 2 --timeout 900 \
  --min-instances 1 --max-instances 1 --no-cpu-throttling \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=True,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$GEMINI_LOCATION,ADK_MODEL=$MODEL,DATA_BUCKET=$BUCKET,BUILD_STAMP=$BUILD_STAMP"

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')
echo; echo ">> Done: $URL   (build $BUILD_STAMP)"
echo ">> Upload datasets with:  gcloud storage cp yourfile.csv gs://$BUCKET/   (no redeploy needed)"
echo ">> Open it and start a NEW session (old sessions are lost on every deploy)."
