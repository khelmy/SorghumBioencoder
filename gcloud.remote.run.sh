#From https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-5-google-cloud
export BUCKET_NAME=sorghumencoder
export JOB_NAME="sorghum_bioencoder"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name src.main \
  --package-path ./src \
  --region $REGION \
  --config=src/cloudml-gpu.yaml \
  -- \
  --train-file ./PickleDump/TrainData.npy
