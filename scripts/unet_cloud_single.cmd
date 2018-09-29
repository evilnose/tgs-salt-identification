call datetime.cmd
set REGION=us-central1
set BUCKET_NAME=central-ml
set JOB_NAME=unet_single_%DATETIME%
set OUTPUT_PATH=gs://%BUCKET_NAME%/%JOB_NAME%
set CLOUD_CONFIG=..\trainer\config.yml
set TRAIN_FILE=gs://%BUCKET_NAME%/data/train.npz


gcloud ml-engine jobs submit training %JOB_NAME% ^
    --job-dir %OUTPUT_PATH% ^
    --runtime-version 1.8 ^
    --module-name trainer.task ^
    --package-path ..\trainer\ ^
    --region %REGION% ^
    --config %CLOUD_CONFIG% ^
    --runtime-version 1.10 ^
    --python-version  3.5 ^
    -- ^
    --train-file %TRAIN_FILE% ^
    --batch-size 32 ^
    --num-epochs 30 ^
    --lambda-obj 6 ^
    --lambda-noobj 0.3 ^
    --lambda-coord 1 ^
    --valid-split 0.1 ^
    --verbose 2
