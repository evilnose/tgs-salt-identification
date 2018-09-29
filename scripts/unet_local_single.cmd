set MODEL_DIR=out\
set TRAIN_FILE=data\train.npz

del /S /Q %MODEL_DIR%logs\*

gcloud ml-engine local train ^
    --module-name trainer.task ^
    --package-path ..\trainer\ ^
    --job-dir %MODEL_DIR% ^
    -- ^
    --train-file %TRAIN_FILE% ^
    --batch-size 4 ^
    --num-epochs 30 ^
    --lambda-obj 5 ^
    --lambda-noobj 0.4 ^
    --lambda-coord 2 ^
    --valid-split 0.1 ^
    --verbose 1
