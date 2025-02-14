#!/bin/bash

CHECKPOINT_BACKWARD=kellag-m2m100

# bootstrap
echo "BOOTSTRAP"
python3 -B train_translation_model.py nresume revert $CHECKPOINT_BACKWARD

# run 10 rounds
for i in {1..10}
do
    echo "ROUND $i"

    python3 -B train_translation_model.py resume revert $CHECKPOINT_BACKWARD
done
