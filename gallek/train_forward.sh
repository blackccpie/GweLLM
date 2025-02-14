#!/bin/bash

CHECKPOINT_FORWARD=gallek-m2m100

# bootstrap
echo "BOOTSTRAP"
python3 -B train_translation_model.py nresume nrevert $CHECKPOINT_FORWARD

# run 10 rounds
for i in {1..10}
do
    echo "ROUND $i"

    python3 -B train_translation_model.py resume nrevert $CHECKPOINT_FORWARD
done
