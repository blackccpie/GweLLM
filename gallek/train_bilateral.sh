#!/bin/bash

# bootstrap
echo "BOOTSTRAP"
python3 -B train_translation_model.py nresume nrevert
python3 -B train_translation_model.py resume revert

# run 10 rounds
for i in {1..10}
do
    echo "ROUND $i"

    python3 -B train_translation_model.py resume nrevert
    python3 -B train_translation_model.py resume revert
done
