#!/bin/bash

# check if CONDA_DEFAULT_ENV is set and not equal to "base"
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "INFO : current Conda environment set to: $CONDA_DEFAULT_ENV"
else
    echo "!!!WARNING : you are in Conda base environment, you will maybe face some missing packages issues!!!"
fi

# list of python tasks
tasks=(
    "train_model.py: trains breton chat model"
    "train_model_mt.py: trains breton chat model (modified tokenizer)" 
    "train_transcript_model.py: trains french <-> breton transcription model")

# function to display the menu
display_menu() {
    echo "Please choose a task to run:"
    for i in "${!tasks[@]}"; do
        task_name=$(echo "${tasks[$i]}" | cut -d':' -f1)
        task_desc=$(echo "${tasks[$i]}" | cut -d':' -f2-)
        echo "$((i+1))) $task_name - $task_desc"
    done
    echo "$((i+2))) Exit"
}

# function to run the selected script
run_script() {
    if [[ $1 -le ${#tasks[@]} && $1 -gt 0 ]]; then
        task_name=$(echo "${tasks[$1-1]}" | cut -d':' -f1)
        python3 -B "$task_name"
    elif [[ $1 -eq $(( ${#tasks[@]} + 1 )) ]]; then
        echo "Exiting..."
        exit 0
    else
        echo "Invalid choice. Please try again."
    fi
}

display_menu
read -p "Enter your choice: " choice
run_script $choice