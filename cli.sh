#!/bin/bash

# Check for minimum number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 command [arguments]"
    exit 1
fi

command=$1
shift  # Remove the first argument (the command)

# Pass the remaining arguments to the Python script
python3 backend.py "$command" "$@"
