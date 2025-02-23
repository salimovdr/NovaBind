#!/bin/bash

FOLDER="SNP_w60s1"

if [ -d "$FOLDER" ]; then
    echo "Удаляю папку $FOLDER..."
    rm -rf "$FOLDER"
    echo "Папка $FOLDER удалена."
else
    echo "Папка $FOLDER не найдена."
fi

