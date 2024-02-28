#!/bin/bash

for i in {1..12}
do
  python3 monitoring.py & python3 python_application.py
  sleep 30
done
