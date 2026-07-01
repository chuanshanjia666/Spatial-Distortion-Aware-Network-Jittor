#!/bin/bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  point2rbox-jittor \
  /bin/bash -c "python3.10 -m jittor.test.test_example 2>&1" | tee "env_check.log"
