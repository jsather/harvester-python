# detector 

Module for implementing pretrained object detector using You Only Look Once, version 2 ([YOLOv2](https://arxiv.org/abs/1612.08242)). 

## How to run
To run, specify network files in `config.py` and create `detector.Detector` object.

```
import detector.Detector as dt

detector = dt()
bbs = dt.detect(image)
```
