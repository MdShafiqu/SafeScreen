# SafeScreen
## Scrren Detection
python train.py  --epochs 100 --batch-size 16 --data data.yaml --weights '' --cfg yolov5s.yaml --workers 0

python val.py --weights ./best.pt --data data.yaml

python detect.py --weights ./best.pt --source ./test-images 

## Moire Pattern Detector

python Moire_Pattern_Detection.py
