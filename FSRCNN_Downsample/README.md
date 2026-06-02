# FSRCNN Downsample Demo

This demo downsamples a webcam frame using `cv2.INTER_AREA` and then upscales it using FSRCNN x4.
No other filters are applied.

## Setup

1. Copy the model file:

```
cp ../FSRCNN_Filter/FSRCNN_x4.pb ./
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run:

```
python main.py
```

Press `q` to quit.
