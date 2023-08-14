# Import required libraries
import argparse
import os
import platform
import sys
import torch
import json
import numpy as np
import cv2

def model_fn(model_dir):
    os.system("pip install paddlepaddle")
    os.system("pip install 'paddleocr>=2.0.1'")
    import paddle
    from paddleocr import PaddleOCR
    model = PaddleOCR(det_model_dir=os.path.join(model_dir,'model/det'),
            rec_model_dir=os.path.join(model_dir,'model/rec/en'),
            cls_model_dir=os.path.join(model_dir,'model/cls'), 
            use_angle_cls=True, lang="en", use_gpu=False)
    print("Model Loaded")
    return model

def input_fn(input_data, content_type):

    if content_type in ['image/png','image/jpeg']:
        img = np.frombuffer(input_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)[..., ::-1]
        return img 
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return

def predict_fn(input_data, model):
    print("Making inference")
    results = model(input_data,cls=True)
    print("Sending back results")
    #print(results)
    return results