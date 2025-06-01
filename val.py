from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("D:\实验结果\最新更新的v8\ourdataset maxup为0.0/v8原始/runs\segment/train\weights/best.pt") # 自己训练结束后的模型权重
    model.val(data='E:\daima\yolov8-seg1/ultralytics\cfg\datasets\coco128-seg.yaml',
              split='val',
              imgsz=640,
              batch=8,
              save_json=True, # if you need to cal coco metrice
              project='runs/val2',
              name='exp',
              )
