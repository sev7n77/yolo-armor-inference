import cv2
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv26 Armor Plate Inference")
    parser.add_argument('--weights', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='raw_video.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载模型
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    print("推理设备:", model.device)
    
    # 2. 读取视频
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 3. 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. 执行推理
        results = model(frame, conf=args.conf, verbose=False)
        
        # 5. 可视化结果
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints # 获取模型预测的关键点
            
            # 绘制检测框、类别、置信度
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    
                    # 画检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 写类别和置信度
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制4个关键点 (必须来自模型预测，不能用检测框四角替代)
            if keypoints is not None and keypoints.xy.shape[1] > 0:
                for kp_set in keypoints.xy:
                    for kp in kp_set:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0: # 排除无效点
                            # 用红色圆点标记关键点
                            cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)
                            
        # 保存处理后的帧
        out.write(frame)
        
        # 可选：实时显示（按 q 键退出）
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Result saved to {args.output}")

if __name__ == '__main__':
    main()