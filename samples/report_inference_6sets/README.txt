KAIST-OTS RGBT 報告用推論範例（6 組）

資料來源：VAL split
信心門檻：0.25
輸入格式：每組包含一張 *_EO.png 與同名 *_IR.png

挑選原則：
- 1 組：0 human + 1 dummy
- 1 組：1 human + 1 dummy
- 1 組：2 human + 1 dummy
- 1 組：3 human + 1 dummy
- 1 組：4 human + 1 dummy
- 1 組：5 human + 1 dummy

資料夾：
- input：6 組原始 EO／IR，共 12 張
- fp16：TensorRT FP16 標註結果與 detections.json
- int8_mixed：TensorRT INT8／FP16 混合精度標註結果與 detections.json

備註：detections.json 內含類別、confidence、xyxy 邊界框與每張影像速度。
