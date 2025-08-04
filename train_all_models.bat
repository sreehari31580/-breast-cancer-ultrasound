@echo off

REM Train EfficientNet-B0
python src/cnn_lstm_resnet_train.py --model efficientnet_b0
rename best_model.pth best_model_efficientnet_b0.pth

REM Train EfficientNet-B3
python src/cnn_lstm_resnet_train.py --model efficientnet_b3
rename best_model.pth best_model_efficientnet_b3.pth

REM Train EfficientNet-B4
python src/cnn_lstm_resnet_train.py --model efficientnet_b4
rename best_model.pth best_model_efficientnet_b4.pth

REM Train DenseNet121
python src/cnn_lstm_resnet_train.py --model densenet121
rename best_model.pth best_model_densenet121.pth

echo Training complete. All models saved for ensemble. 