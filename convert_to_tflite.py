import tensorflow as tf

MODEL_PATH = "model/keypoint_classifier/keypoint_classifier.keras"
TFLITE_PATH = "model/keypoint_classifier/keypoint_classifier.tflite"

# 重新加载 keras 模型（不要转换为 SavedModel）
model = tf.keras.models.load_model(MODEL_PATH)

# 清理 TF 会话（可选）
tf.keras.backend.clear_session()

# 不启用 GPU（避免 Windows 下某些卷积 kernel 冲突）
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# 直接从 Keras 模型转换（绕过 MLIR bug）
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 建议使用 float16 —— 最稳定、文件小、不会触发动态量化 bug
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("转换成功，保存到：", TFLITE_PATH)
