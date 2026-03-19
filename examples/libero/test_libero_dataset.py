import tensorflow as tf
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# 指定单个 TFRecord 文件路径
file_path = base_dir + "/modified_libero_rlds/libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00000-of-00032"

# 读取 TFRecordDataset
raw_dataset = tf.data.TFRecordDataset(file_path)

# 取第一条原始样本查看内容
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())  # 解析成 tf.train.Example
    
    # 打印所有 key 和每个 key 对应的 Feature 内容
    for key, feature in example.features.feature.items():
        if feature.bytes_list.value:
            print(f"{key}: bytes_list, length={len(feature.bytes_list.value)}")
        elif feature.int64_list.value:
            print(f"{key}: int64_list, values={feature.int64_list.value}")
        elif feature.float_list.value:
            print(f"{key}: float_list, values={feature.float_list.value}")