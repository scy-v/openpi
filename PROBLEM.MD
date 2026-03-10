1. <project_folder>/examples/libero/convert_libero_data_to_lerobot.py运行时，出现两个依赖版本不正确
   - ml-dtypes版本过低，没有int2 type，需在pyproject.toml中修改ml-dtypes==0.5.0，重新uv sync.
   - protobuf版本过低，没有runtime_version.py，这个低版本包被作者固定在uv.lock中，假设不修改的话，可以uv pip install --upgrade protobuf，此时会升级protobuf，但是由于uv自动解析uv.lock的版本依赖，当运行程序后，会自动变为原版本，所以，升级后，先访问protobuf源码路径，将runtime_version.py代码copy出来，降级protobuf版本后，再copy到源码路径。
2. lerobot
   - 更换pyproject.toml中lerobot commit的hash为d6ea3bbce0fc8c75138f02639ac4ceaf12a67829，这是rerun upgrade前的最后一个版本，v0.3.4，也是应用了新数据集格式的版本。
   - v0.3.4版本需去掉import lerobot.common.datasets.lerobot_dataset as lerobot_dataset 中的common
3. 注意config中所有的修改