# Running the Demo

## Prerequisites

- Install [GPU version of TensorFlow](
https://www.tensorflow.org/install/install_linux)

- Clone this and TensorFlow repository

``` shell
cd &&
git clone git@github.com:tensorflow/tensorflow.git &&
git clone git@github.com:zyin3/see-celeb.git
```

- Download the datasets

``` shell
mkdir -p ~/datasets/lfw && cd ~/datasets/lfw &&
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz &&
tar -zxvf lfw.tgz
```

- Train the final 2 layers of Inception v3 model

``` shell
cd ~/see-celeb/src &&
sh scripts/finetune_inception_v3_on_lfw.sh
```

- Export inference graph

``` shell
cd ~/see-celeb/src &&
python export_inference_graph.py --alsologtostderr --model_name=inception_v3 \
--output_file /tmp/inception_v3_inf_graph.pb --dataset_name=lfw
```

- Freeze the graph

``` shell
cd ~/tensorflow &&
bazel build tensorflow/python/tools:freeze_graph &&
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/celeb-models/inception_v3/model.ckpt-<step number> \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1
```

## Launch the image classification demo

``` shell
cd ~/see-celeb/src/serving && sh run.sh
```

Then open http://localhost:5001 in your browser.
