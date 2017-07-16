# Running the Demo

## Prerequisites

- Clone TensorFlow Repository

``` shell
git clone
```

- Export inference graph

``` shell
cd ~/see-celeb/src &&
python export_inference_graph.py --alsologtostderr --model_name=inception_v3 \
--output_file /tmp/inception_v3_inf_graph.pb --dataset_name=<data set name>
```

- Freeze the graph

``` shell
cd ~/tensorflow &&
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/celeb-models/inception_v3/model.ckpt-<step number> \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1
```

## Launch the image classification demo

``` shell
cd ~/see-celeb/src/serving && sh run.sh
```

Then open http://localhost:5001 in your browser
