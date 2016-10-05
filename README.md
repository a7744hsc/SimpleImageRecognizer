# SimpleImageRecognizer
A simple image recognize web based on Flask and Tesnsorflow.

###How to run this web server
1. Install requirements; py3.5,flask,tensorflow.
2. Download the pretrained inception model from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz, put 'classify_image_graph_def.pb','imagenet_2012_challenge_label_map_proto.pbtxt' and 'imagenet_synset_to_human_label_map.txt' into tfwrapper/models.
3. export FLASK_APP=SimpleImageRecognizer.py
4. Run the Flask web server. In develop env, use command 'python -m flask run'.






###TODO:
- [ ] add a requirement file .
- [ ] get rid of temp jpg file.
- [ ] use self-trained model.
- [ ] enhance UI.
- [ ] refine imageRcognizer(By now, most code in this file are copied from tf example).
- [ ] use my own model.
- [ ] use my own model.