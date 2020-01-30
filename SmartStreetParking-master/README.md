# Smart Street Parking

This is a web-application designed for the Smart Street Parking project

# Technical Information

 - The application mainly uses Python Flask framework
 - TensorFLow's object detection libirary is used for the detection and analysis process
 - Pylivecap is used for capturing the live information
   Other approaches will be used after intergrating the application with the CCTV camera
 - May use MongoDB to store the data in the future
 
 
# User Guide

Please check `requirements.txt` and install required libiraries. Also, you need to download file `ssd_inception_v2.pb`and move it into this directory: `test_ckpt`. This file can be found at 
`https://github.com/tensorflow/models/tree/master/research/object_detection/test_ckpt` 

Currently, the model used for recognision is the default one powered by TensorFLow. Due to size limitation, you need to download the model by yourself. You can just run `model.py` to solve this issue. We will try other models later in the future.

After setting up the environment, you can run the application by typing
`python app.py`.

If you find that the image is now correctly displayed, please try to disable the cache function for this website in your broswer. However, this should not happen as each image is with a unique time stamp.

Now the APP can do recognision by each single parking spot. When the place is is available, it will be shown in green. After being occupied by a car, the spot(cell) will turn red.


