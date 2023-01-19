# ISC-TE
SCRIBBLE-SUPERVISED TARGET EXTRACTION METHOD BASED ON INNER STRUCTURE-CONSTRAINT FOR REMOTE SENSING IMAGES

# Prepare Dataset
'''
sh preprocess_dataset.sh
'''

Note: After running the above code, you will get a directory named "examples" where input images are stored 
in 'examples/images/{image_type}/' and 'preprocess_dataset.sh' for trainning and testing respectively.
Besides, scribble labels in which target region, background and unknown area are correspondingly annotated by 
pixels with values of 1, 0 and 250 are generated and stored in 'examples/labels/{image_type}/'. To apply other 
datasets, you can modify "preprocess_dataset.sh" and "preprocess.py". In testing, fully-annotated labels in
'examples/GT/{image_type}/' are used for validation, which need to be manually created.

# Train
'''
python Train.py --config_path config/model.json
'''

# Test
'''
python Inference.py --config_path config/model.json
'''

# Train + Test
'''
sh run.sh
'''

# Hyper-parameter setting
In the paper, the hyper-parameters "num" and "degree" in "TRAIN" are set to 0.3 and 0.7, which has been decided
after enough comparative experiments. Prediction results in testing with different hyper-parameter settings are 
partly shown below, proving the effectiveness of our setting in the paper. 

Input image
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/input.png)

"num":0.1, "degree":0.7
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0107.png)

"num":0.3, "degree":0.7
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0307.png)

"num":0.5, "degree":0.7
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0507.png)

"num":0.7, "degree":0.7
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0707.png)

"num":0.9, "degree":0.7
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0907.png)

"num":0.3, "degree":0.1
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0301.png)

"num":0.3, "degree":0.3
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0303.png)

"num":0.3, "degree":0.5
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0305.png)

"num":0.3, "degree":0.9
![image](https://github.com/yitongli123/ISC-TE/blob/main/images/0309.png)




