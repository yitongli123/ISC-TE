# ISC-TE
SCRIBBLE-SUPERVISED TARGET EXTRACTION METHOD BASED ON INNER STRUCTURE-CONSTRAINT FOR REMOTE SENSING IMAGES

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
In the paper, the "n_epochs", "num" and "degree" in "TRAIN" are set to 1000, 0.3 and 0.7,  
which has been decided after enough comparative experiments. Prediction results in testing 
with different hyper-parameter settings are partly shown below, proving the effectiveness 
of our setting in the paper. 
