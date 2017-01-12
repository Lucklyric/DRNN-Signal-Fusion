import data_util
import numpy as np
inputs,outputs=data_util.prepare_data()
segments,labels = data_util.extract_segments(inputs,outputs,10)
print (np.shape(segments),np.shape(labels))


