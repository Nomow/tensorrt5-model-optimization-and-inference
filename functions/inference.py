import numpy as np
import tensorflow as tf

## mean average precision 
## tensorflow implmementation
## @param ytrue - ground truth data y_pred - predicted values
## @return precision coefficent
## @usage coeff = mAP(ground_truth_data, predicted_data)
def mAP(y_true, y_pred):
    
    y_true = np.array(y_true).astype(np.int64);
    y_true = tf.identity(y_true);

    y_pred = np.array(y_pred).astype(np.float32);
    y_pred = tf.identity(y_pred); # np to tensor
    _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 1);
    sess = tf.Session();
    sess.run(tf.local_variables_initializer());
    tf_map = sess.run(m_ap);
    return tf_map;

#The engine is the primary element of TensorRT. It is used to generate a tensorrt.IExecutionContext that can perform inference.
## @param trt_engine_file_path - path to .engine file
## @return ICudaEngine
## @usage engine = loadEngine("/path/to/engine")
def loadEngine(trt_engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR);
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(trt_engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read());
            return engine;
        
#Calculates number of batches required for inference
## @param batch_size - allowed ammount of imgs per batch, nb_of_images - total number of images
## @return number of batches
## @usage nb_of_batches = calculateNbOfBatches(100, 500)
def calculateNbOfBatches(batch_size, nb_of_images):
    nb_of_batches = ((nb_of_images // batch_size) + (1 if (nb_of_images % batch_size) else 0));
    return nb_of_batches;


#Calculates image indices to use for inference on specified batch
## @param batch_size - allowed ammount of imgs per batch,
## @param nb_of_batches - total ammount of batches,
## @param current_batch - batch that is currecntly processed [from 0 to batch_size - 1]
## @param nb_of_imgs - nb of imgs beign processed,
## @return image index from, to  use for inference
## @usage img_indices = calculateImgIndicesToUseForInference(batch_size, nb_of_batches, current_batch, nb_of_imgs)
def calculateImgIndicesToUseForInference(batch_size, nb_of_batches, current_batch, nb_of_imgs):
    from_index = current_batch * batch_size;
    to_index = 0;
    if (current_batch + 1 == nb_of_batches):
        to_index = current_batch * batch_size  + nb_of_imgs - i * batch_size;
    else:
        to_index = (current_batch + 1) * batch_size;        
    return from_index, to_index;