import numpy as np
import tensorflow as tf

## mean average precision 
## tensorflow implmementation
## @param ytrue - ground truth data y_pred - predicted values
## @return precision coefficent
## @usage coeff = mAP(ground_truth_data, predicted_data)
def mAP(y_true, y_pred):
    
    y_true = np.array(y_true).astype(np.int64)
    y_true = tf.identity(y_true)

    y_pred = np.array(y_pred).astype(np.float32)
    y_pred = tf.identity(y_pred) # np to tensor
    _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 1)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    tf_map = sess.run(m_ap)
    return tf_map