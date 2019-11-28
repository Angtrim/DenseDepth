import tensorflow as tf
import tensorflow.keras.backend as K

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def edges_depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    # Gradient magnitude of the true depth map
    grad_magn = K.sqrt(K.pow(dx_true, 2) + K.pow(dy_true, 2))

    # Mask to divide high freq to low frew component
    mask = (grad_magn - K.min(grad_magn)) / (K.max(grad_magn) - K.min(grad_magn))

    # High freq and low freq depthmaps
    hf_y_true = (1 - mask) * y_true
    lf_y_true = (mask) * y_true

    hf_y_pred = (1 - mask) * y_pred
    lf_y_pred = (mask) * y_pred

    # MAE of low freq
    low_freq_loss = K.mean(K.abs(lf_y_pred - lf_y_true), axis=-1)

    # GRAD of hf depth
    dy_true_hf, dx_true_hf = tf.image.image_gradients(hf_y_true)
    dy_pred_hf, dx_pred_hf = tf.image.image_gradients(hf_y_pred)

    # MAE of hf freq
    high_freq_loss = K.mean(K.abs(dy_pred_hf - dy_true_hf) + K.abs(dx_pred_hf - dx_true_hf), axis=-1)

    # Weights
    w1 = 1.0
    w2 = 2.0

    return (w1 * K.mean(high_freq_loss)) + (w2 * K.mean(low_freq_loss))



