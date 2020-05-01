from tensorflow.keras import backend as K


def pred_accuracy(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.equal(y_true, y_pred))


def pred_abs_error(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.abs(y_pred - y_true))
