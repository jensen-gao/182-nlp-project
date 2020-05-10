from tensorflow.keras import backend as K


def ord_pred_accuracy(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.equal(y_true, y_pred))


def ord_pred_abs_error(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.abs(y_pred - y_true))


def pred_abs_error(y_true, y_pred):
    y_pred = K.cast(K.argmax(y_pred, axis=-1), y_pred.dtype)
    y_true = K.cast(K.sum(y_true, axis=-1), y_pred.dtype)
    return K.mean(K.abs(y_pred - y_true))
