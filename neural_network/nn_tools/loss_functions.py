import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall


def weighted_categorical_crossentropy(class_weights):
    def wce_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        weights_tensor = tf.reshape(weights_tensor, (1, -1))  # Shape: (1, num_classes)

        weights = tf.reduce_sum(weights_tensor * y_true, axis=-1)  # Shape: (batch_size,)
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * cross_entropy)

        return weighted_loss

    return wce_loss


def opt_specificity(optimal_threshold=0.5):
    def specificity(y_true, y_pred):
        # Convert predictions to binary using a threshold (default is 0.5)
        y_pred_binary = tf.cast(y_pred >= optimal_threshold, tf.float32)

        # Calculate True Negatives and False Positives
        true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
        false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)

        # Compute specificity
        specificity_value = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())

        return specificity_value

    return specificity


def opt_sensitivity(optimal_threshold=0.5):
    def sensitivity(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred >= optimal_threshold, tf.float32)

        true_positives = K.sum(K.cast(y_true * y_pred_binary, tf.float32))
        false_negatives = K.sum(K.cast(y_true * (1 - y_pred_binary), tf.float32))
        specificity_value = true_positives / (true_positives + false_negatives + K.epsilon())
        return specificity_value

    return sensitivity


def bal_accuracy(y_true, y_pred):
    # Convert predictions to binary using a threshold of 0.5 (or adjust as needed)
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    true_positives = K.sum(K.cast(y_true * y_pred_binary, tf.float32))
    true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), tf.float32))
    false_positives = K.sum(K.cast((1 - y_true) * y_pred_binary, tf.float32))
    false_negatives = K.sum(K.cast(y_true * (1 - y_pred_binary), tf.float32))

    # Calculate recall for each class
    recall_pos = tf.clip_by_value(true_positives / (true_positives + false_negatives + K.epsilon()),
                                  K.epsilon(), 1 - K.epsilon())
    recall_neg = tf.clip_by_value(true_negatives / (true_negatives + false_positives + K.epsilon()),
                                  K.epsilon(), 1 - K.epsilon())

    # Compute balanced accuracy
    balanced_acc = (recall_pos + recall_neg) / 2.0
    return (1 - balanced_acc) + K.epsilon()


def focal_loss(gamma=2.2, alpha=0.4):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        cross_entropy_pos = -y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        cross_entropy_neg = -(1 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1 - y_pred)

        # Weighted focal loss (balancing factor alpha)
        loss = tf.sqrt(alpha * cross_entropy_pos + (1 - alpha) * cross_entropy_neg)

        return tf.reduce_mean(loss)

    return focal_loss_fixed


def beta_f1(beta=1.0, opt_threshold=0.5):
    b2 = beta**2

    def f1_score(y_true, y_pred):
        # Threshold predictions
        y_pred = tf.cast(y_pred >= opt_threshold, tf.float32)

        # Calculate true positives, false positives, and false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp
        fn = tf.reduce_sum(y_true) - tp

        # Precision and Recall
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())

        # F-beta Score
        f1 = (1 + b2) * (precision * recall) / (b2 * precision + recall + tf.keras.backend.epsilon())

        return f1

    return f1_score


def negative_predictive_value(opt_threshold=0.5):
    def npv(y_true, y_pred):
        y_pred = tf.cast(y_pred >= opt_threshold, tf.float32)

        # Calculate true negatives, Calculate false negatives
        true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        # Compute Negative Predictive Value
        np_value = true_negatives / (true_negatives + false_negatives + tf.keras.backend.epsilon())

        return np_value

    return npv


def weighted_auc(class_weights):
    neg_weight = class_weights[0]
    pos_weight = class_weights[1]

    def auc_loss(y_true, y_pred):
        """
        AUC loss function: minimizes the pairwise ranking loss to approximate AUC.
        y_true: Ground truth labels, expected to be 0 or 1.
        y_pred: Predicted probabilities (output of the model).
        """
        y_true = tf.cast(y_true, tf.float32)

        # Separate positive and negative samples
        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]

        # Compute pairwise differences: positive - negative
        pairwise_diff = tf.expand_dims(pos_pred, axis=1) - tf.expand_dims(neg_pred, axis=0)

        # Sigmoid for differentiable approximation of the step function
        surrogate_loss = tf.nn.sigmoid(-pairwise_diff)

        # Apply weights
        pair_weights = pos_weight * neg_weight  # Pair weight for positive-negative combinations
        weighted_loss = surrogate_loss * pair_weights

        # Average loss over all pairs
        loss = tf.reduce_mean(weighted_loss)

        return loss

    return auc_loss


"""-------------------------------------------Combined Functions-----------------------------------------------------"""


def comb_focal_wce_f1(beta=1.0, opt_threshold=0.5, class_weights=(0.9, 1.5)):
    loss_wce_fn = weighted_categorical_crossentropy(class_weights)
    f1_score_fn = beta_f1(beta, opt_threshold)
    auc_fn = weighted_auc(class_weights)
    npv_fn = negative_predictive_value(opt_threshold)

    def combined_wl_loss(y_true, y_pred):
        # Calculate individual losses
        loss_fl = focal_loss()(y_true, y_pred)
        wce_loss = loss_wce_fn(y_true, y_true)
        f1_loss = tf.math.log1p(1 - f1_score_fn(y_true, y_pred))
        auc_loss = auc_fn(y_true, y_pred)
        npv_loss = tf.math.log1p(1 - npv_fn(y_true, y_pred))

        # Weighted sum of the losses
        weight_fl = 1.0
        weight_f1 = 1.5
        weight_wce = 1.0
        weight_auc = 1.5
        weight_npv = 2.0

        combined_loss_value = (weight_fl * loss_fl + weight_wce * wce_loss +
                               weight_f1 * f1_loss + weight_auc * auc_loss + weight_npv * npv_loss)

        return combined_loss_value

    return combined_wl_loss


def combined_wl_loss_fn2(class_weights):
    loss_wce = weighted_categorical_crossentropy(class_weights)

    def combined_wl_loss(y_true, y_pred):
        # Calculate individual losses
        loss_fl = focal_loss()(y_true, y_pred)
        wce_loss = loss_wce(y_true, y_true)
        bal_loss = bal_accuracy(y_true, y_pred)
        epsilon = tf.keras.backend.epsilon()

        # Weighted sum of the losses
        weight_fl = 1.0
        weight_bal = 1.5
        weight_wce = 2.5

        combined_loss_value = (
            weight_fl * loss_fl + weight_wce * wce_loss + weight_bal * bal_loss
        ) + epsilon

        return combined_loss_value

    return combined_wl_loss


def combined_wl_loss_fn3(class_weights):
    loss_wce = weighted_categorical_crossentropy(class_weights)

    def combined_wl_loss(y_true, y_pred):
        # Calculate individual losses
        loss_fl = focal_loss()(y_true, y_pred)
        wce_loss = loss_wce(y_true, y_true)
        bal_loss = bal_accuracy(y_true, y_pred)
        epsilon = tf.keras.backend.epsilon()

        # Weighted sum of the losses
        weight_fl = 1.0
        weight_bal = 1.5
        weight_wce = 2.5

        combined_loss_value = (
            weight_fl * loss_fl + weight_wce * wce_loss + weight_bal * bal_loss
        ) + epsilon

        return combined_loss_value

    return combined_wl_loss


def combined_wl_loss_fn():
    def combined_wl_loss(y_true, y_pred):
        # Calculate individual losses
        loss_f1 = focal_loss()(y_true, y_pred)
        loss_bal = 1 - bal_accuracy(y_true, y_pred)

        loss_f1 = tf.math.log1p(loss_f1)
        loss_bal = tf.math.log1p(loss_bal)

        # Weighted sum of the losses
        weight_fl = 0.65
        weight_bal = 0.35
        combined_loss_value = (
            weight_fl * loss_f1 + weight_bal * loss_bal
        )

        return combined_loss_value

    return combined_wl_loss


def weighted_huber_loss(delta=1.5, weight=1.5):
    def huber_loss(y_true, y_pred):
        error = (y_pred - y_true) * 10
        abs_error = tf.math.abs(error)

        # Penalize if prediction is on the wrong side of zero
        wrong_side_mask = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)

        # Huber loss calculation
        quadratic = tf.math.minimum(abs_error, delta)
        linear = abs_error - quadratic

        # Weighted loss for wrong side predictions
        h_loss = .5 * tf.math.exp(quadratic) ** 2 + delta * linear
        weight_h_loss = (1 + wrong_side_mask * (weight - 1)) * h_loss

        return tf.reduce_mean(weight_h_loss)

    return huber_loss


def weighted_jeff_loss(sign_adj=.75, sum_adj=.5):
    def jeff_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        error = tf.math.square(y_true - y_pred)
        same_side = y_true * y_pred
        true_pred_sum = tf.math.add(tf.math.abs(y_true), tf.math.abs(y_pred))
        same_side_mask = tf.cast(tf.sign(y_true) == tf.sign(y_pred), tf.float32)

        sign_penalty = tf.where(tf.less(same_side, 0.0), error*sign_adj, 0.0)
        sum_penalty = tf.where(tf.less(true_pred_sum, tf.math.abs(y_true)*2), error*sum_adj, 0.0)
        sum_penalty = sum_penalty * same_side_mask
        add_penalties = tf.math.add(sign_penalty, sum_penalty)

        return tf.math.add(error, add_penalties)

    return jeff_loss
