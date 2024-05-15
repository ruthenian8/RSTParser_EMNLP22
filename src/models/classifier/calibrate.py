import torch
import csv


def csv_to_cm(file, vocab):
    # Initialize the confusion matrix tensor with zeros
    reader = csv.reader(open(file).read().strip().split("\n"))
    csv_data = list(reader)
    vocab_size = len(vocab)
    confusion_matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    
    # Extract the header (column labels)
    column_labels = csv_data[0][1:]
    
    # Populate the confusion matrix tensor
    for row in csv_data[1:]:
        row_label = row[0]
        if row_label not in vocab:
            continue
        row_index = vocab[row_label]
        for col_index, value in enumerate(row[1:]):
            col_label = column_labels[col_index]
            if col_label not in vocab:
                continue
            col_index = vocab[col_label]
            confusion_matrix[row_index, col_index] = int(value)
    
    return confusion_matrix


def calculate_calibrated_confidence(confusion_matrix, label_indices, confidence, num_classes):
    """
    Calculate calibrated confidence for an annotator's label based on a confusion matrix.
    
    Parameters:
        confusion_matrix (torch.Tensor): A confusion matrix of size CxC.
        label_indices (torch.Tensor): The primary label indices for which the confidence is calculated.
        confidence (float): Original confidence level of the annotator on the primary label.
        num_classes (int): Number of classes.
    
    Returns:
        torch.Tensor: The calibrated posterior probabilities P(y_i = l_{im} | a_{m}).
    """

    # Calculate prior probability P(y_i = l_{im} | c_{im})
    p_label = 1 / num_classes
    p_not_label = (num_classes - 1) / num_classes
    p_c_given_label = confidence
    p_c_given_not_label = 1 - confidence
    p_c = p_c_given_label * p_label + p_c_given_not_label * p_not_label
    p_hat_yi_given_c = (p_c_given_label * p_label) / p_c

    # Extract counts for P(a_m|y_i) and P(a_m|\neg y_i)
    true_pos = confusion_matrix[label_indices, label_indices]
    false_pos = confusion_matrix[:, label_indices].sum(dim=0) - true_pos
    false_neg = confusion_matrix[label_indices, :].sum(dim=1) - true_pos
    true_neg = confusion_matrix.sum() - true_pos - false_pos - false_neg

    # Calculate probabilities
    p_am_given_yi = true_pos.float() / (true_pos + false_neg).float()
    p_am_given_not_yi = false_pos.float() / (false_pos + true_neg).float()

    # Handle zero denominators
    p_am_given_yi[torch.isnan(p_am_given_yi)] = 0
    p_am_given_not_yi[torch.isnan(p_am_given_not_yi)] = 0

    # Calculate posterior probability P(y_i = l_{im} | a_{m})
    p_am = p_am_given_yi * p_hat_yi_given_c + p_am_given_not_yi * (1 - p_hat_yi_given_c)
    p_yi_given_am = (p_am_given_yi * p_hat_yi_given_c) / p_am
    p_yi_given_am[torch.isnan(p_yi_given_am)] = confidence  # Handle zero probabilities

    return p_yi_given_am


def generate_soft_labels(num_classes, primary_class_indices, calibrated_confidences):
    """
    Generate a soft label for a single annotator based on calibrated confidence.
    
    Parameters:
        num_classes (int): Total number of classes.
        primary_class_indices (torch.Tensor): Indices of the primary classes (0-based).
        calibrated_confidences (torch.Tensor): Calibrated confidences for the primary classes.
    
    Returns:
        torch.Tensor: Soft label vectors for the given annotator.
    """
    batch_size = primary_class_indices.size(0)
    soft_labels = torch.full((batch_size, num_classes), 0.0).to("cuda:0")
    fill_value = (1 - calibrated_confidences) / (num_classes - 1)
    soft_labels += fill_value.unsqueeze(1)

    # Set the calibrated confidence to the primary class
    soft_labels[torch.arange(batch_size), primary_class_indices] = calibrated_confidences
    
    return soft_labels


def merge_soft_labels(soft_labels):
    """
    Merge multiple soft labels into a final single soft label.
    
    Parameters:
        soft_labels (torch.Tensor): Tensor containing soft labels for each annotator in each row.
    
    Returns:
        torch.Tensor: Final soft label after averaging.
    """
    # Take the mean of soft labels along rows (i.e., across annotators)
    final_soft_label = torch.mean(soft_labels, dim=0)
    
    return final_soft_label


def get_soft_labels(conf_matrix, label_idx, label_2_idx, confidence, num_classes):
    """
    Generate a final soft label by combining two annotators' labels.
    
    Parameters:
        conf_matrix (torch.Tensor): Confusion matrix of size CxC.
        label_idx (torch.Tensor): Indices of the primary labels (0-based) for the first annotator.
        label_2_idx (torch.Tensor): Indices of the primary labels (0-based) for the second annotator.
        confidence (float): Original confidence level of the annotators on the primary label.
        num_classes (int): Number of classes.
    """
    calibrated_confidence = calculate_calibrated_confidence(conf_matrix, label_idx, confidence, num_classes)
    other_confidence = calculate_calibrated_confidence(conf_matrix.transpose(0, 1), label_2_idx, 1-confidence, num_classes)

    calibrated_confidences = torch.stack([calibrated_confidence, other_confidence], dim=1)  # Two annotators with their calibrated confidences for the primary class

    # Generate soft labels for each annotator
    soft_labels = torch.stack([
        generate_soft_labels(num_classes, indices, confidences)
        for indices, confidences in zip([label_idx, label_2_idx], calibrated_confidences.transpose(0, 1))
    ])

    # Merge soft labels into a final single soft label
    final_soft_label = merge_soft_labels(soft_labels)
    return final_soft_label
