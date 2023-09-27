import torch
import torch.nn.functional as F

def get_multi_task_loss(logits, batch, args, task_weights=None):
    """
    Compute multi-task loss.

    Args:
        logits (torch.Tensor): Predicted logits for all tasks. Shape: [num_tasks, *]
        batch (dict): Batch containing labels and masks for all tasks.
        args: Arguments containing loss function choice.
        task_weights (list or torch.Tensor, optional): Weights for each task. If not provided,
                                                       all tasks are treated equally.

    Returns:
        torch.Tensor: Total multi-task loss.
    """

    y_seq = batch['y_seq']
    y_mask = batch['y_mask']
    # If no specific task weights are provided, assume equal weights for all tasks.
    if task_weights is None:
        task_weights = torch.ones(logits.shape[1]).to(logits.device)
    else:
        task_weights = torch.tensor(task_weights).to(logits.device)

    if args.loss_fn == 'binary_cross_entropy_with_logits':
        # Compute BCE loss for all tasks
        losses = F.binary_cross_entropy_with_logits(logits, y_seq, weight=y_mask, reduction='none')
        # Sum over the sequence dimension and then divide by the sum of the masks for each task
        losses = torch.sum(losses, dim=-1) / torch.sum(y_mask, dim=-1)
    elif args.loss_fn == 'mse':
        # Compute MSE loss for all tasks, adjust to sum the losses and then average over tasks
        losses = F.mse_loss(logits, y_seq, reduction='sum').div(logits.shape[1])
    else:
        raise Exception('Loss function is illegal or not found.')
    # Weighted sum of all task losses
    total_loss = torch.sum(task_weights * losses)

    return total_loss

def model_step(batch, models, train_model, args, task_weights=None):
    """
    Single step of running model on a batch x,y for multi-task learning and computing the loss.
    Returns various stats of this single forward and backward pass.

    Args:
        batch: whole batch dict, can be used by various special args
        models: dict of models. The main model, named "model" must return logit, hidden, activ for each task.
        train_model: Backward pass is computed if set to True.

    Returns:
        loss: scalar for loss on batch as a tensor
        preds: predicted labels as numpy array
        probs: softmax probabilities as numpy array
        golds: labels at the trajectory level, numpy array version of arg y
        patient_golds: labels at the patient level for each task
        pids: deidentified patient ids as a list of strings
        censor_times: feature rep for batch
        days_to_censor: the time before censorship as a tensor
        dates: the admission date as a tensor
    """
    logits = models[args.model_name](batch['x'], batch)
    loss = get_multi_task_loss(logits, batch, args,task_weights=task_weights)

    if train_model:
        loss.backward()

    # Use sigmoid for multi-label tasks and convert to numpy
    probs = torch.sigmoid(
        logits).cpu().data.numpy()  # Shape is T, B, len(args.month_endpoints) where T is number of tasks
    preds = probs > .5
    golds = batch['y'].data.cpu().numpy()  # This is a 2D tensor with tasks as the first dimension
    patient_golds = batch['future_cancer_tensor'].data.cpu().numpy() # This is a 2D tensor with tasks as the first dimension
    pids = batch['patient_id'].cpu().numpy()
    censor_times = batch['time_at_event'].cpu().numpy()
    days_to_censor = batch['days_to_censor'].cpu().numpy()
    dates = batch['admit_date'].cpu().numpy()

    return loss, preds, probs, golds, patient_golds, pids, censor_times, days_to_censor, dates
