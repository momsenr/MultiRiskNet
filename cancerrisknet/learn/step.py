import torch
import torch.nn.functional as F

def get_multi_task_loss(logits, batch, args, log_vars=None,smart=False,smart_verbose=False):
    """
    Compute multi-task loss with uncertainty.

    Args:
        logits (torch.Tensor): Predicted logits for all tasks. Shape: [num_tasks, *]
        batch (dict): Batch containing labels and masks for all tasks.
        args: Arguments containing loss function choice.
        log_vars (torch.Tensor): Logarithm of the uncertainty for each task.

    Returns:
        torch.Tensor: Total multi-task loss.
    """
    y_seq = batch['y_seq']
    y_mask = batch['y_mask']

    if args.loss_fn == 'binary_cross_entropy_with_logits':
        # Compute BCE loss for all tasks
        losses = F.binary_cross_entropy_with_logits(logits, y_seq, weight=y_mask, reduction='none')
        
        if args.focal_loss_gamma != 0:
            p = torch.sigmoid(logits)
            p_t = p * y_seq + (1 - p) * (1 - y_seq)
            losses = losses * ((1 - p_t) ** args.focal_loss_gamma)

        # Sum over the sequence dimension and then divide by the sum of the masks for each task
        losses = torch.sum(losses, dim=(0,2)) / torch.sum(y_mask, dim=(0,2))

    elif args.loss_fn == 'mse':
        # Compute MSE loss for all tasks, adjust to sum the losses and then average over tasks
        losses = F.mse_loss(logits, y_seq, reduction='sum').div(logits.shape[1])
    else:
        raise Exception('Loss function is illegal or not found.')

    if log_vars is not None:
        losses = torch.exp(-log_vars).to(logits.device) * losses + log_vars
        final_loss= torch.sum(losses)
    if(smart==True):
        final_loss = torch.max(losses)
        if(smart_verbose==True):
            print("losses:",losses)
    else:
        final_loss = torch.sum(losses)/args.num_tasks
    
    return final_loss


def model_step(batch, models, train_model, args,smart_loss=False,smart_verbose=False):
    """
    Single step of running model on a batch x,y for multi-task learning and computing the loss.
    Returns various stats of this single forward and backward pass.

    Args:
        batch: whole batch dict, can be used by various special args
        models: dict of models. The main model, named "model" must return logit, hidden, activ for each task.
        train_model: Backward pass is computed if set to True.

    Returns:
        loss: scalar for loss on batch as a tensor
        probs: softmax probabilities as numpy array
    """
    logits = models[args.model_name](batch['x'], batch)
    loss = get_multi_class_loss(logits, batch, args)

    if train_model:
        loss.backward()

    probs = F.softmax(logits, dim=1).cpu().data.numpy()

    return loss.cpu().data.item(), probs

def get_multi_class_loss(logits, batch, args):
    """
    Compute loss

    Args:

    Returns:
        torch.Tensor: Total multi-task loss.
    """
    y_seq = batch['y_seq']
    y_mask = batch['y_mask']
    B, C, T = logits.shape


    # Convert y_seq to long type, move to batch prerprocess later
    y_seq = y_seq.long()

    # Flatten the logits, mask and labels for use in cross_entropy
    logits_flat = logits.transpose(1, 2).reshape(B * T, C)
    y_seq_flat = y_seq.view(-1)
    y_mask_flat = y_mask.view(-1)
    
    # Apply the mask to select the relevant elements
    logits_masked = logits_flat[y_mask_flat]
    y_seq_masked = y_seq_flat[y_mask_flat]
    
    if(args.loss_weights=='time'):
        # Create a weight tensor for different timepoints
        weights = torch.tensor([1.5, 1.5, 1.5, 1, 1], dtype=torch.float32, device=logits.device)
        # Repeat the weights for each batch and flatten
        weights_repeated = weights.repeat(B, 1).flatten()
        # Apply mask to the weights
        weights_masked = weights_repeated[y_mask_flat]

        # Compute the loss with weights
        total_loss = F.cross_entropy(logits_masked, y_seq_masked, reduction='none')
        # Apply the weights and take the mean
        weighted_loss = (total_loss * weights_masked).mean()

        return weighted_loss
    elif args.loss_weights == 'class_PC':
        # Define class-based weights
        class_weights = torch.tensor([1.7, 1, 1.7], dtype=torch.float32, device=logits.device)

        # Compute the loss with class-based weights
        class_weighted_loss = F.cross_entropy(logits_masked, y_seq_masked, weight=class_weights)

        return class_weighted_loss
    elif args.loss_weights == 'class':
        # Define class-based weights
        class_weights = torch.tensor([2, 1, 2], dtype=torch.float32, device=logits.device)

        # Compute the loss with class-based weights
        class_weighted_loss = F.cross_entropy(logits_masked, y_seq_masked, weight=class_weights)

        return class_weighted_loss
    else:
        total_loss = F.cross_entropy(logits_masked, y_seq_masked )
        return total_loss
