import torch
import torch.nn as nn

class CumulativeProbabilityLayer(nn.Module):
    """
        The cumulative layer which defines the monotonically increasing risk scores.
    """
    def __init__(self, num_features, max_followup, args):
        super(CumulativeProbabilityLayer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features,  max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        if(args.enforce_strict_monotonicity):
            self.monotonicity_activation = nn.Softplus()
        else:
            self.monotonicity_activation = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.triu(mask, diagonal=0)
        self.register_buffer('upper_triangular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.monotonicity_activation(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triangular_mask  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob


class OneHotLayer(nn.Module):
    """
        One-hot embedding for categorical inputs.
    """
    def __init__(self, num_classes, padding_idx):
        super(OneHotLayer, self).__init__()
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, num_classes, padding_idx=padding_idx)
        self.embed.weight.data = torch.eye(num_classes)
        self.embed.weight.requires_grad_(False)

    def forward(self, x):
        return self.embed(x)


class AttributionModel(nn.Module):
    """
        Model wrapper for posthoc attribution analyses.
    """

    def __init__(self, model, args):
        super().__init__()
        if isinstance(model, dict):
            self.model = model[args.model_name]
        else:
            self.model = model

    def forward(self, x, age_seq, time_seq, age, batch):
        #Here we need to copy the batch and add the age_seq and time_seq, since this forward pass
        #will be called several times by the LayerIntegratedGradients.attribute module with variable batch sizes.
        #Therefore, we cannot simpy pass batch, but need to pass a copy of it.
        batch_copy=batch.copy()
        batch_copy['age_seq'] = age_seq
        batch_copy['time_seq'] = time_seq
        batch_copy['age'] = age

        y = self.model(x, batch=batch_copy)
        return y

class MultiTaskCumulativeProbabilityLayer(nn.Module):
    """
        The cumulative layer for multi-task learning which defines the
        monotonically increasing risk scores for each task.
    """

    def __init__(self, num_features, max_followup, args):
        super(MultiTaskCumulativeProbabilityLayer, self).__init__()
        self.args = args

        # Vectorized task-specific hazard functions and base hazard functions
        self.hazard_fcs = nn.Linear(num_features, self.args.num_tasks * max_followup)
        self.base_hazard_fcs = nn.Linear(num_features, self.args.num_tasks)

        if (args.enforce_strict_monotonicity):
            self.monotonicity_activation = nn.Softplus()
        else:
            self.monotonicity_activation = nn.ReLU(inplace=True)

        # Adjusted mask for multiple tasks
        mask = torch.ones([max_followup, max_followup])
        mask = torch.triu(mask, diagonal=0).unsqueeze(0).repeat(self.args.num_tasks, 1, 1)
        self.register_buffer('upper_triangular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fcs(x)
        pos_hazard = self.monotonicity_activation(raw_hazard)
        return pos_hazard.view(-1, self.args.num_tasks,
                               raw_hazard.shape[1] // self.args.num_tasks)  # Reshape to [B, num_tasks, max_followup]

    def forward(self, x):
        hazards_output = self.hazards(x)
        B, _, T = hazards_output.size()

        # Vectorized computation of expanded hazards
        expanded_hazards = hazards_output.unsqueeze(-1).expand(B, self.args.num_tasks, T, T)
        masked_hazards = expanded_hazards * self.upper_triangular_mask

        cum_prob = torch.sum(masked_hazards, dim=2) + self.base_hazard_fcs(x).view(B, self.args.num_tasks, 1)

        return cum_prob


class MultiClassCumulativeProbabilityLayer(nn.Module):
    """
        The cumulative layer for multi-task learning which defines the
        monotonically increasing risk scores for each task.
    """

    def __init__(self, num_features, max_followup, args):
        super(MultiClassCumulativeProbabilityLayer, self).__init__()
        self.args = args
        self.num_classes = args.num_tasks + 1

        self.class_probabilities_fcs = nn.Linear(num_features, self.num_classes * max_followup)

    def forward(self, x):
        preds = self.class_probabilities_fcs(x)
        return preds.view(-1, self.num_classes,
                               preds.shape[1] // self.num_classes)
 
