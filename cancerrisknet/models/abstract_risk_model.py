import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import warnings
import numpy as np
from cancerrisknet.models.pools.factory import get_pool
from cancerrisknet.models.utils import MultiTaskCumulativeProbabilityLayer,CumulativeProbabilityLayer
from cancerrisknet.models.factory import RegisterModel


class AbstractRiskModel(nn.Module):
    """
        The overall abstract model framework for all model architectures. The model is consists of embedding layers,
        encoding layers, and prediction layers a discrete time survival objective.
    """

    def __init__(self, args):

        super(AbstractRiskModel, self).__init__()

        self.args = args
        self.vocab_size = len(args.code_to_index_map) + 1
        self.code_embed = nn.Embedding(self.vocab_size, args.hidden_dim, padding_idx=0)
        #kept_token_vec = torch.nn.Parameter(torch.ones([1, 1, 1]),  requires_grad=False)
        #self.register_parameter('kept_token_vec', kept_token_vec)

        self.pool = get_pool(args.pool_name)(args)
        self.dropout = nn.Dropout(p=args.dropout)

        if(args.model_name == 'transformer_softsharing' and args.transformer_forced_on_task == False):
            hidden_dim = 2 * args.hidden_dim + 1 if self.args.add_age_neuron else 2*args.hidden_dim
        else:
            hidden_dim = args.hidden_dim + 1 if self.args.add_age_neuron else args.hidden_dim
        
        #For transformer_softsharing: At a later stage, we could add a flag to choose if we want to independent or one shared layers (for the two tasks).
        #In the first case, we really force the two transformers to learn one task each. In this current implementation we just have two different
        #transformers, which could learn anything (but not necessarily one task each).
        if(args.transformer_forced_on_task):
            self.prob_of_failure_layer_pancreatic = CumulativeProbabilityLayer(hidden_dim, len(args.month_endpoints), args)
            self.prob_of_failure_layer_ovarian = CumulativeProbabilityLayer(hidden_dim, len(args.month_endpoints), args)
        else:
            self.prob_of_failure_layer = MultiTaskCumulativeProbabilityLayer(hidden_dim, len(args.month_endpoints), args)
            
        if args.use_uncertainty_loss_weights:
            self.log_vars = nn.Parameter(torch.full((args.num_tasks,), -1 / args.num_tasks, device='cuda'))


        if args.use_time_embed:
            if args.model_name != 'transformer':
                warnings.warn("[W] Time embedding here is designed for transformer only. "
                              "But it can work with {} too.".format(args.model_name))
            self.t_embed_add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
            self.t_embed_scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)

        if args.use_age_embed:
            self.a_embed_add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
            self.a_embed_scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)

    def condition_on_pos_embed(self, x, embed, embed_type='time'):
        if embed_type == 'time':
            return self.t_embed_scale_fc(embed) * x + self.t_embed_add_fc(embed)
        elif embed_type == 'age':
            return self.a_embed_scale_fc(embed) * x + self.a_embed_add_fc(embed)
        else:
            raise NotImplementedError("Embed type {} not supported".format(embed_type))
            
    def get_embeddings(self, x, batch=None):
        token_embed = self.code_embed(x)
        return token_embed

    def forward(self, x, batch=None):
        embed_x = self.get_embeddings(x, batch)

        if self.args.use_time_embed:
            time = batch['time_seq'].float()
            embed_x = self.condition_on_pos_embed(embed_x, time, 'time')

        if self.args.use_age_embed:
            age = batch['age_seq'].float()
            embed_x = self.condition_on_pos_embed(embed_x, age, 'age')

        if(self.args.model_name == 'transformer_softsharing'):
            #todo: can this be parallelized?
            seq_hidden1,seq_hidden2 = self.encode_trajectory(embed_x, batch)
            seq_hidden1 = seq_hidden1.transpose(1, 2)
            seq_hidden2 = seq_hidden2.transpose(1, 2)
            hidden1 = self.dropout(self.pool(seq_hidden1))
            hidden2 = self.dropout(self.pool(seq_hidden2))
            if(self.args.transformer_forced_on_task==False):
                hidden= torch.cat((hidden1,hidden2 ), dim=1)
        else:
            seq_hidden = self.encode_trajectory(embed_x, batch)
            seq_hidden = seq_hidden.transpose(1, 2)
            hidden = self.dropout(self.pool(seq_hidden))

        if self.args.add_age_neuron:
            age_in_year = batch['age']/365.
            age_in_year = age_in_year.unsqueeze(1)  # This makes age_in_year's shape [B, 1]
            if(self.args.transformer_forced_on_task==False):
                hidden = torch.cat((hidden, age_in_year), dim=1)  # Concatenate along the second dimension (feature dimension)
            else:
                hidden1 = torch.cat((hidden1, age_in_year), dim=1)
                hidden2 = torch.cat((hidden2, age_in_year), dim=1) 
        
        if(self.args.transformer_forced_on_task==False):
            logit = self.prob_of_failure_layer(hidden)
        else:
            logit1 = self.prob_of_failure_layer_pancreatic(hidden1)
            logit2 = self.prob_of_failure_layer_ovarian(hidden2)
            logit = torch.stack((logit1, logit2), dim=1)
        return logit
