import torch
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel
from cancerrisknet.models.factory import RegisterModel
from cancerrisknet.models.transformer import TransformerLayer

@RegisterModel("transformer_softsharing")
class TransformerSoftSharing(AbstractRiskModel):
    def __init__(self, args):
        super(TransformerSoftSharing, self).__init__(args)

        for layer in range(args.num_layers):
            transformer_layer_pancreatic = TransformerLayer(args)
            self.add_module('pancreatic_transformer_layer_{}'.format(layer), transformer_layer_pancreatic)
            transformer_layer_ovarian = TransformerLayer(args)
            self.add_module('ovarian_transformer_layer_{}'.format(layer), transformer_layer_ovarian)
        self.args = args


    def encode_trajectory(self, embed_x, batch=None):
        embed_x_pancreatic = embed_x
        for indx in range(self.args.num_layers):
            name = 'pancreatic_transformer_layer_{}'.format(indx)
            embed_x_pancreatic = self._modules[name](embed_x_pancreatic)

        embed_x_ovarian = embed_x
        for indx in range(self.args.num_layers):
            name = 'ovarian_transformer_layer_{}'.format(indx)
            embed_x_ovarian = self._modules[name](embed_x_ovarian)

        return embed_x_pancreatic, embed_x_ovarian

    def soft_sharing_loss(self):
        loss = 0
        #tood: make this work with more than one layer
        for param_p, param_o in zip(self.pancreatic_transformer_layer_0.parameters(), self.ovarian_transformer_layer_0.parameters()):
            diff=torch.norm(param_p - param_o, p='fro')
            absolute=torch.norm(param_p, p='fro') + 1e-10
            loss += diff/absolute
        return loss * self.args.soft_sharing_lambda