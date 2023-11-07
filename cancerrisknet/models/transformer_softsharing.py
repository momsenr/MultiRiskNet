import torch
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel
from cancerrisknet.models.factory import RegisterModel
from cancerrisknet.models.transformer import Transformer

@RegisterModel("transformer_softsharing")
class TransformerSoftSharing(AbstractRiskModel):
    def __init__(self, args):
        super(TransformerSoftSharing, self).__init__(args)
        self.transformer_pancreatic = Transformer(args)  # Transformer for pancreatic cancer
        self.transformer_ovarian = Transformer(args)     # Transformer for ovarian cancer
        self.args = args

    def encode_trajectory(self, embed_x, batch=None):
        embed_x_pancreatic = self.transformer_pancreatic.encode_trajectory(embed_x,batch)
        embed_x_ovarian = self.transformer_ovarian.encode_trajectory(embed_x, batch)
        seq= torch.cat((embed_x_pancreatic, embed_x_ovarian), dim=2)
        return seq

    def soft_sharing_loss(self):
        loss = 0
        for param_p, param_o in zip(self.transformer_pancreatic.parameters(), self.transformer_ovarian.parameters()):
            loss += torch.norm(param_p - param_o, p='fro')
        return loss * self.args.soft_sharing_lambda