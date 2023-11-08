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
        #not needed anymore since we embed the codes in the shared class
        del self.transformer_pancreatic.code_embed
        del self.transformer_ovarian.code_embed

    def encode_trajectory(self, embed_x, batch=None):
        embed_x_pancreatic = self.transformer_pancreatic.encode_trajectory(embed_x,batch)
        embed_x_ovarian = self.transformer_ovarian.encode_trajectory(embed_x, batch)
        return embed_x_pancreatic, embed_x_ovarian

    def soft_sharing_loss(self):
        loss = 0
        for param_p, param_o in zip(self.transformer_pancreatic.parameters(), self.transformer_ovarian.parameters()):
            diff=torch.norm(param_p - param_o, p='fro')
            absolute=torch.norm(param_p, p='fro') + 1e-10
            loss += diff/absolute
        return loss * self.args.soft_sharing_lambda