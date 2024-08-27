# tiktoken
import torch
import torch.nn as nn


class MultiHeadAttention(nn.module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        """

        :param x: input text vectors batch
        :return: context vectors for next token prediction
        """

        query = ""
        key = ""
        values = ""



        return
