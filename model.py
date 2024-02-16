# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
# import paddle
# import paddle.nn as nn
import torch
import torch.nn as nn


class BaselineGruModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           batch_first=True)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        x = torch.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])
        x_enc = torch.cat((x_enc, x), 1)
        # x_enc = torch.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        # dec = torch.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out:]
        return sample  # [B, L, D]
    


class TransformerModel(nn.Module):
    """
    Desc: A simple Transformer model for time series forecasting
    """
    # def __init__(self, input_size, output_size, num_layers, hidden_size, dropout=0.1):
    def __init__(self, settings):
        super(TransformerModel, self).__init__()
        self.input_size = settings["in_var"]
        self.output_size = settings["out_var"]
        # self.hidden_size = 10
        self.num_layers = settings["lstm_layer"]
        self.output_len = settings["output_len"]

        # Define transformer layers (encoder and decoder)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.input_size, nhead=2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)


        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, src, tgt):
        # src: [batch_size, seq_len, input_size]
        # print("aaa")
        # print(f"src.shape = {src.shape}")
        x = torch.zeros([src.shape[0], 144, src.shape[2]])
        src = torch.cat((src, x), 1)
        # src = src.permute(1, 0, 2)  # Change to [batch_size, seq_len, input_size]
        # print("bbb")
        # print(f"src.shape = {src.shape}")
        # Encoder
        encoder_output = self.transformer_encoder(src)

        # Decoder
        decoder_output = self.transformer_decoder(tgt, encoder_output)
        output = self.fc(decoder_output)  # Map decoder output to output size
        
        # output = self.transformer_encoder(src)
        # print(f"output.shape = {output.shape}")
        # print(f"output[-1].shape = {output[-1].shape}")
        # output = self.fc(output)  # Take the last output of the transformer layers
        # print(f"output.shape = {output.shape}")
        return output
