# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from undreamt import data
from undreamt.attention import GlobalAttention

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNAttentionDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, layers=1, dropout=0, input_feeding=True):
        super(RNNAttentionDecoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size, padding_idx=0)
        self.attention = GlobalAttention(hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        self.input_size = embedding_size + hidden_size if input_feeding else embedding_size
        self.stacked_rnn = StackedGRU(self.input_size, hidden_size, layers=layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, lengths, word_embeddings, hidden, context, context_mask, prev_output, generator):
        embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        output = prev_output
        scores = []
        for emb in embeddings.split(1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(0), output], 1)
            else:
                input = emb.squeeze(0)
            output, hidden = self.stacked_rnn(input, hidden)
            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            scores.append(generator(output))
        return torch.stack(scores), hidden, output

    def initial_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)


# Based on OpenNMT-py
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1