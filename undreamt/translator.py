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

from undreamt import data, devices

import random
import torch
import torch.nn as nn
from torch.autograd import Variable


class Translator:
    def __init__(self, encoder_embeddings, decoder_embeddings, generator, src_dictionary, trg_dictionary, encoder,
                 decoder, denoising=True, device=devices.default):
        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.generator = generator
        self.src_dictionary = src_dictionary
        self.trg_dictionary = trg_dictionary
        self.encoder = encoder
        self.decoder = decoder
        self.denoising = denoising
        self.device = device
        weight = device(torch.ones(generator.output_classes()))
        weight[data.PAD] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def _train(self, mode):
        self.encoder_embeddings.train(mode)
        self.decoder_embeddings.train(mode)
        self.generator.train(mode)
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.criterion.train(mode)

    def encode(self, sentences, train=False):
        self._train(train)
        ids, lengths = self.src_dictionary.sentences2ids(sentences, sos=False, eos=True)

        if train and self.denoising:  # Add order noise
            for i, length in enumerate(lengths):
                if length > 2:
                    for it in range(length//2):
                        j = random.randint(0, length-2)
                        ids[j][i], ids[j+1][i] = ids[j+1][i], ids[j][i]

        varids = self.device(Variable(torch.LongTensor(ids), requires_grad=False, volatile=not train))
        hidden = self.device(self.encoder.initial_hidden(len(sentences)))
        hidden, context = self.encoder(ids=varids, lengths=lengths, word_embeddings=self.encoder_embeddings, hidden=hidden)
        return hidden, context, lengths

    def mask(self, lengths):
        batch_size = len(lengths)
        max_length = max(lengths)
        if max_length == min(lengths):
            return None
        mask = torch.ByteTensor(batch_size, max_length).fill_(0)
        for i in range(batch_size):
            for j in range(lengths[i], max_length):
                mask[i, j] = 1
        return self.device(mask)

    def greedy(self, sentences, max_ratio=2, train=False):
        self._train(train)
        input_lengths = [len(data.tokenize(sentence)) for sentence in sentences]
        hidden, context, context_lengths = self.encode(sentences, train)
        context_mask = self.mask(context_lengths)
        translations = [[] for sentence in sentences]
        prev_words = len(sentences)*[data.SOS]
        pending = set(range(len(sentences)))
        output = self.device(self.decoder.initial_output(len(sentences)))
        while len(pending) > 0:
            var = self.device(Variable(torch.LongTensor([prev_words]), requires_grad=False))
            logprobs, hidden, output = self.decoder(var, len(sentences)*[1], self.decoder_embeddings, hidden, context, context_mask, output, self.generator)
            prev_words = logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            for i in pending.copy():
                if prev_words[i] == data.EOS:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    if len(translations[i]) >= max_ratio*input_lengths[i]:
                        pending.discard(i)
        return self.trg_dictionary.ids2sentences(translations)

    def beam_search(self, sentences, beam_size=12, max_ratio=2, train=False):
        self._train(train)
        batch_size = len(sentences)
        input_lengths = [len(data.tokenize(sentence)) for sentence in sentences]
        hidden, context, context_lengths = self.encode(sentences, train)
        translations = [[] for sentence in sentences]
        pending = set(range(batch_size))

        hidden = hidden.repeat(1, beam_size, 1)
        context = context.repeat(1, beam_size, 1)
        context_lengths *= beam_size
        context_mask = self.mask(context_lengths)
        ones = beam_size*batch_size*[1]
        prev_words = beam_size*batch_size*[data.SOS]
        output = self.device(self.decoder.initial_output(beam_size*batch_size))

        translation_scores = batch_size*[-float('inf')]
        hypotheses = batch_size*[(0.0, [])] + (beam_size-1)*batch_size*[(-float('inf'), [])]  # (score, translation)

        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            var = self.device(Variable(torch.LongTensor([prev_words]), requires_grad=False))
            logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask, output, self.generator)
            prev_words = logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()

            word_scores, words = logprobs.topk(k=beam_size+1, dim=2, sorted=False)
            word_scores = word_scores.squeeze(0).data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            words = words.squeeze(0).data.cpu().numpy().tolist()

            for sentence_index in pending.copy():
                candidates = []  # (score, index, word)
                for rank in range(beam_size):
                    index = sentence_index + rank*batch_size
                    for i in range(beam_size + 1):
                        word = words[index][i]
                        score = hypotheses[index][0] + word_scores[index][i]
                        if word != data.EOS:
                            candidates.append((score, index, word))
                        elif score > translation_scores[sentence_index]:
                            translations[sentence_index] = hypotheses[index][1] + [word]
                            translation_scores[sentence_index] = score
                best = []  # score, word, translation, hidden, output
                for score, current_index, word in sorted(candidates, reverse=True)[:beam_size]:
                    translation = hypotheses[current_index][1] + [word]
                    best.append((score, word, translation, hidden[:, current_index, :].data, output[current_index].data))
                for rank, (score, word, translation, h, o) in enumerate(best):
                    next_index = sentence_index + rank*batch_size
                    hypotheses[next_index] = (score, translation)
                    prev_words[next_index] = word
                    hidden[:, next_index, :] = h
                    output[next_index, :] = o
                if len(hypotheses[sentence_index][1]) >= max_ratio*input_lengths[sentence_index] or translation_scores[sentence_index] > hypotheses[sentence_index][0]:
                    pending.discard(sentence_index)
                    if len(translations[sentence_index]) == 0:
                        translations[sentence_index] = hypotheses[sentence_index][1]
                        translation_scores[sentence_index] = hypotheses[sentence_index][0]
        return self.trg_dictionary.ids2sentences(translations)

    def score(self, src, trg, train=False):
        self._train(train)

        # Check batch sizes
        if len(src) != len(trg):
            raise Exception('Sentence and hypothesis lengths do not match')

        # Encode
        hidden, context, context_lengths = self.encode(src, train)
        context_mask = self.mask(context_lengths)

        # Decode
        initial_output = self.device(self.decoder.initial_output(len(src)))
        input_ids, lengths = self.trg_dictionary.sentences2ids(trg, eos=False, sos=True)
        input_ids_var = self.device(Variable(torch.LongTensor(input_ids), requires_grad=False))
        logprobs, hidden, _ = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hidden, context, context_mask, initial_output, self.generator)

        # Compute loss
        output_ids, lengths = self.trg_dictionary.sentences2ids(trg, eos=True, sos=False)
        output_ids_var = self.device(Variable(torch.LongTensor(output_ids), requires_grad=False))
        loss = self.criterion(logprobs.view(-1, logprobs.size()[-1]), output_ids_var.view(-1))

        return loss