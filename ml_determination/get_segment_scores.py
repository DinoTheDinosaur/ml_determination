import ml_determination.data
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM


class Scorer:

    def __init__(
        self,
        data_path='',
        model_path='',
        oov='<unk>', sent_boundary='<eos>', device='cpu'
    ):
        self.sent_boundary = sent_boundary
        self.oov = oov
        self.device = torch.device(device)

        print("Load vocabulary.")
        corpus = data.Corpus(data_path)
        self.ntokens = len(corpus.dictionary)
        self.vocab = corpus.dictionary.word2idx

        print("Load model and criterion.")
        with open(model_path, 'rb') as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.eval()
        self.criterion = torch.nn.NLLLoss(reduction='none')


    def get_input_and_target(self, hyps):
        batch_size = len(hyps)
        assert batch_size > 0

        # Preprocess input and target sequences
        inputs, outputs = [], []
        for hyp in hyps:
            input_string = self.sent_boundary + ' ' + hyp
            output_string = hyp + ' ' + self.sent_boundary
            input_ids, output_ids = [], []
            for word in input_string.split():
                try:
                    input_ids.append(self.vocab[word])
                except KeyError:
                    input_ids.append(self.vocab[self.oov])
            for word in output_string.split():
                try:
                    output_ids.append(self.vocab[word])
                except KeyError:
                    output_ids.append(self.vocab[self.oov])
            inputs.append(input_ids)
            outputs.append(output_ids)

        batch_lens = [len(seq) for seq in inputs]
        seq_lens = torch.LongTensor(batch_lens)
        max_len = max(batch_lens)

        # Zero padding for input and target sequences.
        data = torch.LongTensor(batch_size, max_len).zero_()
        target = torch.LongTensor(batch_size, max_len).zero_()
        for idx, seq_len in enumerate(batch_lens):
            data[idx, :seq_len] = torch.LongTensor(inputs[idx])
            target[idx, :seq_len] = torch.LongTensor(outputs[idx])
        data = data.t().contiguous().to(self.device)
        target = target.t().contiguous().view(-1).to(self.device)
        return data, target, seq_lens, inputs


    def compute_sentence_score(self, data, target):
        with torch.no_grad():
            output = self.model(data)
            loss = self.criterion(output.view(-1, self.ntokens), target)
            loss = torch.reshape(loss, data.size())
            loss = loss.t() # [batch_size, length]
        sent_scores = (loss.cpu().numpy() * 10000).astype(int) / 10000
        return sent_scores


    def compute_scores(self, sentence):
        data, targets, seq_lens, inputs = self.get_input_and_target([sentence])
        scores = self.compute_sentence_score(data, targets)[0][1:]
        return scores

class SegmentsScorerBERT:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.model = BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased")
        self.mask_index = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    def compute_scores(self, sentence):
        scores = []
        text = "[CLS] %s [SEP]" % sentence
        tokens = self.tokenizer.tokenize(text)
        input_indexes = []
        masked_indexes = []
        indexes = self.tokenizer.convert_tokens_to_ids(tokens)
        sentence_id = 0
        for i, token_index in enumerate(indexes):
            if token_index in [101,102]:
                continue
            input_indexes += [indexes[:i] + [self.mask_index] + indexes[i+1:]]
            masked_indexes += [(sentence_id, i, token_index)]
            sentence_id += 1
        tokens_tensor = torch.tensor(input_indexes)
        with torch.no_grad():
            outputs = self.model(tokens_tensor)[0]
        scores = np.array(list(outputs[index] for index in masked_indexes))
        return scores