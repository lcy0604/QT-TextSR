import torch
import torch.nn.functional as F

from .base import BaseConvertor

class CTCConvertor(BaseConvertor):
    """Convert between text, index and tensor for CTC loss-based pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none, the file
            is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        lower (bool): If True, convert original string to lower case.
        max_seq_len (int): Max length of text. All text longer than
            max_seq_len will be truncated, and shorter text will be padded
            with `BLK` token.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 lower=False,
                 max_seq_len=25,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.update_dict()

    def update_dict(self):
        # CTC-blank
        blank_token = '<BLK>'
        self.blank_idx = 0
        self.idx2char.insert(0, blank_token)

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append('<UKN>')
            self.unknown_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """Convert text-string to ctc-loss input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            dict (str: tensor | list[tensor]):
                targets (tensor): Tensor of size (N, T), where N is the
                    batch size, and T is the max length of target.
                target_lengths (tensor): torch.IntTensot([5,5]).
        """

        indexes = self.str2idx(strings)
        length = [len(index) for index in indexes]
        batch_text = torch.LongTensor(len(indexes),
                                      self.max_seq_len).fill_(self.blank_idx)
        for i, text in enumerate(indexes):
            text = list(text)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len]
            batch_text[i, :len(text)] = torch.LongTensor(text)

        return batch_text, torch.IntTensor(length)

    def tensor2idx(self, output, topk=1, return_topk=False):
        """Convert model output tensor to index-list.
        Args:
            output (tensor): The model outputs with size: N * T * C.
            img_metas (list[dict]): Each dict contains one image info.
            topk (int): The highest k classes to be returned.
            return_topk (bool): Whether to return topk or just top1.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]]
                (
                    indexes_topk (list[list[list[int]->len=topk]]):
                    scores_topk (list[list[list[float]->len=topk]])
                ).
        """

        assert isinstance(topk, int)
        assert topk >= 1

        batch_size = output.size(0)
        output = F.softmax(output, dim=2)
        output = output.cpu().detach()
        batch_topk_value, batch_topk_idx = output.topk(topk, dim=2)
        batch_max_idx = batch_topk_idx[:, :, 0]
        scores_topk, indexes_topk = [], []
        scores, indexes = [], []
        feat_len = output.size(1)
        for b in range(batch_size):
            decode_len = feat_len
            pred = batch_max_idx[b, :]
            select_idx = []
            prev_idx = self.blank_idx
            for t in range(decode_len):
                tmp_value = pred[t].item()
                if tmp_value not in (prev_idx, self.blank_idx):
                    select_idx.append(t)
                prev_idx = tmp_value
            select_idx = torch.LongTensor(select_idx)
            topk_value = torch.index_select(batch_topk_value[b, :, :], 0,
                                            select_idx)  # valid_seqlen * topk
            topk_idx = torch.index_select(batch_topk_idx[b, :, :], 0,
                                          select_idx)
            topk_idx_list, topk_value_list = topk_idx.numpy().tolist(
            ), topk_value.numpy().tolist()
            indexes_topk.append(topk_idx_list)
            scores_topk.append(topk_value_list)
            indexes.append([x[0] for x in topk_idx_list])
            scores.append([x[0] for x in topk_value_list])

        if return_topk:
            return indexes_topk, scores_topk

        return indexes, scores
