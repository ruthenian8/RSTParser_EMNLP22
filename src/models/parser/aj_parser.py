from typing import Optional, Tuple
import random

from models.parser import ParserBase
from data.tree import AttachTree, RSTTree
from data.doc import Doc
from data.dataset import Dataset
from models.parser.organization_feature import OrganizationFeature as OrgFeat
from models.parser.utils import batch_iter

from supar.models.const.aj.transform import AttachJuxtaposeTree

import torch


class AJParser(ParserBase):
    def __init__(self, classifier):
        super(AJParser, self).__init__(classifier)

    def generate_action_sequence(self, tree: AttachTree):
        actions = AttachJuxtaposeTree.tree2action(tree)
        head_list, parent_list, child_list = [], [], []
        for action in actions:
            head_list.append(action[0])
            parent_list.append(action[1])
            child_list.append(action[2])
        return head_list, parent_list, child_list

    def generate_training_samples(
        self,
        dataset: Dataset,
        unit_type: str,
        batch_size: Optional[int] = None,
    ):
        head_vocab = dataset.head_vocab
        parent_vocab = dataset.fully_label_vocab
        child_vocab = dataset.fully_label_vocab

        samples = []
        doc: Doc
        for doc in dataset:
            tree = doc.tree

            head_list, parent_list, child_list = self.generate_action_sequence(tree)
            xs, ys, fs = [], [], []
            for idx, head, parent, child in zip(range(len(head_list)), head_list, parent_list, child_list):
                s1, s2, q1 = (0, idx), (idx+1, idx+1), (0, idx+1)
                head_idx = head_vocab[str(head)]
                parent_idx = parent_vocab[parent]
                child_idx = child_vocab[child]
                org_feat = self.get_organization_features(s1, s2, q1, doc)
                xs.append({"s1": s1, "s2": s2, "q1": q1})
                ys.append({"head": head_idx, "parent": parent_idx, "child": child_idx})
                fs.append({"org": org_feat})

            if unit_type == "document":
                samples.append({"doc": doc, "span": xs, "label": ys, "feat": fs})
            elif unit_type == "span":
                for x, y, f in zip(xs, ys, fs):
                    samples.append({"doc": doc, "span": x, "label": y, "feat": f})
            elif unit_type == "span_fast":
                assert batch_size > 1
                # should use Trainer.reload_dataloaders_every_n_epochs=1
                indices = list(range(len(xs)))
                random.shuffle(indices)
                xs = [xs[i] for i in indices]
                ys = [ys[i] for i in indices]
                fs = [fs[i] for i in indices]
                for feats in batch_iter(list(zip(xs, ys, fs)), batch_size):
                    b_xs, b_ys, b_fs = list(zip(*feats))
                    samples.append({"doc": doc, "span": b_xs, "label": b_ys, "feat": b_fs})
            else:
                raise ValueError("Invalid batch unit_type ({})".format(unit_type))

        return samples

    def select_action_and_labels(self, bert_output, span, feat, state, gold_act=None):
        head_vocab = self.classifier.head_vocab
        parent_vocab = self.classifier.parent_vocab
        child_vocab = self.classifier.child_vocab

        head_scores, parent_scores, child_scores = self.classifier.predict(bert_output, span, feat)

        # select allowed action
        head = head_vocab.lookup_token(torch.argmax(head_scores))
        parent = parent_vocab.lookup_token(torch.argmax(parent_scores))
        child = child_vocab.lookup_token(torch.argmax(child_scores))

        return head, parent, child


    def parse(self, doc: Doc):
        bert_output = self.classifier.encoder(doc)
        action_sequence = []
        num_leaves = len(doc.edus)
        for idx in range(num_leaves):
            s1, s2, q1 = (0, idx), (idx+1, idx+1), (0, idx+1)

            span = {"s1": s1, "s2": s2, "q1": q1}
            feat = {"org": self.get_organization_features(s1, s2, q1, doc, self.classifier.device)}

            # predict action and labels
            head, parent, child = self.select_action_and_labels(bert_output, span, feat, None)
            action_sequence.append((head, parent, child))

        new_tree = AttachJuxtaposeTree.totree(list(map(str, range(num_leaves))), "S")
        new_tree = AttachJuxtaposeTree.action2tree(new_tree, action_sequence)
        return new_tree

    def parse_with_naked_tree(self, doc: Doc, tree: RSTTree | AttachTree):
        return super().parse_with_naked_tree(doc, tree)

    def get_organization_features(
        self, s1: Tuple[int], s2: Tuple[int], q1: Tuple[int], doc: Doc, device=None
    ):
        # span == (-1, -1) -> edus = []
        edus = doc.edus
        s1_edus = edus[slice(*s1)]
        s2_edus = edus[slice(*s2)]
        q1_edus = edus[slice(*q1)]

        # init features
        features = []

        if not self.classifier.disable_org_sent:
            # for Stack 1 and Stack2
            features.append(OrgFeat.IsSameSent(s2_edus, s1_edus))
            features.append(OrgFeat.IsContinueSent(s2_edus, s1_edus))

            # for Stack 1 and Queue 1
            features.append(OrgFeat.IsSameSent(s1_edus, q1_edus))
            features.append(OrgFeat.IsContinueSent(s1_edus, q1_edus))

            # for Stack 1, 2 and Queue 1
            features.append(
                OrgFeat.IsSameSent(s2_edus, s1_edus) & OrgFeat.IsSameSent(s1_edus, q1_edus)
            )

            # starts and ends a sentence
            features.append(OrgFeat.IsStartSent(s1_edus))
            features.append(OrgFeat.IsStartSent(s2_edus))
            features.append(OrgFeat.IsStartSent(q1_edus))
            features.append(OrgFeat.IsEndSent(s1_edus))
            features.append(OrgFeat.IsEndSent(s2_edus))
            features.append(OrgFeat.IsEndSent(q1_edus))

            # starts and ends a document
            features.append(OrgFeat.IsStartDoc(s1_edus))
            features.append(OrgFeat.IsStartDoc(s2_edus))
            features.append(OrgFeat.IsStartDoc(q1_edus))
            features.append(OrgFeat.IsEndDoc(s1_edus))
            features.append(OrgFeat.IsEndDoc(s2_edus))
            features.append(OrgFeat.IsEndDoc(q1_edus))

        if not self.classifier.disable_org_para:
            # for Stack 1 and Stack2
            features.append(OrgFeat.IsSamePara(s2_edus, s1_edus))
            features.append(OrgFeat.IsContinuePara(s2_edus, s1_edus))

            # for Stack 1 and Queue 1
            features.append(OrgFeat.IsSamePara(s1_edus, q1_edus))
            features.append(OrgFeat.IsContinuePara(s1_edus, q1_edus))

            # for Stack 1, 2 and Queue 1
            features.append(
                OrgFeat.IsSamePara(s2_edus, s1_edus) & OrgFeat.IsSamePara(s1_edus, q1_edus)
            )

            # starts and ends a paragraph
            features.append(OrgFeat.IsStartPara(s1_edus))
            features.append(OrgFeat.IsStartPara(s2_edus))
            features.append(OrgFeat.IsStartPara(q1_edus))
            features.append(OrgFeat.IsEndPara(s1_edus))
            features.append(OrgFeat.IsEndPara(s2_edus))
            features.append(OrgFeat.IsEndPara(q1_edus))

        # convert to index
        bias = torch.tensor([2 * i for i in range(len(features))], dtype=torch.long, device=device)
        features = torch.tensor(features, dtype=torch.long, device=device)
        return bias + features
