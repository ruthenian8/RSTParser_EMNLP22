"""
Multitask classifier based on the shift-reduce classifier version 1.
"""
import torch
import torch.nn as nn

from data.batch import Batch
from data.doc import Doc
from models.classifier import ShiftReduceClassifierBase
from models.classifier.linear import FeedForward


class MultitaskSRClassifierV1(ShiftReduceClassifierBase):
    def __init__(self, *args, **kwargs):
        super(MultitaskSRClassifierV1, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.act_vocab = self.DATASET.action_vocab
        self.nuc_vocab = self.DATASET.nucleus_vocab
        self.rel_vocab = self.DATASET.relation_vocab

        embed_dim = self.encoder.get_embed_dim() * 3
        feat_embed_dim = self.get_org_embedding_dim()
        embed_dim += feat_embed_dim

        self.out_linear_action = FeedForward(
            embed_dim, self.hidden_dim, len(self.act_vocab), self.dropout_p
        )
        self.out_linear_nucleus = FeedForward(
            embed_dim, self.hidden_dim, len(self.nuc_vocab), self.dropout_p
        )
        self.out_linear_relation = FeedForward(
            embed_dim, self.hidden_dim, len(self.rel_vocab), self.dropout_p
        )
        self.out_subset = FeedForward(
            self.encoder.get_embed_dim(), self.hidden_dim, 2, self.dropout_p
        )

        assert self.act_vocab["<pad>"] == self.nuc_vocab["<pad>"] == self.rel_vocab["<pad>"]
        pad_idx = self.act_vocab["<pad>"]
        self.pad_idx = pad_idx
        self.xent_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.binary_xent_loss = nn.CrossEntropyLoss()

    def forward(self, doc: Doc, spans: dict, feats: dict):
        document_embedding = self.encoder(doc)
        cls_embeddings = document_embedding["cls_embeddings"]
        # do averaging for special token embeddings
        # special_token_embeddings = document_embedding["special_token_embeddings"]
        special_token_embeddings = (cls_embeddings + document_embedding["special_token_embeddings"]) / 2
        span_embeddings = []
        for span, feat in zip(spans, feats):
            s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
            s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
            q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
            embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)

            if not self.disable_org_feat:
                org_emb = self.org_embed(feat["org"]).view(-1)
                embedding = torch.cat((embedding, org_emb), dim=0)

            span_embeddings.append(embedding)
        special_token_embeddings = special_token_embeddings.repeat(len(span_embeddings), 1)
        span_embeddings = torch.stack(span_embeddings, dim=0)
        # predict label scores for act_nuc and rel
        act_scores = self.out_linear_action(span_embeddings)
        nuc_scores = self.out_linear_nucleus(span_embeddings)
        rel_scores = self.out_linear_relation(span_embeddings)
        # conditionally predict subset scores
        subset_scores = self.out_subset(special_token_embeddings) if self.model_subsets else None

        output = {
            "act_scores": act_scores,
            "nuc_scores": nuc_scores,
            "rel_scores": rel_scores,
            "subset_scores": subset_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        labels = batch.label
        act_idx = labels["act"]
        nuc_idx = labels["nuc"]
        rel_idx = labels["rel"]
        # add subset loss
        subset_idx = labels["subset"]
        subset_loss = self.binary_xent_loss(output["subset_scores"], subset_idx) if self.model_subsets else torch.zeros_like(1)

        act_loss = self.xent_loss(output["act_scores"], act_idx)
        nuc_loss = self.xent_loss(output["nuc_scores"], nuc_idx)
        rel_loss = self.xent_loss(output["rel_scores"], rel_idx)
        if torch.all(nuc_idx == self.pad_idx):
            # if action is shift, there are no nuc and relation labels
            # and xent_loss return NaN.
            nuc_loss = torch.zeros_like(nuc_loss)
            rel_loss = torch.zeros_like(rel_loss)

        # add subset loss
        if self.model_subsets:
            loss = (act_loss + nuc_loss + rel_loss + subset_loss) / 4
        else:
            loss = (act_loss + nuc_loss + rel_loss) / 3

        return {
            "loss": loss,
            # add subset loss
            "subset_loss": subset_loss,
            "act_loss": act_loss,
            "nuc_loss": nuc_loss,
            "rel_loss": rel_loss,
        }

    def predict(self, document_embedding, span: dict, feat: dict):
        s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
        s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
        q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
        embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)
        if not self.disable_org_feat:
            org_emb = self.org_embed(feat["org"]).view(-1)
            embedding = torch.cat((embedding, org_emb), dim=0)

        act_scores = self.out_linear_action(embedding)
        nuc_scores = self.out_linear_nucleus(embedding)
        rel_scores = self.out_linear_relation(embedding)
        subset_scores = self.out_subset(embedding) if self.model_subsets else None

        return act_scores, nuc_scores, rel_scores, subset_scores
