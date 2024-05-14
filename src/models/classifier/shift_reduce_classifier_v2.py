import torch
import torch.nn as nn
import os

from data.batch import Batch
from data.doc import Doc
from models.classifier import ShiftReduceClassifierBase
from models.classifier.linear import FeedForward
from models.classifier.calibrate import get_soft_labels, csv_to_cm


class ShiftReduceClassifierV2(ShiftReduceClassifierBase):
    def __init__(self, use_soft_labels: bool, confusion_matrix_file: str, *args, **kwargs):
        super(ShiftReduceClassifierV2, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.act_vocab = self.DATASET.action_vocab
        self.ful_vocab = self.DATASET.fully_label_vocab

        embed_dim = self.encoder.get_embed_dim() * 3
        feat_embed_dim = self.get_org_embedding_dim()
        embed_dim += feat_embed_dim

        self.out_linear_act = FeedForward(
            embed_dim, self.hidden_dim, len(self.act_vocab), self.dropout_p
        )
        self.out_linear_label = FeedForward(
            embed_dim, self.hidden_dim, len(self.ful_vocab), self.dropout_p
        )

        assert self.act_vocab["<pad>"] == self.ful_vocab["<pad>"]
        pad_idx = self.act_vocab["<pad>"]
        self.pad_idx = pad_idx
        self.xent_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.xent_loss_full = nn.CrossEntropyLoss()

        self.use_soft_labels = use_soft_labels
        self.confusion_matrix = csv_to_cm(confusion_matrix_file, self.ful_vocab).to("cuda:0")

    @classmethod
    def params_from_config(cls, config):
        params = super().params_from_config(config)
        params.update(
            {
                "use_soft_labels": config.use_soft_labels,
                "confusion_matrix_file": os.path.join(
                    config.data_dir, config.confusion_matrix_file
                ),
            }
        )
        return params

    def forward(self, doc: Doc, spans: dict, feats: dict):
        document_embedding = self.encoder(doc)

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

        span_embeddings = torch.stack(span_embeddings, dim=0)

        # predict label scores for act_nuc and rel
        act_scores = self.out_linear_act(span_embeddings)
        label_scores = self.out_linear_label(span_embeddings)

        output = {
            "act_scores": act_scores,
            "ful_scores": label_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        labels = batch.label
        act_idx = labels["act"]
        ful_idx = labels["ful"]
        sec_idx = labels["sec"]
        calibrated_prob_scores = None
        act_loss = self.xent_loss(output["act_scores"], act_idx)

        if self.use_soft_labels:
            calibrated_prob_scores = get_soft_labels(
                self.confusion_matrix,
                ful_idx,
                sec_idx,
                self.DATASET.confidence,
                num_classes=len(self.ful_vocab),
            )
            ful_loss = self.xent_loss_full(output["ful_scores"], calibrated_prob_scores)
        else:
            ful_loss = self.xent_loss(output["ful_scores"], ful_idx)

        if torch.all(ful_idx == self.pad_idx):
            # if action is shift, there are no nuc and relation labels
            # and xent_loss return NaN.
            ful_loss = torch.zeros_like(ful_loss)

        loss = (act_loss + ful_loss) / 2

        return {
            "loss": loss,
            "act_loss": act_loss,
            "ful_loss": ful_loss,
        }

    def predict(self, document_embedding, span: dict, feat: dict):
        s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
        s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
        q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
        embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)
        if not self.disable_org_feat:
            org_emb = self.org_embed(feat["org"]).view(-1)
            embedding = torch.cat((embedding, org_emb), dim=0)

        act_scores = self.out_linear_act(embedding)
        label_scores = self.out_linear_label(embedding)
        return act_scores, label_scores
