from models.classifier import ClassifierBase
from models.classifier.linear import FeedForward
from data.doc import Doc
from data.batch import Batch
import torch
import torch.nn as nn


class AJClassifier(ClassifierBase):
    def __init__(self, hidden_dim: int, dropout_p: float = 0.2, *args, **kwargs):
        super(AJClassifier, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.org_embed = self.init_org_embeddings()

        self.head_vocab = self.DATASET.head_vocab
        self.parent_vocab = self.DATASET.fully_label_vocab
        self.child_vocab = self.DATASET.fully_label_vocab

        embed_dim = self.encoder.get_embed_dim() * 3
        feat_embed_dim = self.get_org_embedding_dim()
        embed_dim += feat_embed_dim

        self.out_linear_head = FeedForward(
            embed_dim, self.hidden_dim, len(self.head_vocab), self.dropout_p
        )
        self.out_linear_parent = FeedForward(
            embed_dim, self.hidden_dim, len(self.parent_vocab), self.dropout_p
        )
        self.out_linear_child = FeedForward(
            embed_dim, self.hidden_dim, len(self.child_vocab), self.dropout_p
        )

        self.xent_loss = nn.CrossEntropyLoss()


    @classmethod
    def params_from_config(cls, config):
        params = super().params_from_config(config)
        params.update(
            {
                "hidden_dim": config.hidden_dim,
                "dropout_p": config.dropout_p,
            }
        )
        return params

    def init_org_embeddings(self):
        if self.disable_org_feat:
            return None

        num_feat = 0
        if not self.disable_org_sent:
            num_feat += 17
        if not self.disable_org_para:
            num_feat += 11

        return nn.Embedding(num_feat * 2, 10)

    def training_loss(self, batch: Batch):
        doc = batch.doc
        spans = batch.span
        feats = batch.feat
        output = self(doc, spans, feats)

        loss_dict = self.compute_loss(output, batch)
        return loss_dict


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
        head_scores = self.out_linear_head(span_embeddings)
        parent_scores = self.out_linear_parent(span_embeddings)
        child_scores = self.out_linear_child(span_embeddings)

        output = {
            "head_scores": head_scores,
            "parent_scores": parent_scores,
            "child_scores": child_scores,
        }
        return output

    def compute_loss(self, output, batch: Batch):
        labels = batch.label
        head_idx = labels["act"]
        parent_idx = labels["nuc"]
        child_idx = labels["rel"]
        head_loss = self.xent_loss(output["head_scores"], head_idx)
        parent_loss = self.xent_loss(output["parent_scores"], parent_idx)
        child_loss = self.xent_loss(output["child_scores"], child_idx)

        loss = (head_loss + parent_loss + child_loss) / 3

        return {
            "loss": loss,
            "head_loss": head_loss,
            "parent_loss": parent_loss,
            "child_loss": child_loss,
        }

    def predict(self, document_embedding, span: dict, feat: dict):
        s1_emb = self.encoder.get_span_embedding(document_embedding, span["s1"])
        s2_emb = self.encoder.get_span_embedding(document_embedding, span["s2"])
        q1_emb = self.encoder.get_span_embedding(document_embedding, span["q1"])
        embedding = torch.cat((s1_emb, s2_emb, q1_emb), dim=0)
        if not self.disable_org_feat:
            org_emb = self.org_embed(feat["org"]).view(-1)
            embedding = torch.cat((embedding, org_emb), dim=0)

        head_scores = self.out_linear_head(embedding)
        parent_scores = self.out_linear_parent(embedding)
        child_scores = self.out_linear_child(embedding)
        return head_scores, parent_scores, child_scores