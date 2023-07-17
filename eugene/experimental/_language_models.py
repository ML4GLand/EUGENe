import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from lego_blocks import FixedPositionalEncoding, Fluffy, Sherlock, Megatron
from pytorch_lightning import LightningModule

logging.getLogger().setLevel(logging.INFO)
from torch.optim.lr_scheduler import StepLR


class SITh(nn.Module):
    def __init__(
        self,
        maxSeqLen: int,
        modelDim: int,
        mlpDim: int,
        depth: int,
        attnHeads: int,
        attnHeadDim: int,
        multitaskOutputs: list,
        trainingObjective: str,
        embeddingDropout: float = 0.0,
        dropout: float = 0.0,
        aggr: str = "context",
        vocabSize: int = 6,
        posEnc: str = "learned",
    ):
        super(SITh, self).__init__()

        assert trainingObjective in {"self-supervised", "supervised"}

        logging.info("Training objective selected: {}".format(trainingObjective))
        self.trainingObjective = trainingObjective

        self.nukTuk = nn.Embedding(
            vocabSize, modelDim
        )  # Project vocab index to modelDim. Why nukTuk? Because we have nucleotides (nuks) and we are tokenizing (tuk)/vectorizing them. Take down unalaq nuktuk!

        self.context = nn.Parameter(
            torch.randn(1, 1, modelDim)
        )  #'[CLS]' token representation

        if posEnc == "learned":
            logging.info("Utilizing FULLY-LEARNED positional encodings..")
            self.positionalEncoding = nn.Parameter(
                torch.randn(1, maxSeqLen + 1, modelDim)
            )

        else:
            positionalEncoding = FixedPositionalEncoding(
                maxSeqLen=maxSeqLen, modelDim=modelDim
            )  # This is one dimension less - no context vector
            positionalEncoding = torch.cat(
                [torch.zeros(1, modelDim), positionalEncoding], axis=0
            )  # Prepend 0 vector for context token - no sequence position

            if posEnc == "joint":
                logging.info(
                    "Utilizing LEARNED positional encodings on an initialized SIN-COS embedding"
                )
                self.positionalEncoding = nn.Parameter(positionalEncoding.unsqueeze(0))

            else:
                logging.info("Utilizing FIXED (sin-cos) positional encodings")
                self.register_buffer(
                    "positionalEncoding",
                    positionalEncoding.unsqueeze(0),
                    persistent=False,
                )  # Ensures pushed to device; don't need saved as part of state_dict()

        self.embeddingDropout = nn.Dropout(embeddingDropout)
        self.preTransformerLayerNorm = nn.LayerNorm(modelDim)
        self.transformer = Megatron(
            modelDim=modelDim,
            blocks=depth,
            mlpDim=mlpDim,
            attnHeadDim=attnHeadDim,
            numHeads=attnHeads,
            dropout=dropout,
        )
        self.aggr = aggr  # how want to predict - on the context/CLS vector ("context") or mean pooling

        # CLS BERT-Style Hidden Layer
        self.clsHidden = nn.Sequential(
            nn.LayerNorm(modelDim), nn.Linear(modelDim, modelDim), nn.GELU()
        )  # technically BERT uses nn.Tanh(), using gelu here instead

        if trainingObjective == "self-supervised":
            # MLM Token level Hidden Layer
            self.jrrToken = nn.Sequential(
                nn.LayerNorm(modelDim),
                nn.Linear(modelDim, modelDim),
                nn.GELU(),
                nn.LayerNorm(modelDim),
            )
            self.decoder = nn.Linear(modelDim, vocabSize, bias=False)
            self.decoder.weight = self.nukTuk.weight  # Implementing weight tying
            self.decoder.bias = nn.Parameter(torch.zeros(vocabSize))

        self.classificationHeads = nn.ModuleList([])
        for mtlPredDim in multitaskOutputs:
            taskHead = nn.Sequential(
                nn.LayerNorm(modelDim), nn.Linear(modelDim, mtlPredDim)
            )
            self.classificationHeads.append(taskHead)

    def forward(
        self, x, mask=None, **kwargs
    ):  # **kwargs in case want to adapt for multiple inputs
        """
        Default in forward method: mask = None; allowed for instances of all sequences of the same length, so
        no masking is needed. In all other instances masks should be provided to prevent attention with padded
        tokens.
        """
        x = self.nukTuk(x)

        (
            batchSize,
            n,
            _,
        ) = (
            x.shape
        )  # n = max seq length --> Recall input sequences are padded to max seq length
        contextTokens = repeat(self.context, "1 1 d -> b 1 d", b=batchSize)
        x = torch.cat([contextTokens, x], dim=1)

        # incorporate the positional information and dropout
        x += self.positionalEncoding
        x = self.embeddingDropout(self.preTransformerLayerNorm(x))

        if mask is not None:
            # current mask dim: [b, n-1] with n-1 == maxSeqLen; need to account for context token in the mask and allow for attention with all other tokens in a sequence
            mask = torch.cat(
                [torch.tensor([False]).repeat(x.size(0), 1).to(mask.device), mask],
                dim=1,
            )  # [b, n-1] --> [b,n]

        x = self.transformer(x, mask)

        classificationRepresentation = x.mean(dim=1) if self.aggr == "mean" else x[:, 0]
        classificationRepresentation = self.clsHidden(classificationRepresentation)

        mtlOutputs = list()
        for task in self.classificationHeads:
            mtlOutputs.append(task(classificationRepresentation))

        if self.trainingObjective == "supervised":
            return mtlOutputs

        tokens = x[:, 1:]  # Get all tokens not including CLS
        reconstruction = self.jrrToken(tokens)
        reconstruction = self.decoder(reconstruction)

        return mtlOutputs, reconstruction

    def training_step(self, batch, batch_idx):
        seq, class_targets, _, _, _ = batch
        if self.trainingObjective == "supervised":
            outs = self(seq)
            loss = F.cross_entropy(outs[0].squeeze(1), class_targets, ignore_index=-1)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        seq, class_targets, _, _, _ = batch
        if self.trainingObjective == "supervised":
            outs = self(seq)
            loss = F.cross_entropy(outs[0].squeeze(1), class_targets, ignore_index=-1)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_sched = StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [lr_sched]
