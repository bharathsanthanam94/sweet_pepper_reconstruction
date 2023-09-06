import copy
import MinkowskiEngine as ME
import torch
from transformer_completion.utils.template_mesh import TemplateMesh
from pytorch_lightning import LightningModule
from transformer_completion.models.backbone import MinkEncoderDecoder


class PreBackbone(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams
        self.template_dim = self.cfg[self.cfg.MODEL.DATASET].TEMPLATE_DIM
        self.n_negs = hparams.PRETRAINING.NUM_NEG

        template = TemplateMesh(type='ico', dimensions=self.template_dim)

        self.template_points, self.template_faces = template.get_vertices_faces(
        )

        backbone = MinkEncoderDecoder(hparams.BACKBONE,
                                      template_points=self.template_points)
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(
            backbone)

        self.accumulate_validation_loss = 0
        self.accumulate_validation_count = 0


    def forward(self, x):
        batch_size = len(x['points'])
        template_points = []
        template_faces = []
        for _ in range(batch_size):
            template = TemplateMesh(type='ico', dimensions=self.template_dim)
            pts, faces = template.get_vertices_faces()
            template_points.append(pts)
            template_faces.append(faces)

        self.template_faces = torch.cat(template_faces, 0)
        self.template_points = torch.cat(template_points, 0)

        feats, coors, pad_masks = self.backbone(x)
        return feats, coors, pad_masks

    def training_step(self, x: dict, idx):
        losses = {}

        x_2nd_view = copy.deepcopy(x)
        x_2nd_view['points'] = x['points_2nd_view']
        x_2nd_view['normals'] = x['normals_2nd_view']

        outputs = self.forward(x)
        outputs_2nd_view = self.forward(x_2nd_view)

        losses = self.get_loss(x, outputs, x_2nd_view, outputs_2nd_view, losses, "loss_pt_con")

        total_loss = sum(losses.values())

        self.log("pretrain_loss",
                 total_loss,
                 batch_size=self.cfg.TRAIN.BATCH_SIZE)


        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI
        return total_loss

    def validation_step(self, x: dict, idx):
        ## FOR NOW EQUAL TO TRAINING_STEP
        losses = {}

        outputs = self.forward(x)
        losses = self.get_loss(x, outputs, losses, "loss_pt_con")

        total_loss = sum(losses.values())

        # do running mean
        self.accumulate_validation_loss += total_loss
        self.accumulate_validation_count += len(x['points'])

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def validation_epoch_end(self, outputs):
        val_loss = self.accumulate_validation_loss / self.accumulate_validation_count
        self.log("val_pretrain_loss",
                 val_loss,
                 batch_size=self.cfg.TRAIN.BATCH_SIZE)
        
        self.accumulate_validation_loss = 0
        self.accumulate_validation_count = 0


    @staticmethod
    def ContrastiveLoss(emb_full, emb_par, labels, tau=0.1):
        """
        emb_full: the embeddings of the full pcd
        emb_par: the embeddings of the partial pcd
        labels: the correspondent class labels for each sample in emb_par
        """
        assert emb_par.shape[0] == labels.shape[0], "mismatch on emb_par and labels shapes!"

        emb_full = F.normalize(emb_full, dim=-1)
        emb_par = F.normalize(emb_par, dim=-1)

        # temperature-scaled cosine similarity
        similarity = torch.einsum('ijk,ilk->il', emb_par.unsqueeze(1), emb_full) / tau

        return F.cross_entropy(similarity, labels)


    def get_loss(self, inputs: dict, outputs: tuple, inputs_2nd_view: dict, outputs_2nd_view: tuple, losses: dict,
                 loss_name: str):

        losses[loss_name] = 0
        feats, coors, pad_masks = outputs
        feats = feats[-1]
        coors = coors[-1]
        pad_masks = pad_masks[-1]

        feats_2nd_view, coors_2nd_view, pad_masks_2nd_view = outputs_2nd_view
        feats_2nd_view = feats_2nd_view[-1]
        coors_2nd_view = coors_2nd_view[-1]
        pad_masks_2nd_view = pad_masks_2nd_view[-1]


        for b_index in range(len(inputs['points'])):

            pt_2nd_view =  coors_2nd_view[b_index][~pad_masks_2nd_view[b_index]]
            pt = coors[b_index][~pad_masks[b_index]]
            
            cdist = torch.cdist(pt, pt_2nd_view)

            idx_pos_in_1st_view = torch.argmin(cdist, dim=0)
            _, top_k_idx_neg_in_1st_view = torch.topk(cdist, self.n_negs, dim=0)
            top_k_idx_neg_in_1st_view = top_k_idx_neg_in_1st_view.T

            feat_pos = feats[b_index][idx_pos_in_1st_view]
            feat_neg = feats[b_index][top_k_idx_neg_in_1st_view]
            feats_1st_view = torch.cat((feat_pos.unsqueeze(1), feat_neg), dim=1)

            labels = torch.zeros(len(feats_1st_view), dtype=torch.long).cuda()
            feats_2nd_view_padded = feats_2nd_view[b_index][~pad_masks_2nd_view[b_index]]
            loss_ce = self.ContrastiveLoss(feats_1st_view, feats_2nd_view_padded, labels)
            losses[loss_name] += loss_ce

        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        # optimizer = MTAdam(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.TRAIN.STEP,
            gamma=self.cfg.TRAIN.DECAY)
        return [optimizer], [scheduler]
