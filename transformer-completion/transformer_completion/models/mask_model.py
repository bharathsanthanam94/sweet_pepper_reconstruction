import matplotlib.pyplot as plt
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from metrics_3d.chamfer_distance import ChamferDistance
from metrics_3d.precision_recall import PrecisionRecall
from pytorch3d.loss import chamfer_distance as ChamferDistanceLoss
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes
from pytorch_lightning import LightningModule
from transformer_completion.models.backbone import MinkEncoderDecoder
from transformer_completion.models.decoder import MaskedTransformerDecoder
from transformer_completion.optim import MTAdam
from transformer_completion.utils.template_mesh import TemplateMesh


class MaskPS(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams
        self.template_dim = self.cfg[self.cfg.MODEL.DATASET].TEMPLATE_DIM

        template = TemplateMesh(type="ico", dimensions=self.template_dim)

        self.template_points, self.template_faces = template.get_vertices_faces()

        hparams.DECODER.NUM_QUERIES = self.template_points.shape[1]

        self.cosine_similarity = torch.nn.CosineSimilarity()
        self.bce = torch.nn.BCELoss()

        backbone = MinkEncoderDecoder(
            hparams.BACKBONE, template_points=self.template_points
        )
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)

        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE,
            hparams[hparams.MODEL.DATASET],
        )

        self.freezeModules()
        self.chamfer_dist = ChamferDistanceLoss

        self.chamfer_dist_metric = ChamferDistance()
        self.precision_recall = PrecisionRecall(min_t=0.001, max_t=0.01, num=100)

        self.inference_time = 0
        self.count = 0

        self.offset_scaling = self.cfg.DECODER.OFFSET_SCALING

        self.warning_no_gt = False

        self.fruit = self.cfg.MODEL.FRUIT
        
        if self.fruit == "SWEETPEPPER":
            self.smooth = self.cfg.MODEL.SMOOTH_SW
        elif self.fruit == "STRABERRY":
            self.smooth = self.cfg.MODEL.SMOOTH_ST
        elif self.fruit == "LEAF":
            self.smooth = self.cfg.MODEL.SMOOTH_LEAF

    def freezeModules(self):
        freeze_dict = {"BACKBONE": self.backbone, "DECODER": self.decoder}
        print("Frozen modules: ", self.cfg.TRAIN.FREEZE_MODULES)
        for module in self.cfg.TRAIN.FREEZE_MODULES:
            for param in freeze_dict[module].parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size = len(x["points"])
        template_points = []
        template_faces = []
        for _ in range(batch_size):
            template = TemplateMesh(type="ico", dimensions=self.template_dim)
            pts, faces = template.get_vertices_faces()
            template_points.append(pts)
            template_faces.append(faces)

        self.template_faces = torch.cat(template_faces, 0)
        self.template_points = torch.cat(template_points, 0)
        feats, coors, pad_masks = self.backbone(x)
    

        #TODO: Bharath: Extract features from ResNetFPN and pass them to decoder
        

        #Here as an addition pass self.template_faces, and all of x
        
        outputs=self.decoder(feats,coors,pad_masks,self.template_points,self.template_faces,x)
        
        #Below code is Fed's
        # outputs = self.decoder(feats, coors, pad_masks, self.template_points)
        # ipdb.set_trace()
        return outputs

    def training_step(self, x: dict, idx):
        losses = {}

        outputs = self.forward(x)
        losses = self.get_loss(x, outputs, losses, "loss_cd")

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                losses = self.get_loss(x, aux_outputs, losses, "loss_cd_" + str(i))

        total_loss = sum(losses.values())

        loss_cd = 0
        loss_confidence = 0
        loss_reg_laplacian = 0
        loss_reg_normals = 0
        for key in losses.keys():
            if "cd" in key:
                loss_cd += losses[key]
            if "conf" in key:
                loss_confidence += losses[key]
            if "laplacian" in key:
                loss_reg_laplacian += losses[key]
            if "normals" in key:
                loss_reg_normals += losses[key]

        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("train_loss_cd", loss_cd, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log(
            "train_loss_conf", loss_confidence, batch_size=self.cfg.TRAIN.BATCH_SIZE
        )
        self.log(
            "train_loss_reg_normals",
            loss_reg_normals,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "train_loss_reg_laplacian",
            loss_reg_laplacian,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI
        return total_loss

    def visualize(self, inputs: dict, outputs: dict):
        # TODO redo this whole shit. still working only with batch size = 0
        return

    def validation_step(self, x: dict, idx):
        outputs = self.forward(x)

        previous_template = outputs["previous_template_points"]
        offsets_scaled = torch.sigmoid(outputs["offsets"]) * self.offset_scaling
        deformed_template = previous_template * offsets_scaled

        confidence_output = torch.sigmoid(outputs["confidence"])
        confidence_output = confidence_output > 0.9

        for b_idx in range(len(confidence_output)):
            pt = deformed_template[b_idx]
            gt = x["extra"]["gt_points"][b_idx]

            pt_mesh = o3d.geometry.TriangleMesh()
            pt_mesh.vertices = o3d.utility.Vector3dVector(pt.cpu())
            pt_mesh.triangles = o3d.utility.Vector3iVector(self.template_faces[0].cpu())
            pt_mesh.remove_vertices_by_mask(~confidence_output[b_idx].cpu().numpy())

            self.chamfer_dist_metric.update(gt, pt_mesh)
            self.precision_recall.update(gt, pt_mesh)

        if self.global_step % self.cfg.TRAIN.MAX_EPOCH == 0:
            self.visualize(x, outputs)

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def test_step(self, x: dict, idx):
        import time

        start = time.time()
        outputs = self.forward(x)

        meshes = self.get_meshes(
            outputs["previous_template_points"],
            outputs["offsets"],
            outputs["confidence"]
        )
        
        if "VIZ_INT" in self.cfg:
            all_meshes = []
            for aux_i, aux_out in enumerate(outputs["aux_outputs"]):
                meshes = self.get_meshes(
                    aux_out["previous_template_points"],
                    aux_out["offsets"],
                    aux_out["confidence"],
                )
                all_meshes.append(meshes)

        self.inference_time += time.time() - start
        self.count += len(outputs["confidence"])

        for batch_idx in range(len(outputs["confidence"])):
            if "extra" not in x.keys():
                gt = x["points"][batch_idx]
                self.warning_no_gt = True
            else:
                gt = x["extra"]["gt_points"][batch_idx]

            pt_mesh = meshes[batch_idx]

            if "VIZ_INT" in self.cfg:
                for int_i, int_mesh in enumerate(all_meshes):
                    # import ipdb;ipdb.set_trace()
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(gt)
                    final_prediction_lineset = (
                        o3d.geometry.LineSet.create_from_triangle_mesh(
                            int_mesh[batch_idx]
                        )
                    )
                    file_mesh=x['filename'][0].split('color/')[1].split('.png')[0]+"_int_mesh_level"+str(int_i)+".ply" #line added
                    o3d.io.write_triangle_mesh(file_mesh, int_mesh[batch_idx]) #line added
                    in_pcd = o3d.geometry.PointCloud()
                    in_pcd.points = o3d.utility.Vector3dVector(x["points"][batch_idx])
                    print("Mesh level", int_i)
                    o3d.visualization.draw_geometries(
                        [final_prediction_lineset, in_pcd],
                        mesh_show_back_face=True,
                        mesh_show_wireframe=True,
                    )

            if "VIZ" in self.cfg:
                gt_pcd = o3d.geometry.PointCloud()
                gt_pcd.points = o3d.utility.Vector3dVector(gt)
                final_prediction_lineset = (
                    o3d.geometry.LineSet.create_from_triangle_mesh(pt_mesh)
                )
                # o3d.io.write_triangle_mesh("predicted_mesh.ply",pt_mesh)
                file_mesh=x['filename'][0].split('color/')[1].split('.png')[0]+"_mesh"+".ply"
                o3d.io.write_triangle_mesh(file_mesh, pt_mesh) #uncomment this
                in_pcd = o3d.geometry.PointCloud()
                in_pcd.points = o3d.utility.Vector3dVector(x["points"][batch_idx])
                # post-process crap
                # pp_mesh = o3d.t.geometry.TriangleMesh.from_legacy(pp_mesh).fill_holes(
                #     hole_size=0.004).to_legacy()
                print("final prediction")
                #uncomment this 
                '''
                o3d.visualization.draw_geometries(
                    [final_prediction_lineset, in_pcd],
                    mesh_show_back_face=True,
                    mesh_show_wireframe=True,
                )
                '''

            self.chamfer_dist_metric.update(gt, pt_mesh)
            self.precision_recall.update(gt, pt_mesh)
            

            # fruit_id = x['filename'][batch_idx].split('/')[-3]
            # if self.warning_no_gt:
            #     fruit_id = x['filename'][batch_idx].split('/')[-1]
            # o3d.io.write_triangle_mesh('./transformer_completion/results/{}.ply'.format(fruit_id), pt_mesh)
        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def validation_epoch_end(self, outputs):
        cd = self.chamfer_dist_metric.compute()
        p, r, f = self.precision_recall.compute_auc()

        self.log("val_chamfer_distance", cd, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_precision_auc", p, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_recall_auc", r, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_fscore_auc", f, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        self.reset_metrics()
        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def test_epoch_end(self, outputs):
        cd = self.chamfer_dist_metric.compute()
        p, r, f = self.precision_recall.compute_auc()
        t = self.inference_time / self.count

        print("------------------------------------")
        print("chamfer distance: {}".format(cd))
        print("computing area under curve")
        print("precision: {}".format(p))
        print("recall: {}".format(r))
        print("fscore: {}".format(f))
        print("------------------------------------")
        p, r, f, _ = self.precision_recall.compute_at_threshold(0.005)
        print("computing at threshold 0.005")
        print("precision: {}".format(p))
        print("recall: {}".format(r))
        print("fscore: {}".format(f))
        print("------------------------------------")
        print("time: {}".format(t))
        self.reset_metrics()
        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

        if self.warning_no_gt:
            print("------------------------------------")
            print("WARNING: no ground truth found, metrics are bullshit")

    def reset_metrics(self):
        self.chamfer_dist_metric.reset()
        self.precision_recall.reset()
        self.inference_time = 0
        self.count = 0

    def get_loss(self, inputs: dict, outputs: dict, losses: dict, loss_name: str):
        # TODO add to config
        loss_cd_w = 1000 * 1000
        loss_reg_norm_w = 0.1
        loss_reg_laplace_w = 0.1

        previous_template = outputs["previous_template_points"]
        offsets_scaled = torch.sigmoid(outputs["offsets"]) * self.offset_scaling
        deformed_template = previous_template * offsets_scaled

        # cosine similarity
        similarity_target = torch.empty(
            (len(inputs["extra"]["gt_points"]), len(previous_template[0]))
        )
        similarity_regression_target = torch.empty(
            (len(inputs["extra"]["gt_points"]), len(previous_template[0]))
        )

        with torch.no_grad():
            normalized_template_pts = torch.nn.functional.normalize(
                previous_template, dim=-1
            )

            for idx, pts in enumerate(inputs["extra"]["gt_points"]):
                pts = torch.from_numpy(pts).float().cuda()
                normalized_pts = torch.nn.functional.normalize(pts)
                cosine_similarity = torch.mm(
                    normalized_pts, normalized_template_pts[idx].T
                )

                try:
                    max_cosine_similarity, _ = torch.max(cosine_similarity, dim=0)
                except IndexError:
                    import ipdb; ipdb.set_trace()
                binary_mask = max_cosine_similarity > 0.99
                similarity_target[idx] = binary_mask
                similarity_regression_target[idx] = max_cosine_similarity

        confidence_output = torch.sigmoid(outputs["confidence"].squeeze(-1))

        # confidence_loss = 10 * torch.nn.L1Loss()(
        #     similarity_regression_target.cuda(), confidence_output)
        if 'WEIGHTS_CONF' in self.cfg.LOSS:
            conf_w = self.cfg.LOSS.WEIGHTS_CONF
        else:
            conf_w = 5
        confidence_loss = conf_w * self.bce(confidence_output, similarity_target.cuda())
        losses[loss_name.replace("cd", "conf")] = confidence_loss
        # ----

        # regularizers
        deformed_template_mesh = Meshes(
            verts=deformed_template, faces=self.template_faces
        )
        loss_normal = mesh_normal_consistency(deformed_template_mesh)
        loss_laplacian = mesh_laplacian_smoothing(
            deformed_template_mesh, method="uniform"
        )
        losses[loss_name.replace("cd", "reg_normals")] = loss_normal * loss_reg_norm_w
        losses[loss_name.replace("cd", "reg_laplacian")] = (
            loss_laplacian * loss_reg_laplace_w
        )
        # ----

        # chamfer
        loss_cd = 0
        for def_tmp, gt_pts, target in zip(
            deformed_template, inputs["extra"]["gt_points"], similarity_target
        ):
            gt = torch.from_numpy(gt_pts).float().cuda().unsqueeze(dim=0)
            def_tmp = def_tmp[target.bool()]
            loss_pts, _ = self.chamfer_dist(def_tmp.unsqueeze(dim=0), gt)
            loss_cd += loss_pts

        losses[loss_name] = loss_cd * loss_cd_w
        # ----
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        # optimizer = MTAdam(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]

    def get_meshes(self, previous_template, offsets, confidence):
        offsets_scaled = torch.sigmoid(offsets) * self.offset_scaling
        deformed_template = previous_template * offsets_scaled
        confidence_output = torch.sigmoid(confidence)
        # confidence_output = confidence_output > 0.9
        meshes = []
        for batch_idx in range(len(confidence_output)):
            pt = deformed_template[batch_idx]
            pt_mesh = o3d.geometry.TriangleMesh()
            pt_mesh.vertices = o3d.utility.Vector3dVector(pt.cpu())
            pt_mesh.triangles = o3d.utility.Vector3iVector(self.template_faces[0].cpu())
            # pt_mesh.remove_vertices_by_mask(~confidence_output[batch_idx].cpu().numpy())
            pt_mesh = pt_mesh.filter_smooth_taubin(self.smooth)
            meshes.append(pt_mesh)
        return meshes
