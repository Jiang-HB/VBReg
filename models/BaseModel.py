import torch, torch.nn as nn, torch.nn.functional as F
from models.common import knn, rigid_transform_3d
from utils.SE3 import transform
from collections import defaultdict

class BaseNonLocalBlock(nn.Module):
    def __init__(self, config):
        super(BaseNonLocalBlock, self).__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.num_heads = config.num_heads
        self.lb_dim = config.lb_dim
        self.blocks = torch.nn.ModuleDict({})
        self.blocks = self.get_message_mlp(self.blocks)
        self.blocks = self.get_projection_mlp(self.blocks)

    def get_message_mlp(self, blocks):
        blocks["message_mlp"] = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.num_channels // 2, self.num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.num_channels // 2, self.num_channels, kernel_size=1),
        )
        return blocks

    def get_projection_mlp(self, blocks):
        blocks["projection_q"] = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        blocks["projection_k"] = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        blocks["projection_v"] = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        return blocks

    def corr_projection(self, infos):
        corr_feat_belief = infos["corr_feat_belief"]  # [B, D, N]
        B, D, N = corr_feat_belief.shape
        infos["Q"] = self.blocks["projection_q"](corr_feat_belief).view([B, self.num_heads, self.num_channels // self.num_heads, N])  # [B, H, D // H, N]
        infos["K"] = self.blocks["projection_k"](corr_feat_belief).view([B, self.num_heads, self.num_channels // self.num_heads, N])  # [B, H, D // H, N]
        infos["V"] = self.blocks["projection_v"](corr_feat_belief).view([B, self.num_heads, self.num_channels // self.num_heads, N])  # [B, H, D // H, N]
        return infos

    def message_passing(self, infos):
        feat_attention = torch.einsum('bhco, bhci->bhoi', infos["Q"], infos["K"]) / (self.num_channels // self.num_heads) ** 0.5  # [B, H, N, N]
        B, _, N, _ = feat_attention.shape
        torch.cuda.empty_cache()
        weight = torch.softmax(infos["spatial_compatibility"][:, None, :, :] * feat_attention, dim=-1)  # [B, H, N, N]
        torch.cuda.empty_cache()
        message = torch.einsum('bhoi, bhci-> bhco', weight, infos["V"]).view([B, -1, N])  # [B, D, N]
        message = self.blocks["message_mlp"](message)
        infos["corr_feat_belief"] = infos["corr_feat_belief"] + message
        if infos["testing"]:
            infos["nonlocal_weight"].append(weight.squeeze(1))
        return infos

    def forward(self, infos):
        infos = self.corr_projection(infos)
        infos = self.message_passing(infos)
        return infos

class NonLocalBlock(BaseNonLocalBlock):
    def __init__(self, configs):
        super(NonLocalBlock, self).__init__(configs)

class BaseNonLocalNet(nn.Module):
    def __init__(self, config):
        super(BaseNonLocalNet, self).__init__()
        self.config = config
        self.in_dim = config.in_dim
        self.num_layers = config.num_layers
        self.num_channels = config.num_channels

        self.blocks = torch.nn.ModuleDict({})
        self.blocks = self.get_init_mlp(self.blocks)
        self.blocks = self.get_layer_mlp(self.blocks)

    def get_init_mlp(self, blocks):
        blocks["init_mlp"] = nn.Conv1d(self.in_dim, self.num_channels, kernel_size=1, bias=True)
        return blocks

    def get_layer_mlp(self, blocks):
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1, bias=True),
                nn.BatchNorm1d(self.num_channels),
                nn.ReLU(inplace=True)
            )
            blocks[f'PointCN_layer_{i}'] = layer
            blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(self.config)
        return blocks

    def init_feat_embedding(self, infos):
        infos["corr_feat_belief"] = self.blocks["init_mlp"](infos["corr_pos"].transpose(2, 1)) # [B, D, N]
        return infos

    def roll(self, infos):
        for i in range(self.num_layers):
            infos["corr_feat_belief"] = self.blocks[f'PointCN_layer_{i}'](infos["corr_feat_belief"])
            infos = self.blocks[f'NonLocal_layer_{i}'](infos)
        return infos

    def forward(self, infos):
        infos = self.init_feat_embedding(infos)
        infos = self.roll(infos)
        infos["corr_feat"] = infos["corr_feat_belief"]  # [B, D, N]
        infos["normed_corr_feat"] = F.normalize(infos["corr_feat"], p=2, dim=1)  # [B, D, N]
        return infos

class NonLocalNet(BaseNonLocalNet):
    def __init__(self, configs):
        super(NonLocalNet, self).__init__(configs)

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.in_dim = config.in_dim # maximum iteration of power iteration algorithm
        self.num_layers = config.num_layers
        self.num_iterations = config.num_iterations  # maximum iteration of power iteration algorithm
        self.ratio = config.ratio  # the maximum ratio of seeds.
        self.num_channels = config.num_channels
        self.inlier_threshold = config.inlier_threshold
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_spat = nn.Parameter(torch.Tensor([config.sigma_d]).float(), requires_grad=False)
        self.k = config.k  # neighborhood number in NSM module.
        self.nms_radius = config.nms_radius  # only used during testing

        self.blocks = torch.nn.ModuleDict({})
        self.blocks = self.get_corr_encoder(self.blocks)
        self.blocks = self.get_corr_classifier(self.blocks)
        self.maps = []

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_corr_encoder(self, blocks):
        blocks["corr_encoder"] = NonLocalNet(configs=self.config)
        return blocks

    def get_corr_classifier(self, blocks):
        blocks["corr_classifier"] = nn.Sequential(
            nn.Conv1d(self.num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
        )
        return blocks

    def get_data(self, data):
        infos = defaultdict(list)
        infos["corr_pos"] = data['corr_pos']  # [B, N, 6]
        infos["src_keypts"] = data['src_keypts']  # [B, N, 3]
        infos["tgt_keypts"] = data['tgt_keypts']  # [B, N, 3]
        infos["hard_gt_labels"] = data['gt_labels'][:, :, 0]  # [B, N]
        infos["soft_gt_labels"] = data['gt_labels'][:, :, 1]  # [B, N]
        infos["testing"] = data['testing']
        infos["vb_eval"] = data["vb_eval"] if "vb_eval" in data else False
        return infos

    def extract_corr_feats(self, infos):
        return self.blocks["corr_encoder"](infos)

    def get_spatial_compatibility(self, infos):
        infos["src_dist"] = torch.norm((infos["src_keypts"][:, :, None, :] - infos["src_keypts"][:, None, :, :]), dim=-1)
        infos["tgt_dist"] = torch.norm((infos["tgt_keypts"][:, :, None, :] - infos["tgt_keypts"][:, None, :, :]), dim=-1)
        spatial_compatibility = infos["src_dist"] - infos["tgt_dist"]
        infos["spatial_compatibility"] = torch.clamp(1.0 - spatial_compatibility ** 2 / self.sigma_spat ** 2, min=0)
        return infos

    def get_feat_compatibility(self, infos):
        normed_corr_feat = infos["normed_corr_feat"]  # [B, D, N]
        if not infos["testing"]:
            feat_compatibility = torch.matmul(normed_corr_feat.transpose(2, 1), normed_corr_feat)  # [B, N, N]
            feat_compatibility = torch.clamp(1 - (1 - feat_compatibility) / self.sigma ** 2, min=0, max=1)  # [B, N, N]
            feat_compatibility[:, torch.arange(feat_compatibility.shape[1]), torch.arange(feat_compatibility.shape[1])] = 0.  # [B, N, N]
        else:
            feat_compatibility = None
        infos["feat_compatibility"] = feat_compatibility
        return infos

    def pred_confidence(self, infos):
        infos["confidence"] = self.blocks["corr_classifier"](infos["corr_feat"]).squeeze(1)  # [B, N]
        return infos

    def pick_seeds_nms(self, infos):
        N = infos["src_dist"].shape[1]
        dists, scores, n_seeds = infos["src_dist"], infos["confidence"], int(N * self.ratio)
        assert scores.shape[0] == 1
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        score_relation = score_relation.bool() | (dists[0] >= self.nms_radius).bool()
        is_local_max = score_relation.min(-1)[0].float()
        infos["seeds"] = torch.argsort(scores * is_local_max, dim=1, descending=True)[:, :n_seeds].detach()  # [B, K]
        return infos

    def pick_seeds_greedy(self, infos):
        N = infos["src_dist"].shape[1]
        infos["seeds"] = torch.argsort(infos["confidence"], dim=1, descending=True)[:, :int(N * self.ratio)]  # [B, K]
        return infos

    # def pick_seeds(self, infos):
    #     if infos["testing"]:
    #         infos = self.pick_seeds_nms(infos)
    #     else:
    #         infos = self.pick_seeds_greedy(infos)
    #     return infos

    def pick_seeds(self, infos):
        if infos["testing"]:
            if infos["vb_eval"]:
                infos = self.pick_seeds_greedy(infos)
            else:
                infos = self.pick_seeds_nms(infos)
        else:
            infos = self.pick_seeds_greedy(infos)
        return infos

    def consensus_sampling(self, infos):
        """
        :param normed_corr_feat: [B, D, N]
        :param seeds: [B, K]
        :return:
        """
        # seed sampling
        infos = self.pick_seeds(infos)

        # consensus_sampling
        normed_corr_feat, seeds = infos["normed_corr_feat"], infos["seeds"]
        B, D, N = normed_corr_feat.shape
        A = min(self.k, N - 1)
        consensus_idxs = knn(normed_corr_feat.transpose(2, 1), k=A, ignore_self=True, normalized=True)  # [B, N, A]
        infos["consensus_idxs"] = consensus_idxs.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, A))  # [B, K, A]
        return infos

    def get_consensus_spatial_compatibility(self, infos):
        """
        :param src_keypts: [B, N, 3]
        :param tgt_keypts: [B, N, 3]
        :return:
        """
        src_keypts, tgt_keypts = infos["src_keypts"], infos["tgt_keypts"]
        consensus_idxs = infos["consensus_idxs"]  # [B, K, A]
        B, K, A = consensus_idxs.shape
        seed_src = src_keypts.gather(dim=1, index=consensus_idxs.view([B, -1])[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        seed_tgt = tgt_keypts.gather(dim=1, index=consensus_idxs.view([B, -1])[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        seed_spatial_compatibility = ((seed_src[:, :, :, None, :] - seed_src[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - \
                                     ((seed_tgt[:, :, :, None, :] - seed_tgt[:, :, None, :, :]) ** 2).sum(-1) ** 0.5  # [B, K, A, A]
        infos["consensus_spatial_compatibility"] = torch.clamp(1 - seed_spatial_compatibility ** 2 / self.sigma_spat ** 2, min=0)  # [B, K, A, A]
        return infos

    def get_consensus_feat_compatibility(self, infos):
        """
        :param normed_corr_feat: [B, D, N]
        :return:
        """
        normed_corr_feat = infos["normed_corr_feat"].transpose(2, 1)  # [B, N, D]
        B, N, D = normed_corr_feat.shape
        consensus_idxs = infos["consensus_idxs"]  # [B, K, A]
        _, K, A = consensus_idxs.shape
        consensus_feats = normed_corr_feat.gather(dim=1, index=consensus_idxs.contiguous().view(B, -1).unsqueeze(-1).expand(-1, -1, D)).contiguous().view(B, K, A, D)  # [B, K, A, D]
        consensus_feat_compatibility = torch.matmul(consensus_feats, consensus_feats.permute(0, 1, 3, 2))  # [B, K, A, A]
        consensus_feat_compatibility = torch.clamp(1 - (1 - consensus_feat_compatibility) / self.sigma ** 2, min=0)  # [B, K, A, A]
        infos["consensus_feat_compatibility"] = consensus_feat_compatibility  # [B, K, A, A]
        return infos

    def get_consensus_compatibility(self, infos):
        infos = self.get_consensus_feat_compatibility(infos)  # [B, K, A, A]
        infos = self.get_consensus_spatial_compatibility(infos)  # [B, K, A, A]
        infos["consensus_compatibility"]= infos["consensus_spatial_compatibility"] * infos["consensus_feat_compatibility"]  # [B, K, A, A]
        return infos

    def cal_leading_eigenvector(self, M, method='power'):
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def get_consensus_weight(self, infos):
        infos = self.get_consensus_compatibility(infos)
        consensus_compatibility = infos["consensus_compatibility"]  # [B, K, A, A]
        B, K, A, _ = consensus_compatibility.shape
        consensus_compatibility[:, :, torch.arange(A), torch.arange(A)] = 0.  # [B, K, A, A]
        consensus_weight = self.cal_leading_eigenvector(consensus_compatibility.reshape(-1, A, A), method='power')  # [B * K, A]
        consensus_weight = consensus_weight.view([B, K, A])  # [B * K, A]
        consensus_weight = consensus_weight / (torch.sum(consensus_weight, dim=-1, keepdim=True) + 1e-6)  # [B * K, A]
        infos["consensus_weight"] = consensus_weight.reshape(B, K, A)
        return infos

    # def metric(self, infos):
    #     infos["seedwise_fitness"] = torch.mean((infos["L2_dis"] < self.inlier_threshold).float(), dim=-1)  # [B, K]
    #     return infos

    def metric(self, infos):
        """
        :param L2_dis: [B, K, N]
        :return:
        """
        seedwise_fitness = (infos["L2_dis"] < self.inlier_threshold).float()  # [B, K, N]
        seedwise_fitness = (seedwise_fitness * (1. - infos["L2_dis"] ** 2 / self.inlier_threshold ** 2)).mean(-1) # - 0.01 * infos["fit_error"]  # [B, K]
        infos["seedwise_fitness"] = seedwise_fitness
        return infos

    def eval_trans(self, infos):
        seedwise_trans = infos["seedwise_trans"]  # [B, K, 4, 4]
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3], infos["src_keypts"].permute(0, 2, 1)) + seedwise_trans[:, :, :3, 3:4]  # [bs, num_seeds, num_corr, 3]
        pred_position = pred_position.permute(0, 1, 3, 2)
        infos["L2_dis"] = torch.norm(pred_position - infos["tgt_keypts"][:, None, :, :], dim=-1)  # [B, K, N]
        infos = self.metric(infos)
        infos["batch_best_guess"] = infos["seedwise_fitness"].argmax(dim=1)  # [B]
        return infos

    def choose_best_trans(self, infos):
        """
        :param seedwise_trans: [B, K, 4, 4]
        :return:
        """
        infos["final_trans"] = infos["seedwise_trans"].gather(dim=1, index=infos["batch_best_guess"][:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)  # [B, 4, 4]
        final_labels = infos["L2_dis"].gather(dim=1, index=infos["batch_best_guess"][:, None, None].expand(-1, -1, infos["L2_dis"].shape[2])).squeeze(1)  # [B, N]
        infos["final_labels"] = (final_labels < self.inlier_threshold).float()  # [B, N]
        return infos

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 0.10: # for 3DMatch
            inlier_threshold_list = [0.10] * 20
        else: # for KITTI
            inlier_threshold_list = [1.2] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1/(1 + (L2_dis/inlier_threshold)**2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans

    def cal_seed_trans(self, infos):
        """
        :param src_keypts: [B, N, 3]
        :param tgt_keypts: [B, N, 3]
        :return:
        """
        consensus_idxs = infos["consensus_idxs"]  # [B, K, A]
        infos = self.get_consensus_weight(infos)
        consensus_weight = infos["consensus_weight"]  # [B, K, A]
        B, K, A = consensus_idxs.shape

        ## seedwise transformation estimation
        src_consensus = infos["src_keypts"].gather(dim=1, index=consensus_idxs.reshape(B, -1)[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        tgt_consensus = infos["tgt_keypts"].gather(dim=1, index=consensus_idxs.reshape(B, -1)[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        infos["src_consensus"] = src_consensus.view([B, K, A, 3])
        infos["tgt_consensus"] = tgt_consensus.view([B, K, A, 3])
        src_consensus, tgt_consensus = src_consensus.view([B * K, A, 3]), tgt_consensus.view([B * K, A, 3])  # [B * K, A, 3]
        infos["seedwise_trans"] = rigid_transform_3d(src_consensus, tgt_consensus, consensus_weight.reshape(B * K, A)).reshape(B, K, 4, 4)
        infos = self.eval_trans(infos)
        infos = self.choose_best_trans(infos)

        if infos["testing"]:
            infos["final_trans"] = self.post_refinement(infos["final_trans"], infos["src_keypts"], infos["tgt_keypts"])

        if not infos["testing"]:
            infos["final_labels"] = infos["confidence"]

        return infos

    def plot_feat_compat_maps(self, infos):
        normed_corr_feat = infos["normed_corr_feat"]  # [B, D, N]
        feat_compatibility = torch.matmul(normed_corr_feat.transpose(2, 1), normed_corr_feat)  # [B, N, N]
        feat_compatibility = torch.clamp(1 - (1 - feat_compatibility) / self.sigma ** 2, min=0, max=1)  # [B, N, N]
        attention_map = feat_compatibility[0].cpu() * infos["spatial_compatibility"][0].cpu()
        inlier_idxs = torch.where(infos["hard_gt_labels"] == 1.0)[1].cpu()
        outlier_idxs = torch.where(infos["hard_gt_labels"] == 0.0)[1].cpu()
        idxs = torch.cat([inlier_idxs, outlier_idxs], dim=0)
        attention_map = attention_map.gather(dim=0, index=idxs[:, None].expand(-1, attention_map.shape[0]))
        attention_map = attention_map.gather(dim=1, index=idxs[None].expand(attention_map.shape[0], -1))

        self.maps.append(attention_map.cpu().numpy())

        # if len(self.maps) > 10:
        #     plot_attention_map(self.maps, img_path=self.config.attention_map_nm)
        #     exit(-1)

    def forward(self, data):

        ## prepare input data
        infos = self.get_data(data)

        ## extract corr feature
        infos = self.get_spatial_compatibility(infos)
        infos = self.extract_corr_feats(infos)
        infos = self.get_feat_compatibility(infos)

        ## inlier confidence estimation
        infos = self.pred_confidence(infos)

        ## consensus sampling
        infos = self.consensus_sampling(infos)

        ## calculate seedwise trans
        infos = self.cal_seed_trans(infos)

        return infos
