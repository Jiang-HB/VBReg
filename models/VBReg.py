import torch, torch.nn.functional as F, torch.nn as nn
from models.VBPointDSC import VBPointDSC, VBNonLocalNet as VBNonLocalNet0, VBNonLocalBlock as VBNonLocalBlock0
from collections import defaultdict

class VBNonLocalBlock(VBNonLocalBlock0):
    def __init__(self, configs):
        super(VBNonLocalBlock, self).__init__(configs)
        self.sigma = configs.sigma.item()

    def message_passing(self, infos):
        infos = super().message_passing(infos)
        if infos["testing"]:
            self.sigma = self.sigma
            corr_feat_belief = infos["corr_feat_belief"]
            normed_corr_feat_belief = F.normalize(corr_feat_belief, p=2, dim=1)  # [B, D, N]
            feat_compatibility = torch.matmul(normed_corr_feat_belief.transpose(2, 1), normed_corr_feat_belief)  # [B, N, N]
            feat_compatibility = torch.clamp(1 - (1 - feat_compatibility) / self.sigma ** 2, min=0, max=1)  # [B, N, N]
            feat_compatibility[:, torch.arange(feat_compatibility.shape[1]), torch.arange(feat_compatibility.shape[1])] = 0.  # [B, N, N]
            spatial_compatibility = infos["spatial_compatibility"]
            compatibility2 = feat_compatibility * spatial_compatibility
            infos["compatibilitys"].append(compatibility2)
        return infos

class VBNonLocalNet(VBNonLocalNet0):
    def __init__(self, configs):
        super(VBNonLocalNet, self).__init__(configs)

    def get_layer_mlp(self, blocks):
        layer = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(self.num_channels),
            nn.ReLU(inplace=True)
        )
        blocks[f'PointCN_layer_0'] = layer
        for i in range(self.num_layers):
            blocks[f'NonLocal_layer_{i}'] = VBNonLocalBlock(self.config)
        return blocks

class VBReg(VBPointDSC):
    def __init__(self, config):
        super(VBReg, self).__init__(config)
        self.k = config.k
        self.ratio = config.ratio
        self.n_points_min = config.n_points_min

    def get_corr_encoder(self, blocks):
        self.config.sigma = self.sigma
        blocks["corr_encoder"] = VBNonLocalNet(configs=self.config)
        return blocks

    def metric(self, infos):
        """
        :param L2_dis: [B, K, N]
        :return:
        """
        seedwise_fitness = (infos["L2_dis"] < self.inlier_threshold).float()  # [B, K, N]
        seedwise_fitness = (seedwise_fitness * (1. - infos["L2_dis"] ** 2 / self.inlier_threshold ** 2)).mean(-1) # - 0.01 * infos["fit_error"]  # [B, K]
        infos["seedwise_fitness"] = seedwise_fitness
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
        seeds = infos["seeds"]  # [B, K]
        spatial_compatibility = infos["spatial_compatibility"]  # [B, N, N]
        B, N , _ = spatial_compatibility.shape
        A = self.k
        K = seeds.shape[1]

        # compatibility matrix
        nonlocal_compatibility_list = infos["nonlocal_weight"]
        compatibility_list = []
        for i in range(len(nonlocal_compatibility_list)):
            compatibility_list.append(nonlocal_compatibility_list[i])
            compatibility_list.append(infos["compatibilitys"][i])
        compatibility_list.reverse()

        compat_map = torch.zeros(B, K, N).to(infos["seeds"].device)  # [B, K, N]
        S_max = None
        for l, compatibility in enumerate(compatibility_list):
            enhanced_compatibility = compatibility
            enhanced_compatibility[:, torch.arange(N), torch.arange(N)] = 0.  # [B, N, N]
            enhanced_compatibility = enhanced_compatibility.gather(dim=1, index=seeds.unsqueeze(-1).expand(-1, -1, N))  # [B, K, N]
            consensus_idxs = torch.topk(enhanced_compatibility, k=A-1, dim=2)[1]  # [B, K, A-1]
            compat_map.scatter_add_(dim=2, index=consensus_idxs, src= torch.ones_like(consensus_idxs).float().to(consensus_idxs.device))  # [B, K, N]

            n = l + 1
            Z = 1.96
            P = compat_map / n
            S = (P + Z ** 2 / (2. * n) - Z * torch.sqrt(P * (1. - P) / n + Z ** 2 / (4 * n ** 2))) / (1. + Z ** 2 / n)
            if l == 0:
                S_max = S
            else:
                S_max = torch.maximum(S, S_max)

        S = S_max
        consensus_idxs = torch.topk(S, k=A-1, dim=2)[1]  # [B, K, A-1]
        consensus_idxs = torch.cat([seeds.unsqueeze(-1), consensus_idxs], 2).reshape(B, K, A)  # [B, K, A]
        infos["consensus_idxs"] = consensus_idxs
        return infos

    def pick_seeds_greedy(self, infos):
        N = infos["src_dist"].shape[1]
        infos["seeds"] = torch.argsort(infos["confidence_prior_forward"], dim=1, descending=True)[:, :infos["n_seeds"]]  # [B, K]
        return infos

    def pick_seeds_nms(self, infos):
        dists, scores, n_seeds = infos["src_dist"], infos["confidence_prior_forward"], infos["n_seeds"]
        assert scores.shape[0] == 1
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        score_relation = score_relation.bool() | (dists[0] >= self.nms_radius).bool()
        is_local_max = score_relation.min(-1)[0].float()
        infos["seeds"] = torch.argsort(scores * is_local_max, dim=1, descending=True)[:, :n_seeds].detach()  # [B, K]
        return infos

    def get_data(self, data):
        infos = defaultdict(list)
        infos["corr_pos"] = data['corr_pos']  # [B, N, 6]
        infos["src_keypts"] = data['src_keypts']  # [B, N, 3]
        infos["tgt_keypts"] = data['tgt_keypts']  # [B, N, 3]
        infos["hard_gt_labels"] = data['gt_labels'][:, :, 0]  # [B, N]
        infos["soft_gt_labels"] = data['gt_labels'][:, :, 1]  # [B, N]
        infos["testing"] = data['testing']
        infos["vb_eval"] = data["vb_eval"] if "vb_eval" in data else False
        infos["n_seeds"] = max(self.n_points_min, int(self.ratio * data['corr_pos'].shape[1]))
        return infos

    def forward(self, data):

        ## prepare input data
        infos = self.get_data(data)
        infos = self.get_spatial_compatibility(infos)

        ## extract corr feature
        infos = self.extract_corr_feats(infos)

        ## inlier confidence estimation
        infos = self.pred_confidence(infos)
        infos["confidence"] = infos["confidence_prior_forward"]

        ## consensus sampling
        infos = self.consensus_sampling(infos)

        ## calculate seedwise trans
        infos = self.cal_seed_trans(infos)

        return infos
