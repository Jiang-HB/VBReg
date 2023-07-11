from models.BaseModel import BaseModel, BaseNonLocalNet, BaseNonLocalBlock
import torch, torch.nn as nn, torch.nn.functional as F
from models.common import knn, rigid_transform_3d

class VBNonLocalBlock(BaseNonLocalBlock):
    def __init__(self, configs):
        super(VBNonLocalBlock, self).__init__(configs)

    def get_projection_mlp(self, blocks):

        blocks["projection_q"] = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        )

        blocks["projection_k"] = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        )

        blocks["projection_v"] = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1)
        )

        return blocks

    def corr_projection(self, infos):
        B, D, N = infos["Q_emb"].shape
        infos["Q"] = self.blocks["projection_q"](infos["Q_emb"]).view([B, self.num_heads, self.num_channels // self.num_heads, N])
        infos["K"] = self.blocks["projection_k"](infos["K_emb"]).view([B, self.num_heads, self.num_channels // self.num_heads, N])
        infos["V"] = self.blocks["projection_v"](infos["V_emb"]).view([B, self.num_heads, self.num_channels // self.num_heads, N])
        return infos

class VBNonLocalNet(BaseNonLocalNet):
    def __init__(self, configs):
        super(VBNonLocalNet, self).__init__(configs)
        self.belief_size = configs.belief_size
        self.hidden_size = configs.hidden_size
        self.state_size = configs.state_size
        self.lb_size = configs.lb_dim
        self.blocks = self.get_rnn(self.blocks)

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

    def get_init_mlp(self, blocks):
        blocks["init_mlp"] = nn.Sequential(nn.Conv1d(self.in_dim, self.num_channels, kernel_size=1, bias=True),
                                           nn.BatchNorm1d(self.num_channels),
                                           nn.ReLU(inplace=True),
                                           nn.Conv1d(self.num_channels, self.num_channels, kernel_size=1, bias=True))
        return blocks

    def get_rnn(self, blocks):

        blocks["rnn_Q"] = nn.GRUCell(self.belief_size, self.belief_size)
        blocks["rnn_K"] = nn.GRUCell(self.belief_size, self.belief_size)
        blocks["rnn_V"] = nn.GRUCell(self.belief_size, self.belief_size)

        blocks["fc_embed_state_action_Q"] = nn.Conv1d(self.state_size + self.num_channels, self.belief_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_prior_Q"] = nn.Conv1d(self.belief_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_prior_Q"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_posterior_Q"] = nn.Conv1d(self.belief_size + self.lb_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_posterior_Q"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_sto_Q"] = nn.Conv1d(self.belief_size + self.state_size, self.num_channels, kernel_size=1, bias=True)
        blocks["fc_label_Q"] = nn.Conv1d(2, self.lb_size, kernel_size=1, bias=True)

        blocks["fc_embed_state_action_K"] = nn.Conv1d(self.state_size + self.num_channels, self.belief_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_prior_K"] = nn.Conv1d(self.belief_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_prior_K"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_posterior_K"] = nn.Conv1d(self.belief_size + self.lb_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_posterior_K"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_sto_K"] = nn.Conv1d(self.belief_size + self.state_size, self.num_channels, kernel_size=1, bias=True)
        blocks["fc_label_K"] = nn.Conv1d(2, self.lb_size, kernel_size=1, bias=True)

        blocks["fc_embed_state_action_V"] = nn.Conv1d(self.state_size + self.num_channels, self.belief_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_prior_V"] = nn.Conv1d(self.belief_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_prior_V"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_posterior_V"] = nn.Conv1d(self.belief_size + self.lb_size, self.hidden_size, kernel_size=1, bias=True)
        blocks["fc_state_posterior_V"] = nn.Conv1d(self.hidden_size, 2 * self.state_size, kernel_size=1, bias=True)
        blocks["fc_embed_belief_sto_V"] = nn.Conv1d(self.belief_size + self.state_size, self.num_channels, kernel_size=1, bias=True)
        blocks["fc_label_V"] = nn.Conv1d(2, self.lb_size, kernel_size=1, bias=True)

        blocks["act_fn"] = torch.nn.ReLU()

        return blocks

    def translation(self, infos, corr_feat, state, belief, lbs, rnn, fc_embed_state_action, fc_embed_belief_prior, fc_state_prior,
                    fc_embed_belief_posterior, fc_state_posterior, fc_label):
        hidden = self.blocks["act_fn"](fc_embed_state_action(torch.cat([state, corr_feat], dim=1)))
        B, D1, N = hidden.shape
        B, D2, N = belief.shape
        # hidden.permute(0, 2, 1)
        belief_ = rnn(hidden.transpose(2, 1).contiguous().view(-1, D1), belief.transpose(2, 1).contiguous().view(-1, D2)).view(B, N, D2).transpose(2, 1).contiguous()
        hidden = self.blocks["act_fn"](fc_embed_belief_prior(belief_))
        prior_means, _prior_std_dev = torch.chunk(fc_state_prior(hidden), 2, dim=1)
        prior_std_dev = torch.nn.Softplus()(_prior_std_dev) + self.config.noise
        states = prior_means + prior_std_dev * torch.randn_like(prior_means).to(prior_std_dev.device)

        posterior_means, posterior_std_dev = None, None
        if not infos["testing"]:
            lb_emb = torch.zeros_like(lbs.unsqueeze(1).expand(-1, 2, -1).float())
            lb_emb = fc_label(torch.scatter(lb_emb, dim=1, index=lbs[:, None].long(), value=1.0))

            hidden = self.blocks["act_fn"](fc_embed_belief_posterior(torch.cat([belief_, lb_emb], dim=1)))
            posterior_means, _posterior_std_dev = torch.chunk(fc_state_posterior(hidden), 2, dim=1)
            posterior_std_dev = torch.nn.Softplus()(_posterior_std_dev) + self.config.noise
            states = posterior_means + posterior_std_dev * torch.randn_like(posterior_std_dev).to(posterior_std_dev.device)

        return prior_means, prior_std_dev, posterior_means, posterior_std_dev, states, belief_

    def roll(self, infos):
        B, D, N = infos["corr_feat_belief"].shape
        state_Q = torch.zeros(B, self.state_size, N).cuda()
        state_K = torch.zeros(B, self.state_size, N).cuda()
        state_V = torch.zeros(B, self.state_size, N).cuda()
        belief_Q = torch.zeros(B, self.belief_size, N).cuda()
        belief_K = torch.zeros(B, self.belief_size, N).cuda()
        belief_V = torch.zeros(B, self.belief_size, N).cuda()

        for i in range(self.num_layers):
            self.config.iter_idx = i
            corr_feat_belief = infos["corr_feat_belief"]
            prior_means_Q, prior_std_dev_Q, posterior_means_Q, posterior_std_dev_Q, state_Q, belief_Q = self.translation(infos,
                                                                                                                         corr_feat_belief,
                                                                                                                         state_Q,
                                                                                                                         belief_Q,
                                                                                                                         infos["hard_gt_labels"],
                                                                                                                         self.blocks["rnn_Q"],
                                                                                                                         self.blocks["fc_embed_state_action_Q"],
                                                                                                                         self.blocks["fc_embed_belief_prior_Q"],
                                                                                                                         self.blocks["fc_state_prior_Q"],
                                                                                                                         self.blocks["fc_embed_belief_posterior_Q"],
                                                                                                                         self.blocks["fc_state_posterior_Q"],
                                                                                                                         self.blocks["fc_label_Q"])

            prior_means_K, prior_std_dev_K, posterior_means_K, posterior_std_dev_K, state_K, belief_K = self.translation(infos,
                                                                                                                         corr_feat_belief,
                                                                                                                         state_K,
                                                                                                                         belief_K,
                                                                                                                         infos["hard_gt_labels"],
                                                                                                                         self.blocks["rnn_K"],
                                                                                                                         self.blocks["fc_embed_state_action_K"],
                                                                                                                         self.blocks["fc_embed_belief_prior_K"],
                                                                                                                         self.blocks["fc_state_prior_K"],
                                                                                                                         self.blocks["fc_embed_belief_posterior_K"],
                                                                                                                         self.blocks["fc_state_posterior_K"],
                                                                                                                         self.blocks["fc_label_K"])
            prior_means_V, prior_std_dev_V, posterior_means_V, posterior_std_dev_V, state_V, belief_V = self.translation(infos,
                                                                                                                         corr_feat_belief,
                                                                                                                         state_V,
                                                                                                                         belief_V,
                                                                                                                         infos["hard_gt_labels"],
                                                                                                                         self.blocks["rnn_V"],
                                                                                                                         self.blocks["fc_embed_state_action_V"],
                                                                                                                         self.blocks["fc_embed_belief_prior_V"],
                                                                                                                         self.blocks["fc_state_prior_V"], self.blocks["fc_embed_belief_posterior_V"],
                                                                                                                         self.blocks["fc_state_posterior_V"],
                                                                                                                         self.blocks["fc_label_K"])

            forward_type = infos["forward_type"]
            infos[f"Q_mu_prior_{forward_type}"].append(prior_means_Q)
            infos[f"Q_sigma_prior_{forward_type}"].append(prior_std_dev_Q)
            infos[f"K_mu_prior_{forward_type}"].append(prior_means_K)
            infos[f"K_sigma_prior_{forward_type}"].append(prior_std_dev_K)
            infos[f"V_mu_prior_{forward_type}"].append(prior_means_V)
            infos[f"V_sigma_prior_{forward_type}"].append(prior_std_dev_V)

            infos[f"Q_mu_posterior_{forward_type}"].append(posterior_means_Q)
            infos[f"Q_sigma_posterior_{forward_type}"].append(posterior_std_dev_Q)
            infos[f"K_mu_posterior_{forward_type}"].append(posterior_means_K)
            infos[f"K_sigma_posterior_{forward_type}"].append(posterior_std_dev_K)
            infos[f"V_mu_posterior_{forward_type}"].append(posterior_means_V)
            infos[f"V_sigma_posterior_{forward_type}"].append(posterior_std_dev_V)

            infos["Q_emb"] = self.blocks["act_fn"](self.blocks["fc_embed_belief_sto_Q"](torch.cat([belief_Q, state_Q], dim=1)))
            infos["K_emb"] = self.blocks["act_fn"](self.blocks["fc_embed_belief_sto_K"](torch.cat([belief_K, state_K], dim=1)))
            infos["V_emb"] = self.blocks["act_fn"](self.blocks["fc_embed_belief_sto_V"](torch.cat([belief_V, state_V], dim=1)))

            infos = self.blocks[f'NonLocal_layer_{i}'](infos)

        return infos

    def forward_prior(self, infos):
        infos["forward_type"] = "prior_forward"
        infos = self.init_feat_embedding(infos)
        infos = self.roll(infos)
        infos["corr_feat_%s" % (infos["forward_type"])] = infos["corr_feat_belief"]
        infos["normed_corr_feat_%s" % (infos["forward_type"])] = F.normalize(infos["corr_feat_%s" % (infos["forward_type"])], p=2, dim=1)  # [B, D, N]
        return infos

    def forward_posterior(self, infos):
        infos["forward_type"] = "posterior_forward"
        infos = self.init_feat_embedding(infos)
        infos = self.roll(infos)
        infos["corr_feat_%s" % (infos["forward_type"])] = infos["corr_feat_belief"]
        infos["normed_corr_feat_%s" % (infos["forward_type"])] = F.normalize(infos["corr_feat_%s" % (infos["forward_type"])], p=2, dim=1)  # [B, D, N]
        return infos

class VBPointDSC(BaseModel):
    def __init__(self, config):
        super(VBPointDSC, self).__init__(config)

    def get_corr_encoder(self, blocks):
        blocks["corr_encoder"] = VBNonLocalNet(configs=self.config)
        return blocks

    def extract_corr_feats(self, infos):
        if infos["testing"]:
            ## prior forward
            infos = self.blocks["corr_encoder"].forward_prior(infos)
        else:
            ## posterior forward
            infos = self.blocks["corr_encoder"].forward_posterior(infos)
            ## prior forward
            infos["testing"] = True
            infos = self.blocks["corr_encoder"].forward_prior(infos)
            infos["testing"] = False
        return infos

    def get_feat_compatibility(self, infos):
        if not infos["testing"]:
            normed_corr_feat = infos["normed_corr_feat_posterior_forward"]  # [B, D, N]
            feat_compatibility = torch.matmul(normed_corr_feat.transpose(2, 1), normed_corr_feat)  # [B, N, N]
            feat_compatibility = torch.clamp(1 - (1 - feat_compatibility) / self.sigma ** 2, min=0, max=1)  # [B, N, N]
            feat_compatibility[:, torch.arange(feat_compatibility.shape[1]), torch.arange(feat_compatibility.shape[1])] = 0.  # [B, N, N]

            normed_corr_feat0 = infos["normed_corr_feat_prior_forward"]  # [B, D, N]
            feat_compatibility0 = torch.matmul(normed_corr_feat0.transpose(2, 1), normed_corr_feat0)  # [B, N, N]
            feat_compatibility0 = torch.clamp(1 - (1 - feat_compatibility0) / self.sigma ** 2, min=0, max=1)  # [B, N, N]
            feat_compatibility0[:, torch.arange(feat_compatibility0.shape[1]), torch.arange(feat_compatibility0.shape[1])] = 0.  # [B, N, N]
            infos["feat_compatibility_posterior_forward"] = feat_compatibility
            infos["feat_compatibility_prior_forward"] = feat_compatibility0
        else:
            feat_compatibility = None
            infos["feat_compatibility"] = feat_compatibility
        return infos

    def pred_confidence(self, infos):
        if infos["testing"]:
            infos["confidence_prior_forward"] = self.blocks["corr_classifier"](infos["corr_feat_prior_forward"]).squeeze(1)  # [B, N]
        else:
            infos["confidence_posterior_forward"] = self.blocks["corr_classifier"](infos["corr_feat_posterior_forward"]).squeeze(1)  # [B, N]
            infos["confidence_prior_forward"] = self.blocks["corr_classifier"](infos["corr_feat_prior_forward"]).squeeze(1)  # [B, N]
        return infos

    def get_consensus_feat_compatibility(self, infos):
        """
        :param normed_corr_feat: [B, D, N]
        :return:
        """
        normed_corr_feat = infos["normed_corr_feat_prior_forward"].transpose(2, 1)  # [B, N, D]
        B, N, D = normed_corr_feat.shape
        consensus_idxs = infos["consensus_idxs"]  # [B, K, A]
        _, K, A = consensus_idxs.shape
        consensus_feats = normed_corr_feat.gather(dim=1, index=consensus_idxs.contiguous().view(B, -1).unsqueeze(-1).expand(-1, -1, D)).contiguous().view(B, K, A, D)  # [B, K, A, D]
        consensus_feat_compatibility = torch.matmul(consensus_feats, consensus_feats.permute(0, 1, 3, 2))  # [B, K, A, A]
        consensus_feat_compatibility = torch.clamp(1 - (1 - consensus_feat_compatibility) / self.sigma ** 2, min=0)  # [B, K, A, A]
        infos["consensus_feat_compatibility"] = consensus_feat_compatibility  # [B, K, A, A]
        return infos

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
        src_consensus = infos["src_keypts"].gather(dim=1, index=consensus_idxs.contiguous().view(B, -1)[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        tgt_consensus = infos["tgt_keypts"].gather(dim=1, index=consensus_idxs.contiguous().view(B, -1)[:, :, None].expand(-1, -1, 3)).view([B, K, A, 3])  # [B, K, A, 3]
        infos["src_consensus"] = src_consensus.view([B, K, A, 3])
        infos["tgt_consensus"] = tgt_consensus.view([B, K, A, 3])
        src_consensus, tgt_consensus = src_consensus.view([B * K, A, 3]), tgt_consensus.view([B * K, A, 3])  # [B * K, A, 3]
        infos["seedwise_trans"] = rigid_transform_3d(src_consensus, tgt_consensus, consensus_weight.contiguous().view(B * K, A)).contiguous().view(B, K, 4, 4)
        infos = self.eval_trans(infos)
        infos = self.choose_best_trans(infos)


        if infos["testing"] and not infos["vb_eval"]:
            infos["final_trans"] = self.post_refinement(infos["final_trans"], infos["src_keypts"], infos["tgt_keypts"])

        if not infos["testing"] or infos["vb_eval"]:
            infos["final_labels_prior_forward"] = infos["confidence_prior_forward"]
            infos["final_labels_posterior_forward"] = infos["confidence_posterior_forward"]

        return infos

    def pick_seeds_greedy(self, infos):
        N = infos["src_dist"].shape[1]
        infos["seeds"] = torch.argsort(infos["confidence_prior_forward"], dim=1, descending=True)[:, :int(N * self.ratio)]  # [B, K]
        return infos

    def pick_seeds_nms(self, infos):
        N = infos["src_dist"].shape[1]
        dists, scores, n_seeds = infos["src_dist"], infos["confidence_prior_forward"], int(N * self.ratio)
        assert scores.shape[0] == 1
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        score_relation = score_relation.bool() | (dists[0] >= self.nms_radius).bool()
        is_local_max = score_relation.min(-1)[0].float()
        infos["seeds"] = torch.argsort(scores * is_local_max, dim=1, descending=True)[:, :n_seeds].detach()  # [B, K]
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
        normed_corr_feat, seeds = infos["normed_corr_feat_prior_forward"], infos["seeds"]
        B, D, N = normed_corr_feat.shape
        A = min(self.k, N - 1)
        consensus_idxs = knn(normed_corr_feat.transpose(2, 1), k=A, ignore_self=True, normalized=True)  # [B, N, A]
        infos["consensus_idxs"] = consensus_idxs.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, A))  # [B, K, A]

        if self.config.is_plot_attention:
            self.plot_feat_compat_maps(infos)

        return infos

    def forward(self, data):

        ## prepare input data
        infos = self.get_data(data)
        infos = self.get_spatial_compatibility(infos)

        ## extract corr feature
        infos = self.extract_corr_feats(infos)
        infos = self.get_feat_compatibility(infos)

        ## inlier confidence estimation
        infos = self.pred_confidence(infos)

        ## consensus sampling
        infos = self.consensus_sampling(infos)

        ## calculate seedwise trans
        infos = self.cal_seed_trans(infos)

        return infos