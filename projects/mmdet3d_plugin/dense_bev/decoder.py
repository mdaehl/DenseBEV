# ---------------------------------------------------
# Copyright (c) DenseBEV 2025. All rights reserved.
# ---------------------------------------------------

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner.base_module import BaseModule, ModuleList
from typing import Optional
from mmcv.cnn.bricks.transformer import build_transformer_layer, TRANSFORMER_LAYER
import copy
import torch
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
from .nms import suppress_detections
import torch.nn as nn
from einops import rearrange, repeat
from .misc_memory import nerf_positional_encoding, pos2posemb1d, MLN
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer
import warnings


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DenseBEVDecoder(BaseModule):
    def __init__(
        self,
        transformerlayers: dict,
        num_layers: int,
        embed_dims: int,
        nms_cfg: dict,
        look_forward_twice: bool,
        return_intermediate: bool,
        use_memory: bool,
        use_memory_loss: bool = False,
        use_velo_memory: bool = False,
        allow_memory_suppress: bool = False,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.look_forward_twice = look_forward_twice
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.use_memory = use_memory

        self.nms_threshold = nms_cfg["threshold"]

        self.n_self_attn_heads = transformerlayers["attn_cfgs"][0]["num_heads"]

        assert isinstance(transformerlayers, dict)
        # create decoder layers
        layers = [
            build_transformer_layer(
                copy.deepcopy(transformerlayers)  # create individual copy
            )
            for _ in range(num_layers)
        ]

        # decoder layer MultiheadAttention
        self.layers = ModuleList(layers)
        self.ref_point_head = MLP(
            self.embed_dims, self.embed_dims // 2, self.embed_dims, 2
        )
        self.norm = nn.LayerNorm(self.embed_dims)

        self.fp16_enabled = False

        if not use_memory:
            assert not use_memory_loss and not allow_memory_suppress and not use_velo_memory

        self.allow_memory_suppress = allow_memory_suppress
        self.use_memory_loss = use_memory_loss
        self.use_velo_memory = use_velo_memory

        if self.use_memory:
            if self.use_velo_memory:
                self.ego_pose_pe = MLN(180)
                self.ego_pose_memory = MLN(180)
            else:
                self.ego_pose_pe = MLN(156)
                self.ego_pose_memory = MLN(156)

            self.time_embedding = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            )

    def be_distinct(
        self,
        reference_points: torch.Tensor,
        tmp: torch.Tensor,
        query: torch.Tensor,
        lid: int,  # layer id
        self_attn_mask: torch.Tensor,
        n_prop: int
    ) -> torch.Tensor:
        bs = query.shape[0]
        n_query = query.shape[1]

        # number of queries used for suppression
        if not self.allow_memory_suppress:
            n_relev = n_query - n_prop
        else:
            n_relev = n_query

        # filter queries, reference points and interim proposal (tmp)
        query = query[:, :n_relev]
        reference_points = reference_points[:, :n_relev]
        tmp = tmp[:, :n_relev]

        # mask of relevant queries taking into account previously suppressed ones
        dis_mask = self_attn_mask[:, :n_relev, :n_relev]

        # get scores
        cls_head = self.cache_dict["cls_branches"][lid]
        scores = cls_head(query)
        scores, class_ids = scores.max(-1)
        scores = scores.sigmoid()   # get highest scores per query

        # get boxes, reference_points are in sigmoid space and contain tmp already
        proposals = self.decode_boxes(reference_points, tmp)

        mask_shape = dis_mask[0].shape
        attn_mask_list = []
        for i in range(bs):
            single_bboxes = proposals[i]
            single_scores = scores[i]
            inv_attn_mask = ~dis_mask[i * self.n_self_attn_heads][0]
            ori_index = inv_attn_mask.nonzero().view(-1)

            # perform nms (returns keep idxs)
            keep_idxs = suppress_detections(single_bboxes[ori_index], 
                                            single_scores[ori_index], 
                                            self.nms_threshold)

            real_keep_index = ori_index[keep_idxs]

            attn_mask = torch.ones(mask_shape, dtype=bool, device=query.device)
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False

            attn_mask = attn_mask[None].repeat(self.n_self_attn_heads, 1, 1)

            attn_mask_list.append(attn_mask)

        attn_mask = torch.cat(attn_mask_list)

        # will be used in loss and inference
        if self.use_memory:
            if self.use_memory_loss:
                distinct_query_mask = torch.ones(self.n_self_attn_heads * bs, n_query, n_query, device=self_attn_mask.device, dtype=bool)
            else:
                distinct_query_mask = torch.zeros(self.n_self_attn_heads * bs, n_query, n_query, device=self_attn_mask.device, dtype=bool)
            distinct_query_mask[:, :n_relev, :n_relev] = ~attn_mask
        else:
            distinct_query_mask = ~attn_mask

        self.cache_dict["distinct_query_mask"].append(distinct_query_mask)

        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, :n_relev, :n_relev] = attn_mask

        return self_attn_mask

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: nn.ModuleList,
        tgt_prop: torch.Tensor = None,
        query_pos_prop: torch.Tensor = None,
        reference_points_prop: torch.Tensor = None,
        **kwargs,
    ):
        bs, n_query = query.shape[:2]
        if self.use_memory:
            n_prop = tgt_prop.shape[1]
        else:
            n_prop = 0
        self_attn_mask = torch.zeros((n_query+ n_prop, n_query + n_prop), device=query.device).bool()
        self_attn_mask = self_attn_mask[None].repeat(bs * self.n_self_attn_heads, 1, 1)

        intermediate = []
        if self.use_memory:
            intermediate_reference_points = [torch.cat([reference_points, reference_points_prop], dim=1)]
        else:
            intermediate_reference_points = [reference_points]
        self.cache_dict["distinct_query_mask"] = []

        for lid, layer in enumerate(self.layers):            
            # get query pos from reference points
            query_sine_embed = coordinate_to_encoding(
                reference_points[..., :2], num_feats=self.embed_dims // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)

            if self.use_memory and lid == 0:
                query, query_pos = self.temporal_alignment_current(query, query_pos)
                query = torch.cat([query, tgt_prop], dim=1).contiguous()
                query_pos = torch.cat([query_pos, query_pos_prop], dim=1).contiguous()
                reference_points = torch.cat([reference_points, reference_points_prop], dim=1).contiguous()

            reference_points_input = reference_points[..., :2].unsqueeze(2)

            query = layer(
                query=query,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                self_attn_mask=self_attn_mask,
                **kwargs,
            )

            assert reference_points.shape[-1] == 3

            tmp = reg_branches[lid](query)

            new_reference_points = torch.zeros_like(reference_points)
            new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                reference_points[..., :2]
            )
            new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                reference_points[..., 2:3]
            )

            new_reference_points[..., :3] = new_reference_points[..., :3].sigmoid()
            reference_points = new_reference_points.detach()

            if lid < (len(self.layers) - 1):
                self_attn_mask = self.be_distinct(
                    reference_points=reference_points,
                    tmp=tmp,
                    query=query,
                    self_attn_mask=self_attn_mask,
                    lid=lid,
                    n_prop=n_prop,
                )

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return query, reference_points

    def decode_boxes(
        self,
        reference_points: torch.Tensor,  # x y z center
        tmp: torch.Tensor,
    ):  # (cx,cy,l,w,cz,h,sin(φ),cos(φ)) - maybe velocity at end
        # center
        cx = self.pc_range[0] + (
            reference_points[..., :1] * (self.pc_range[3] - self.pc_range[0])
        )
        cy = self.pc_range[1] + (
            reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
        )
        cz = self.pc_range[2] + (
            reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2])
        )

        # size
        w = tmp[..., 2:3].exp()
        l = tmp[..., 3:4].exp()
        h = tmp[..., 5:6].exp()

        # rotation
        rot_sine = tmp[..., 6:7]
        rot_cosine = tmp[..., 7:8]
        rot = torch.atan2(rot_sine, rot_cosine)

        # combine
        if tmp.size(-1) > 8:  # nuscenes schema with velocity
            vx = tmp[..., 8:9]
            vy = tmp[..., 9:10]
            proposal = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
        else:
            proposal = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

        return proposal
    
    def temporal_alignment_current(self, tgt, query_pos):
        bs, n_q = query_pos.shape[:2]

        # set motion
        zero_memory_timestamp = torch.zeros(bs, n_q, 1, device=tgt.device)
        if self.use_velo_memory:
            zero_reference_points_fill = torch.zeros(bs, n_q, 3, device=tgt.device)  # represent zero velocity and timestamp
        else:
            zero_reference_points_fill = torch.zeros(bs, n_q, 1, device=tgt.device)  # represent zero timestamp

        rec_ego_pose = rearrange(repeat(torch.eye(4, device=tgt.device)[:3], "m n -> bs n_q m n", bs=bs, n_q=n_q), "bs n_q m n -> bs n_q (m n)")
        rec_ego_motion = torch.cat([zero_reference_points_fill, rec_ego_pose], dim=-1)
        rec_ego_motion = nerf_positional_encoding(rec_ego_motion)

        # update tgt/queries via MLN
        tgt = self.ego_pose_memory(tgt, rec_ego_motion)
        # update query pos via MLN and (zero) timestamp
        query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
        query_pos += self.time_embedding(pos2posemb1d(zero_memory_timestamp))
        return tgt, query_pos


@TRANSFORMER_LAYER.register_module()
class MemoryDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                            f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    dim = 1 if self.batch_first else 0
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=dim)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=dim)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
