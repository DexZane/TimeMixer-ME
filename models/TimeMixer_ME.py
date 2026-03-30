import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
import math


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.seq_len // (configs.down_sampling_window**i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class AnyVariateAttention(nn.Module):
    def __init__(self, configs):
        super(AnyVariateAttention, self).__init__()
        self.d_model = configs.d_model
        self.channel_independence = configs.channel_independence
        self.N = configs.enc_in

        self.W_Q = nn.Linear(configs.d_model, configs.d_model)
        self.W_K = nn.Linear(configs.d_model, configs.d_model)
        self.W_V = nn.Linear(configs.d_model, configs.d_model)

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.output_projection = nn.Linear(configs.d_model, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, return_attention=False):
        """x: [batch_size, seq_len, d_model]"""
        x = self.layer_norm(x)
        attention_weights = None

        if self.channel_independence == 1:
            B_N, T, D = x.shape
            B = B_N // self.N
            N = self.N

            # Reshape to [B, N, T, D] and permute to [B, T, N, D] for cross-variate attention
            x_reshaped = x.view(B, N, T, D).permute(0, 2, 1, 3)

            Q = self.W_Q(x_reshaped)
            K = self.W_K(x_reshaped)
            V = self.W_V(x_reshaped)

            scale = math.sqrt(self.d_model)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            output = torch.matmul(attention_weights, V)

            # Permute back to [B, N, T, D] and reshape to [B*N, T, D]
            output = output.permute(0, 2, 1, 3).contiguous().view(B_N, T, D)
        else:
            Q = self.W_Q(x)
            K = self.W_K(x)
            V = self.W_V(x)

            scale = math.sqrt(self.d_model)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            output = torch.matmul(attention_weights, V)

        output = self.output_projection(output)
        output = output + x

        if return_attention:
            return output, attention_weights
        return output


class MTMEM(nn.Module):
    def __init__(self, configs):
        super(MTMEM, self).__init__()
        self.d_model = configs.d_model
        self.causal_levels = max(1, int(configs.causal_levels))
        valid_heads = self._get_valid_num_heads(configs.d_model, self.causal_levels)

        # 短期记忆单元
        self.short_term_memory = nn.ModuleDict(
            {
                "local_conv": nn.Conv1d(
                    configs.d_model,
                    configs.d_model,
                    kernel_size=3,
                    padding=1,
                    groups=configs.d_model,
                ),
                "attention": nn.MultiheadAttention(
                    configs.d_model, num_heads=valid_heads, dropout=configs.dropout
                ),
            }
        )

        # 长期记忆单元
        self.long_term_memory = nn.ModuleDict(
            {
                "global_pool": nn.AdaptiveAvgPool1d(1),
                "query_proj": nn.Linear(configs.d_model, configs.d_model),
                "memory_proj": nn.Linear(configs.d_model, configs.d_model),
            }
        )
        # 将memory_bank定义为模块的直接参数
        self.memory_bank = nn.Parameter(
            torch.randn(configs.num_memories, configs.d_model)
        )

        # 时间尺度自适应层
        self.scale_adapter = nn.ModuleDict(
            {
                "short_term": nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model // 2),
                    nn.LayerNorm(configs.d_model // 2),
                    nn.ReLU(),
                    nn.Linear(configs.d_model // 2, 1),
                    nn.Sigmoid(),
                ),
                "long_term": nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model // 2),
                    nn.LayerNorm(configs.d_model // 2),
                    nn.ReLU(),
                    nn.Linear(configs.d_model // 2, 1),
                    nn.Sigmoid(),
                ),
            }
        )

        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(configs.d_model * 4, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )

        # 周期性模式检测器

        # 周期性模式检测器
        self.periodicity_detector = nn.Sequential(
            nn.Conv1d(
                configs.d_model,
                configs.d_model,
                kernel_size=7,
                padding=3,
                groups=configs.d_model,
            ),
            nn.GELU(),
        )

    @staticmethod
    def _get_valid_num_heads(d_model, preferred_heads):
        for heads in range(min(preferred_heads, d_model), 0, -1):
            if d_model % heads == 0:
                return heads
        return 1

    def forward(self, x):
        B, T, D = x.size()

        # 1. 短期依赖建模
        # 局部特征提取
        local_feat = self.short_term_memory["local_conv"](x.transpose(1, 2)).transpose(
            1, 2
        )
        # 局部注意力
        short_term_attn, _ = self.short_term_memory["attention"](
            x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1)
        )
        short_term_attn = short_term_attn.transpose(0, 1)

        # 2. 长期依赖建模
        # 全局上下文
        global_context = self.long_term_memory["global_pool"](
            x.transpose(1, 2)
        ).transpose(1, 2)
        # 记忆检索
        query = self.long_term_memory["query_proj"](global_context)
        memory = self.long_term_memory["memory_proj"](self.memory_bank)

        memory_weights = F.softmax(
            torch.matmul(query, memory.t()) / math.sqrt(D), dim=-1
        )
        long_term_memory = torch.matmul(memory_weights, memory).expand(-1, T, -1)

        # 3. 周期性模式检测
        periodicity = self.periodicity_detector(x.transpose(1, 2)).transpose(1, 2)

        # 4. 时间尺度自适应融合
        short_term_weight = self.scale_adapter["short_term"](x)
        long_term_weight = self.scale_adapter["long_term"](x)

        # 5. 特征增强与融合
        enhanced_features = self.feature_enhancement(
            torch.cat(
                [
                    x * short_term_weight + short_term_attn * (1 - short_term_weight),
                    local_feat,
                    x * long_term_weight + long_term_memory * (1 - long_term_weight),
                    periodicity,
                ],
                dim=-1,
            )
        )

        return enhanced_features + x  # 残差连接


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == "moving_avg":
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError("decompsition is error")

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        # 添加Any-variate Attention层
        self.any_variate_attention = AnyVariateAttention(configs)

        # MTMEM模块的初始化
        self.mtmem = MTMEM(configs)

    def forward(self, x_list, return_attention=False):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        attn_weights_list = []
        processed_x_list = []
        for x in x_list:
            if return_attention:
                attn_out, attn_weights = self.any_variate_attention(
                    x, return_attention=True
                )
                attn_weights_list.append(attn_weights)
            else:
                attn_out = self.any_variate_attention(x)
            processed_x_list.append(self.mtmem(attn_out))
        x_list = processed_x_list

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        # down_sampling_layers=0 yields a single scale; skip cross-scale mixing safely.
        if len(season_list) == 1:
            out_season_list = [season_list[0].permute(0, 2, 1)]
            out_trend_list = [trend_list[0].permute(0, 2, 1)]
        else:
            out_season_list = self.mixing_multi_scale_season(season_list)
            # top-down trend mixing
            out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        if return_attention:
            return out_list, attn_weights_list
        return out_list


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(
                    self.configs.enc_in,
                    affine=True,
                    non_norm=True if configs.use_norm == 0 else False,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == "long_term_forecast":
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True
                )

                self.out_res_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window**i),
                            configs.seq_len // (configs.down_sampling_window**i),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window**i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True
                )
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(
                self.configs.down_sampling_window, return_indices=False
            )
        elif self.configs.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :: self.configs.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, :: self.configs.down_sampling_window, :
                ]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        B = x_enc[0].size(0)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(
                range(len(x_enc)),
                x_enc,
            ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        all_attn_weights = []
        for i in range(self.layer):
            if self.configs.output_attention:
                enc_out_list, block_attn_weights = self.pdm_blocks[i](
                    enc_out_list, return_attention=True
                )
                all_attn_weights.append(block_attn_weights)
            else:
                enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        if self.configs.output_attention:
            return dec_out, all_attn_weights
        return dec_out

    def future_multi_mixing(self, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(-1, self.configs.c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(
                range(len(x_list[0])), enc_out_list, x_list[1]
            ):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(
            range(len(x_enc)),
            x_enc,
        ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        )

        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(
                range(len(x_enc)),
                x_enc,
            ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        )

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Other tasks not implemented yet")
