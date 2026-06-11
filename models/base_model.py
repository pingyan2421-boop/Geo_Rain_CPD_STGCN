import tensorflow as tf
from models.layers import *
import numpy as np
import os


def build_model(inputs, stage, n_his, Ks, Kt, blocks, keep_prob, n_pred=1, residual_scale=0.1,
                rain_features=None, rain_susceptibility=None, hydro_flow_weight=None, dynamic_rain_graph=False,
                temporal_encoder="none", forecast_context=None, source_aware_residual=False,
                forecast_source_index=None, source_delta_scale=0.35, return_aux=False):
    """
    基于变化点检测 (CPD) 和 FiLM 调制的 STGCN 核心架构。
    """
    x = inputs
    aux = {}
    rain_context = None
    if rain_features is not None and temporal_encoder == "tcn_attention":
        rain_context, temporal_attention = hydro_tcn_attention(rain_features)
        aux["hydro_context"] = rain_context
        aux["temporal_attention"] = temporal_attention

    # 堆叠 ST-Conv 块
    for i, channels in enumerate(blocks):
        x = st_conv_block(
            x,
            stage,
            Ks,
            Kt,
            channels,
            i,
            keep_prob,
            rain_features=rain_features,
            rain_context=rain_context,
            forecast_context=forecast_context,
            rain_susceptibility=rain_susceptibility,
            hydro_flow_weight=hydro_flow_weight,
            dynamic_rain_graph=dynamic_rain_graph,
        )

    # 最终输出层
    last_input = inputs[:, -1:, :, :]
    physics_prior = tf.tile(last_input, [1, n_pred, 1, 1])

    residual_base = output_layer(x, 0, 'output_layer', n_pred=n_pred)
    residual_body = residual_base
    if source_aware_residual and forecast_context is not None and forecast_source_index is not None:
        source_col = tf.gather(tf.cast(forecast_context, tf.float32), int(forecast_source_index), axis=1)
        source_col = tf.reshape(tf.clip_by_value(source_col, 0.0, 1.0), [-1, 1, 1, 1])
        official_delta = output_layer(x, 0, 'official_residual_delta', n_pred=n_pred, zero_init=True)
        fallback_delta = output_layer(x, 0, 'fallback_residual_delta', n_pred=n_pred, zero_init=True)
        source_delta = (1.0 - source_col) * official_delta + source_col * fallback_delta
        residual_body = residual_base + float(source_delta_scale) * source_delta
        aux["forecast_source_switch_mean"] = tf.reduce_mean(source_col)
        aux["source_delta"] = source_delta
        aux["baseline_pred"] = physics_prior + residual_scale * residual_base

    residual = residual_scale * residual_body
    y = physics_prior + residual

    # ---------------------------------------------------------
    # 学术加分项：计算 Copy Loss (Persistence Loss)
    # 用于保证平稳阶段预测值的稳定性
    # ---------------------------------------------------------
    copy_loss = tf.reduce_mean(tf.square(residual))

    if return_aux:
        return y, copy_loss, aux
    return y, copy_loss


def model_save(sess, saver, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    saver.save(sess, os.path.join(path, 'model'), global_step=epoch)
    print(f'>> Model successfully saved at epoch {epoch}')
