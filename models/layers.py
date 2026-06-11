# layers.py —— 变量命名空间修复版
import tensorflow as tf


def hydro_tcn_attention(rain_features, hidden_units=16, dilations=(1, 2, 4), scope='hydro_tcn_attention'):
    """Encode rain/hydro history with multi-scale TCN and temporal attention."""
    with tf.compat.v1.variable_scope(scope):
        rain = tf.cast(rain_features, tf.float32)
        x = rain
        for i, dilation in enumerate(dilations):
            conv = tf.compat.v1.layers.conv1d(
                x,
                filters=hidden_units,
                kernel_size=3,
                padding='same',
                dilation_rate=int(dilation),
                activation=tf.nn.relu,
                name=f'tcn_d{dilation}_{i}',
            )
            if x.get_shape().as_list()[-1] == hidden_units:
                x = x + 0.5 * conv
            else:
                x = conv
        score = tf.compat.v1.layers.dense(x, 1, activation=None, name='attention_score')
        attention = tf.nn.softmax(score, axis=1, name='temporal_attention')
        context = tf.reduce_sum(x * attention, axis=1, name='hydro_context')
        return context, tf.squeeze(attention, axis=-1)


def film_layer(x, stage, c_out, scope, rain_features=None, rain_context=None, forecast_context=None):
    """
    FiLM (Feature-wise Linear Modulation) 调制层。
    通过学习阶段嵌入来动态调整特征图的缩放 (gamma) 和偏移 (beta)。
    """
    with tf.compat.v1.variable_scope(scope):
        s = tf.cast(stage, tf.float32)
        s_mean = tf.reduce_mean(s, axis=1, keepdims=True)  # [B,1]
        condition = s_mean
        if rain_features is not None:
            rain = tf.cast(rain_features, tf.float32)
            rain_mean = tf.reduce_mean(rain, axis=1)
            condition = tf.concat([condition, rain_mean], axis=1)
        if rain_context is not None:
            condition = tf.concat([condition, tf.cast(rain_context, tf.float32)], axis=1)
        if forecast_context is not None:
            condition = tf.concat([condition, tf.cast(forecast_context, tf.float32)], axis=1)

        gamma = tf.compat.v1.layers.dense(condition, c_out, activation=tf.nn.tanh, name='gamma')
        beta = tf.compat.v1.layers.dense(condition, c_out, activation=None, name='beta')

        gamma = tf.reshape(gamma, [-1, 1, 1, c_out])
        beta = tf.reshape(beta, [-1, 1, 1, c_out])

        # 🌟 Stage-aware scaling（修复版）
        stage_scale = tf.cast(stage, tf.float32) / 5.0
        scale = tf.reduce_mean(stage_scale, axis=1, keepdims=True)
        scale = tf.reshape(scale, [-1, 1, 1, 1])  # ⭐关键修复

        gamma = 0.2 * gamma
        beta = 0.05 * beta

        return x * (1.0 + gamma) + beta


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='GLU'):
    _, T, N, _ = x.get_shape().as_list()
    x_pad = tf.pad(x, [[0, 0], [Kt - 1, 0], [0, 0], [0, 0]])
    in_channels = x.get_shape().as_list()[-1]

    # 权重和偏置的定义（依赖外部的 variable_scope 保证唯一性）
    wt = tf.compat.v1.get_variable('wt', shape=[Kt, 1, in_channels, 2 * c_out if act_func == 'GLU' else c_out])
    bt = tf.compat.v1.get_variable('bt', initializer=tf.zeros([2 * c_out if act_func == 'GLU' else c_out]))

    x_conv = tf.nn.conv2d(x_pad, wt, [1, 1, 1, 1], padding='VALID') + bt
    if act_func == 'GLU':
        x_p, x_q = tf.split(x_conv, 2, axis=-1)
        return x_p * tf.nn.sigmoid(x_q)
    else:
        return tf.nn.relu(x_conv)


def rain_graph_gates(x, rain_features, rain_susceptibility, hydro_flow_weight=None, gate_strength=0.15,
                     forecast_context=None):
    _, T, N, _ = x.get_shape().as_list()
    rain = tf.cast(rain_features, tf.float32)
    susceptibility = tf.reshape(tf.cast(rain_susceptibility, tf.float32), [1, 1, N, 1])
    susceptibility = tf.clip_by_value(susceptibility, 0.0, 1.0)
    susceptibility = susceptibility - tf.reduce_mean(susceptibility)
    if hydro_flow_weight is not None:
        flow_weight = tf.reshape(tf.cast(hydro_flow_weight, tf.float32), [1, 1, N, 1])
        susceptibility = susceptibility * tf.clip_by_value(flow_weight, 0.5, 1.5)
    hydro_intensity = tf.reduce_mean(tf.nn.relu(rain), axis=[1, 2])
    if forecast_context is not None:
        forecast_intensity = tf.reduce_mean(tf.nn.relu(tf.cast(forecast_context, tf.float32)), axis=1)
        hydro_intensity = 0.5 * hydro_intensity + 0.5 * forecast_intensity
    hydro_intensity = tf.minimum(hydro_intensity, 3.0)
    hydro_intensity = tf.reshape(hydro_intensity, [-1, 1, 1, 1])
    alpha = tf.compat.v1.get_variable(
        'rain_gate_alpha',
        initializer=tf.constant(float(gate_strength), dtype=tf.float32),
    )
    gate = 1.0 + alpha * hydro_intensity * susceptibility
    return tf.clip_by_value(gate, 0.75, 1.25)


def spatio_conv_layer(x, Ks, c_in, c_out, rain_features=None, rain_susceptibility=None,
                      hydro_flow_weight=None, dynamic_rain_graph=False, forecast_context=None):
    _, T, N, _ = x.get_shape().as_list()
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]

    gate = None
    if dynamic_rain_graph and rain_features is not None and rain_susceptibility is not None:
        gate = rain_graph_gates(
            x,
            rain_features,
            rain_susceptibility,
            hydro_flow_weight=hydro_flow_weight,
            forecast_context=forecast_context,
        )
        x = x * gate

    x_reshaped = tf.reshape(x, [-1, N, c_in])
    x_gcn = tf.matmul(kernel, x_reshaped)
    x_gcn = tf.reshape(x_gcn, [-1, T, N, c_in])
    if gate is not None:
        x_gcn = x_gcn * gate

    ws = tf.compat.v1.get_variable('ws', shape=[1, 1, c_in, c_out])
    bs = tf.compat.v1.get_variable('bs', initializer=tf.zeros([c_out]))
    return tf.nn.conv2d(x_gcn, ws, [1, 1, 1, 1], padding='SAME') + bs


def st_conv_block(x, stage, Ks, Kt, channels, scope, keep_prob, rain_features=None,
                  rain_context=None, forecast_context=None, rain_susceptibility=None,
                  hydro_flow_weight=None, dynamic_rain_graph=False):
    c_si, c_t, c_oo = channels

    with tf.compat.v1.variable_scope(f'st_block_{scope}'):

        # 1️⃣ 时间卷积1
        with tf.compat.v1.variable_scope('tconv1'):
            x_t1 = temporal_conv_layer(x, Kt, c_si, c_t)

        # 2️⃣ 空间卷积
        with tf.compat.v1.variable_scope('sconv'):
            x_s = spatio_conv_layer(
                x_t1,
                Ks,
                c_t,
                c_t,
                rain_features=rain_features,
                rain_susceptibility=rain_susceptibility,
                hydro_flow_weight=hydro_flow_weight,
                dynamic_rain_graph=dynamic_rain_graph,
                forecast_context=forecast_context,
            )

        # 3️⃣ 🌟 FiLM（正确调用）
        x_film = film_layer(
            x_s,
            stage,
            c_t,
            'stage_film',
            rain_features=rain_features,
            rain_context=rain_context,
            forecast_context=forecast_context,
        )

        # 4️⃣ 🌟 稳定残差融合（推荐）
        x_mod = 0.85 * x_s + 0.15 * x_film

        # 5️⃣ 时间卷积2
        with tf.compat.v1.variable_scope('tconv2'):
            x_t2 = temporal_conv_layer(x_mod, Kt, c_t, c_oo)

        x_ln = layer_norm(x_t2, 'ln')

        return tf.nn.dropout(x_ln, keep_prob)

def layer_norm(x, scope):
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keepdims=True)
    with tf.compat.v1.variable_scope(scope):
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.compat.v1.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        return (x - mu) / (tf.sqrt(sigma + 1e-6)) * gamma + beta


def output_layer(x, Ko, scope, n_pred=1, zero_init=False):
    with tf.compat.v1.variable_scope(scope):
        _, T, N, C = x.get_shape().as_list()

        w = tf.compat.v1.get_variable('w', shape=[T, 1, C, C])
        b = tf.compat.v1.get_variable('b', initializer=tf.zeros([C]))
        res = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='VALID') + b

        out_initializer = tf.zeros_initializer() if zero_init else None
        w_out = tf.compat.v1.get_variable('w_out', shape=[1, 1, C, n_pred], initializer=out_initializer)
        b_out = tf.compat.v1.get_variable('b_out', initializer=tf.zeros([n_pred]))
        y = tf.nn.conv2d(res, w_out, [1, 1, 1, 1], padding='SAME') + b_out
        # [B, 1, N, H] -> [B, H, N, 1]
        return tf.transpose(y, [0, 3, 2, 1])
