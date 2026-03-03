# src/vrpo_elbo.py

import torch
import torch.nn.functional as F


import torch


def forward_process_text_only(
    batch_ids: torch.Tensor,
    prompt_index: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
):
    """
    Batched forward_process，参考 LLaDA 官方实现，但允许每个样本的 prompt_len 不同。

    Args:
        batch_ids:    [B, L] 整条序列 (prompt + answer [+ pad])
        prompt_index: [B, L] bool, True 表示 prompt token（不 mask）
        mask_id:      [MASK] 的 token id，例如 126336
        eps:          为了数值稳定，mask 比例的最小值

    Returns:
        noisy_batch: [B, L]  把被选中的 answer token 替换为 mask_id 后的序列
        p_mask:      [B, L]  每个样本的 mask 比例（标量）广播到序列长度，
                              后续可用作 1/t 或 t 的近似
    """
    device = batch_ids.device
    B, L = batch_ids.shape

    # 每个样本的 prompt_len & answer_len
    prompt_len = prompt_index.sum(dim=1)           # [B]
    target_len = (L - prompt_len).to(torch.long)   # [B]

    # 初始化
    is_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    p = torch.ones(B, device=device)  # 先全部置为 1，避免 0

    for i in range(B):
        pl = int(prompt_len[i].item())
        tl = int(target_len[i].item())

        # 没有 answer（极端情况）：不做 mask，p[i] 保持为 1.0
        if tl <= 0:
            continue

        # 随机采样 k_i ∈ {1, ..., tl}
        k_i = torch.randint(1, tl + 1, (1,), device=device).item()

        # 在 answer 区间长度 tl 上构造掩码：
        # 先取前 k_i 个为 True，再随机打乱
        indices = torch.arange(tl, device=device)
        cur = indices < k_i                  # 长度 tl 的 bool 向量，前 k_i 个为 True
        perm = torch.randperm(tl, device=device)
        cur = cur[perm]                      # 打乱
        # 填回到整条序列中的 answer 段 [pl : pl + tl)
        is_mask[i, pl:pl + tl] = cur

        # 该样本的 mask 比例 r_i = k_i / tl，保证 ≥ 1/tl
        r_i = float(k_i) / float(tl)
        p[i] = max(r_i, eps)                 # 防止过小，避免后续 1/p 爆炸

    # 根据 is_mask 把 token 换成 mask_id
    noisy_batch = torch.where(
        is_mask,
        torch.full_like(batch_ids, mask_id),
        batch_ids,
    )

    # p_mask: [B, L]，每个样本一个标量广播
    p_mask = p.unsqueeze(1).expand(B, L)     # [B, L]

    return noisy_batch, p_mask


def gather_text_logits(
    logits: torch.Tensor,             # [B, L_logits, V]
    audio_info_lengths: torch.Tensor, # [B]
    L_text: int,
):
    """
    把带 speech 的 logits 映射回纯文本坐标系 [B, L_text, V]，
    方便在纯文本空间上做各种 loss。
    """
    device = logits.device
    B, L_logits, V = logits.shape
    audio_info_lengths = audio_info_lengths.to(device).view(B)

    # text_positions: [B, L_text] = 0..L_text-1
    text_positions = torch.arange(L_text, device=device).unsqueeze(0).expand(B, -1)
    ai = audio_info_lengths.unsqueeze(1)  # [B,1]

    extra = L_logits - L_text  # speech_len
    if extra < 0:
        raise ValueError("L_logits < L_text??")

    # j_text < ai -> j_logits = j_text
    # j_text >= ai -> j_logits = j_text + extra
    shift_mask = text_positions >= ai
    logits_positions = text_positions + shift_mask.long() * extra  # [B, L_text]

    # flatten gather
    batch_id = torch.arange(B, device=device).unsqueeze(1).expand(B, L_text)
    flat_idx = batch_id * L_logits + logits_positions  # [B, L_text]
    logits_flat = logits.reshape(B * L_logits, V)
    gathered = logits_flat[flat_idx]  # [B, L_text, V]
    return gathered


def get_logits_diffa(model,
                     seq_batch: torch.Tensor,
                     batch_other: dict,
                     t_for_model: torch.Tensor):
    """
    适配 DIFFA 的 logits 计算函数。
    """
    if t_for_model.dim() == 2:
        t_for_model = t_for_model[:, 0]       # [B]

    outputs = model(
        input_ids=seq_batch,
        input_audio_features=batch_other["input_audio_features"],
        input_audio_features_mask=batch_other["input_audio_features_mask"],
        audio_info_lengths=batch_other["audio_info_lengths"],
        prompt_lengths=batch_other["prompt_lengths"],
        num_prompt_tokens=None,
        labels=None,
        t=t_for_model,  # [B]
    )
    return outputs.logits


def elbo_mc_diffa_single(
    model,
    seq_batch: torch.Tensor,
    prompt_index: torch.Tensor,
    batch_other: dict,
    mc_steps: int = 8,
    mask_id: int = 126336,
):
    """
    不做 antithetic、只对单个模型进行 MC 估计 ELBO。

    Args:
        model: DIFFAModel (policy 或 ref)
        seq_batch: [B, L]
        prompt_index: [B, L] bool
        batch_other: dict, audio 等
        mc_steps: MC 次数
        mask_id: [MASK] id

    Returns:
        elbo: [B]
    """
    device = seq_batch.device
    B, L = seq_batch.shape

    elbo_sum = torch.zeros(B, device=device)

    for _ in range(mc_steps):
        perturbed, p_mask = forward_process_text_only(seq_batch, prompt_index, mask_id)
        mask_index = perturbed.eq(mask_id)   # [B, L]

        t_for_model = p_mask  

        logits = get_logits_diffa(model, perturbed, batch_other, t_for_model)
        # logits: [B, L, V]

        # 只在 mask 位置算 CE
        logits_m = logits[mask_index]        # [M, V]
        target_m = seq_batch[mask_index]     # [M]
        p_mask_m = p_mask[mask_index]        # [M]

        loss_tok = F.cross_entropy(
            logits_m, target_m, reduction="none"
        ) / p_mask_m                          # [M]

        # 把 token loss 聚合回样本级别 [B]
        loss_full = torch.zeros(B, device=device)
        idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, L)
        loss_mat = torch.zeros(B, L, device=device)
        loss_mat[mask_index] = loss_tok
        loss_full = loss_mat.sum(dim=1)      # 每个样本所有 mask token loss 之和

        # elbo = -loss
        elbo_sum = elbo_sum - loss_full

    elbo = elbo_sum / mc_steps  # [B]
    return elbo



def estimate_elbo_both(
    model_theta,
    model_ref,
    seq_batch: torch.Tensor,
    prompt_index: torch.Tensor,
    batch_other: dict,
    mc_steps: int = 8,
    mask_id: int = 126336,
):

    device = seq_batch.device
    B, L_text = seq_batch.shape

    audio_info_lengths = batch_other["audio_info_lengths"].to(device).view(B)  # ★ 用这个
    elbo_theta_sum = torch.zeros(B, device=device)
    elbo_ref_sum = torch.zeros(B, device=device)

    for _ in range(mc_steps):
        # 1) 文本坐标上做随机 mask
        perturbed, p_mask = forward_process_text_only(seq_batch, prompt_index, mask_id)
        mask_index_text = perturbed.eq(mask_id)   # [B, L_text]

        # 2) 构造 t（先简单用每个样本一个标量）
        t_for_model = p_mask[:, 0]   # [B]

        # 3) 前向：logits 长度 = L_text + speech_len
        logits_theta = get_logits_diffa(model_theta, perturbed, batch_other, t_for_model)
        with torch.no_grad():
            logits_ref = get_logits_diffa(model_ref, perturbed, batch_other, t_for_model)

        assert logits_theta.shape == logits_ref.shape
        B2, L_logits, V = logits_theta.shape
        assert B2 == B

        extra = L_logits - L_text
        if extra < 0:
            raise ValueError(
                f"Unexpected logits length: L_logits={L_logits} < L_text={L_text}"
            )

        # 4) 文本坐标 -> logits 坐标
        # text_positions[i, :] = 0 .. L_text-1
        text_positions = torch.arange(L_text, device=device).unsqueeze(0).expand(B, -1)
        ai = audio_info_lengths.unsqueeze(1)  # [B,1]

        shift_mask = text_positions >= ai
        logits_positions = text_positions + shift_mask.long() * extra  # [B, L_text]

        # 5) flatten gather：只取 mask 的位置
        logits_theta_flat = logits_theta.reshape(B * L_logits, V)
        logits_ref_flat = logits_ref.reshape(B * L_logits, V)
        batch_id = torch.arange(B, device=device).unsqueeze(1).expand(B, L_text)
        flat_indices = batch_id * L_logits + logits_positions  # [B, L_text]

        flat_indices_m = flat_indices[mask_index_text]  # [M]
        logits_theta_m = logits_theta_flat[flat_indices_m]  # [M, V]
        logits_ref_m   = logits_ref_flat[flat_indices_m]    # [M, V]

        target_m = seq_batch[mask_index_text]  # [M]
        p_mask_m = p_mask[mask_index_text]     # [M]

        loss_tok_theta = F.cross_entropy(
            logits_theta_m, target_m, reduction="none"
        ) / p_mask_m
        loss_tok_ref = F.cross_entropy(
            logits_ref_m, target_m, reduction="none"
        ) / p_mask_m

        # 6) 聚合回样本级别
        loss_mat_theta = torch.zeros(B, L_text, device=device)
        loss_mat_ref = torch.zeros(B, L_text, device=device)
        loss_mat_theta[mask_index_text] = loss_tok_theta
        loss_mat_ref[mask_index_text] = loss_tok_ref

        loss_full_theta = loss_mat_theta.sum(dim=1)
        loss_full_ref = loss_mat_ref.sum(dim=1)

        elbo_theta_sum -= loss_full_theta
        elbo_ref_sum   -= loss_full_ref

    elbo_theta = elbo_theta_sum / mc_steps
    elbo_ref = elbo_ref_sum / mc_steps
    return elbo_theta, elbo_ref



def mdm_sft_loss_on_batch(
    model,
    seq_batch: torch.Tensor,       # [B, L_text]
    prompt_index: torch.Tensor,    # [B, L_text] bool
    batch_other: dict,
    mask_id: int = 126336,
):
    """
    一个和 Stage1 一致、但在 VRPO 阶段使用的 MDM-SFT loss：
    - 只在文本 token 上 mask + CE
    - speech 通过 logits 映射绕开，不参与 loss
    """
    device = seq_batch.device
    B, L_text = seq_batch.shape

    perturbed, p_mask = forward_process_text_only(seq_batch, prompt_index, mask_id)
    mask_index = perturbed.eq(mask_id)   # [B, L_text]

    if not mask_index.any():
        return torch.zeros([], device=device)

    # t 先用每个样本一个标量（和 VRPO ELBO 一致）
    t_for_model = p_mask[:, 0]  # [B]

    # 1) 跑 DIFFA 拿 logits（带 speech）
    logits_full = get_logits_diffa(model, perturbed, batch_other, t_for_model)
    B2, L_logits, V = logits_full.shape
    assert B2 == B

    # 2) 映射回纯文本坐标系 [B, L_text, V]
    audio_info_lengths = batch_other["audio_info_lengths"].to(device).view(B)
    logits_text = gather_text_logits(logits_full, audio_info_lengths, L_text)  # ★★关键

    # 3) 只在被 mask 的文本位置算 CE
    logits_m = logits_text[mask_index]        # [M, V]
    target_m = seq_batch[mask_index]          # [M]
    p_mask_m = p_mask[mask_index]             # [M]

    loss_tok = F.cross_entropy(
        logits_m, target_m, reduction="none"
    ) / p_mask_m

    loss = loss_tok.mean()
    return loss

def vrpo_loss_on_batch(
    model_theta,
    model_ref,
    batch: dict,
    beta: float = 0.2,      # 原文 β=0.2
    mc_steps: int = 8,
    mask_id: int = 126336,
    sft_weight: float = 0.05,
):
    device = batch["chosen_input_ids"].device

    seq_chosen = batch["chosen_input_ids"]
    seq_rejected = batch["rejected_input_ids"]
    B, Lc_total = seq_chosen.shape
    _, Lr_total = seq_rejected.shape

    prompt_lengths = batch["prompt_lengths"].to(device)  # [B,1]

    # 构造 prompt_index
    idx_c = torch.arange(Lc_total, device=device).unsqueeze(0).repeat(B, 1)
    idx_r = torch.arange(Lr_total, device=device).unsqueeze(0).repeat(B, 1)
    prompt_index_c = idx_c < prompt_lengths
    prompt_index_r = idx_r < prompt_lengths

    # 1) VRPO(DPO) 部分：ELBO 差分
    elbo_c_theta, elbo_c_ref = estimate_elbo_both(
        model_theta,
        model_ref,
        seq_chosen,
        prompt_index_c,
        batch,
        mc_steps=mc_steps,
        mask_id=mask_id,
    )

    elbo_r_theta, elbo_r_ref = estimate_elbo_both(
        model_theta,
        model_ref,
        seq_rejected,
        prompt_index_r,
        batch,
        mc_steps=mc_steps,
        mask_id=mask_id,
    )

    s = beta * ((elbo_c_theta - elbo_c_ref) - (elbo_r_theta - elbo_r_ref))  # [B]
    dpo_loss = -torch.log(torch.sigmoid(s)).mean()
    print(f"chose: {elbo_c_theta - elbo_c_ref} | reject: {elbo_r_theta - elbo_r_ref}")
    # 2) MDM-SFT 辅助 loss：只在 chosen 上做一次 masked CE 即可
    # sft_loss = mdm_sft_loss_on_batch(
    #     model=model_theta,
    #     seq_batch=seq_chosen,
    #     prompt_index=prompt_index_c,
    #     batch_other=batch,
    #     mask_id=mask_id,
    # )

    total_loss = dpo_loss #+ sft_weight * sft_loss
    return total_loss
