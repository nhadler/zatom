"""Multimodal flow transformer (MFT).

Adapted from:
    - https://github.com/apple/ml-simplefold
    - https://github.com/facebookresearch/flow_matching
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.utils import expand_tensor_like
from torch.nn.attention import SDPBackend

from zatom.flow.interpolants import Interpolant
from zatom.models.components.losses import InterDistancesLoss
from zatom.utils import pylogger
from zatom.utils.training_utils import (
    BEST_DEVICE,
    SDPA_BACKENDS,
    HistogramTimeDistribution,
    sample_logit_normal,
    sample_uniform_rotation,
    weighted_rigid_align,
)
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = pylogger.RankedLogger(__name__)

LOSS_TIME_FUNCTIONS = Literal["none", "beta"]
GRAD_DECAY_METHODS = Literal["none", "linear_decay", "truncated_decay", "piecewise_decay"]


#################################################################################
#                           Multimodal Flow Transformer                         #
#################################################################################


class MFT2(nn.Module):
    """Multimodal flow model with an encoder-decoder architecture (v2).

    Typical usage:
    - `forward`:    Called during training to compute loss and optional stats.
    - `sample`:     Called during inference to run Euler sampler to generate new samples.

    Args:
        multimodal_model: The multimodal model to instantiate (e.g., MultimodalDiT).
        time_embedder: Time embedder module.
        dataset_embedder: Dataset embedder module.
        spacegroup_embedder: Spacegroup embedder module.
        token_pos_embedder: Token positional embedder module.
        atom_pos_embedder: Atom positional embedder module.
        atom_encoder_transformer: Multimodal atom encoder Transformer module.
        trunk: The Transformer trunk module.
        atom_decoder_transformer: Multimodal atom decoder Transformer module.
        atom_types_interpolant: Interpolant for atom types modality.
        pos_interpolant: Interpolant for atom positions modality.
        frac_coords_interpolant: Interpolant for fractional coordinates modality.
        lengths_scaled_interpolant: Interpolant for scaled lengths modality.
        angles_radians_interpolant: Interpolant for angles in radians modality.
        hidden_size: Hidden size of the model.
        token_num_heads: Number of (token) attention heads in the token trunk.
        atom_num_heads: Number of (atom) attention heads in the atom encoder/decoder.
        atom_hidden_size_enc: Hidden size of the atom encoder.
        atom_hidden_size_dec: Hidden size of the atom decoder.
        atom_n_queries_enc: Number of queries in the atom encoder.
        atom_n_keys_enc: Number of keys in the atom encoder.
        atom_n_queries_dec: Number of queries in the atom decoder.
        atom_n_keys_dec: Number of keys in the atom decoder.
        max_num_elements: Maximum number of elements in the dataset.
        batch_size_scale_factor: Factor by which to scale the global batch size when using a specific (e.g., 180M) model variant.
        use_length_condition: Whether to use the length condition.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        interdist_loss: Type of interatomic distance loss to use. If None, no interatomic distance loss is used.
        time_distribution: Distribution to sample time points from. Must be one of (`uniform`, `beta`, `histogram`).
        time_alpha_factor: Alpha factor for beta time distribution.
    """

    def __init__(
        self,
        multimodal_model: partial[Callable[..., nn.Module]],
        time_embedder: nn.Module,
        dataset_embedder: nn.Module,
        spacegroup_embedder: nn.Module,
        token_pos_embedder: nn.Module,
        atom_pos_embedder: nn.Module,
        atom_encoder_transformer: nn.Module,
        trunk: nn.Module,
        atom_decoder_transformer: nn.Module,
        atom_types_interpolant: Interpolant,
        pos_interpolant: Interpolant,
        frac_coords_interpolant: Interpolant,
        lengths_scaled_interpolant: Interpolant,
        angles_radians_interpolant: Interpolant,
        hidden_size: int = 768,
        token_num_heads: int = 12,
        atom_num_heads: int = 4,
        atom_hidden_size_enc: int = 256,
        atom_hidden_size_dec: int = 256,
        atom_n_queries_enc: int = 32,
        atom_n_keys_enc: int = 128,
        atom_n_queries_dec: int = 32,
        atom_n_keys_dec: int = 128,
        max_num_elements: int = 100,
        batch_size_scale_factor: int = 1,
        use_length_condition: bool = True,
        jvp_attn: bool = False,
        interdist_loss: InterDistancesLoss | None = None,
        time_distribution: Literal["uniform", "beta", "histogram"] = "beta",
        time_alpha_factor: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__()

        self.atom_types_interpolant = atom_types_interpolant
        self.pos_interpolant = pos_interpolant
        self.frac_coords_interpolant = frac_coords_interpolant
        self.lengths_scaled_interpolant = lengths_scaled_interpolant
        self.angles_radians_interpolant = angles_radians_interpolant

        self.batch_size_scale_factor = batch_size_scale_factor
        self.class_dropout_prob = dataset_embedder.dropout_prob
        self.jvp_attn = jvp_attn
        self.interdist_loss = interdist_loss
        self.time_alpha_factor = time_alpha_factor

        self.vocab_size = max_num_elements

        # Define time distribution
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(0, 1)
        elif time_distribution == "beta":
            self.time_distribution = torch.distributions.Beta(time_alpha_factor, 1)
        elif time_distribution == "histogram":
            # TODO: chore: make pretty
            print("Using histogram time distribution")
            self.time_distribution = HistogramTimeDistribution(
                torch.tensor([0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4])
            )
        else:
            raise ValueError(
                f"Invalid time distribution: {time_distribution}. Must be one of (`uniform`, `beta`, `histogram`)."
            )

        # Build multimodal model
        self.model = multimodal_model(
            time_embedder=time_embedder,
            dataset_embedder=dataset_embedder,
            spacegroup_embedder=spacegroup_embedder,
            token_pos_embedder=token_pos_embedder,
            atom_pos_embedder=atom_pos_embedder,
            trunk=trunk,
            atom_encoder_transformer=atom_encoder_transformer,
            atom_decoder_transformer=atom_decoder_transformer,
            hidden_size=hidden_size,
            token_num_heads=token_num_heads,
            atom_num_heads=atom_num_heads,
            atom_hidden_size_enc=atom_hidden_size_enc,
            atom_hidden_size_dec=atom_hidden_size_dec,
            atom_n_queries_enc=atom_n_queries_enc,
            atom_n_keys_enc=atom_n_keys_enc,
            atom_n_queries_dec=atom_n_queries_dec,
            atom_n_keys_dec=atom_n_keys_dec,
            max_num_elements=max_num_elements,
            use_length_condition=use_length_condition,
            add_mask_atom_type=False,
            treat_discrete_modalities_as_continuous=False,
            remove_t_conditioning=False,
            jvp_attn=jvp_attn,
            **kwargs,
        )

        # Define modalities and auxiliary tasks
        self.modals = ["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]
        self.auxiliary_tasks = self.model.auxiliary_tasks

    @typecheck
    def sample(
        self,
        feats: Dict[str, torch.Tensor],
        mask: Bool["b m"],  # type: ignore
        steps: int = 100,
        cfg_scale: float = 2.0,
        enable_zero_centering: bool = True,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
        modal_input_dict: (
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] | None
        ) = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], None]:
        """ODE-driven sampling with MFT using classifier-free guidance.

        Args:
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                ref_pos: Reference atom positions tensor.
                ref_space_uids: Reference space unique IDs tensor.
                atom_to_token: One-hot mapping from atom indices to token indices.
                atom_to_token_idx: Mapping from atom indices to token indices.
                max_num_tokens: Maximum number of unmasked tokens for each batch element.
                token_index: Indices of the tokens in the batch.
            mask: True if valid token, False if padding.
            steps: Number of integration steps for the multimodal ODE solver.
            cfg_scale: Classifier-free guidance scale.
            enable_zero_centering (bool): Whether to allow centering of continuous modalities
                at the origin after each denoising step. Defaults to ``True``.
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.
            modal_input_dict: If not None, a dictionary specifying input modalities to use and their input metadata.
                The keys should be a subset of `["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]`,
                and the values should be tuples of (x_t, t).
            kwargs: Additional keyword arguments (not used).

        Returns:
            A list of predicted modalities as a dictionary and a null variable (for sake of API compatibility).
        """
        device = mask.device
        batch_size, num_tokens = mask.shape

        if modal_input_dict is None:
            # Define time points and corresponding noised inputs for each modality
            modal_input_dict = {}
            for modal in self.modals:
                assert modal in kwargs, f"Missing required modality input: {modal}"
                t = torch.zeros(batch_size, device=BEST_DEVICE)

                if modal == "atom_types" and self.treat_discrete_modalities_as_continuous:
                    # Maybe convert atom types to random continuous (one-hot) vectors
                    kwargs[modal] = torch.randn(
                        batch_size, num_tokens, self.vocab_size, device=device
                    )

                # Set up modality inputs for classifier-free guidance
                kwargs[modal] = torch.cat([kwargs[modal], kwargs[modal]], dim=0)  # [2B, N, C]
                t = torch.cat([t, t], dim=0)  # [2B]

                modal_input_dict[modal] = (kwargs[modal], t)

        # Set up conditioning inputs for classifier-free guidance
        mask = torch.cat([mask, mask], dim=0)  # [2B, M]
        for feat in feats:
            if feat in ("dataset_idx", "spacegroup"):
                feats[feat] = torch.cat(
                    [feats[feat], torch.zeros_like(feats[feat])], dim=0
                )  # [2B]
            else:
                feats[feat] = torch.cat([feats[feat], feats[feat]], dim=0)  # [2B, M, ...]

        # Predict each modality in one step
        pred_modals = self.flow.sample(
            x_init=[
                modal_input_dict["atom_types"][0],
                modal_input_dict["pos"][0],
                modal_input_dict["frac_coords"][0],
                modal_input_dict["lengths_scaled"][0],
                modal_input_dict["angles_radians"][0],
            ],
            time_grid=None,  # For now, use same time point for all modalities
            device=mask.device,
            steps=steps,
            feats=feats,
            mask=mask,
            cfg_scale=cfg_scale,
            enable_zero_centering=enable_zero_centering,
            sdpa_backends=sdpa_backends,
        )

        # Prepare denoised modalities and remove null class samples
        denoised_modals_list = [
            {
                modal: (
                    # Take the first half of the batch
                    (
                        pred_modals[modal_idx].argmax(dim=-1)
                        if self.treat_discrete_modalities_as_continuous
                        else pred_modals[modal_idx]
                    )
                    .detach()
                    .chunk(2, dim=0)[0]
                    .reshape(batch_size * num_tokens)
                    if modal == "atom_types"
                    else pred_modals[modal_idx].detach().chunk(2, dim=0)[0]
                )
                for modal_idx, modal in enumerate(self.modals)
            }
        ]

        return denoised_modals_list, None

    @typecheck
    def forward(
        self,
        atom_types: Int["b m"],  # type: ignore
        pos: Float["b m 3"],  # type: ignore - referenced via `locals()`
        frac_coords: Float["b m 3"],  # type: ignore - referenced via `locals()`
        lengths_scaled: Float["b 1 3"],  # type: ignore - referenced via `locals()`
        angles_radians: Float["b 1 3"],  # type: ignore - referenced via `locals()`
        feats: Dict[str, torch.Tensor],
        mask: Bool["b m"],  # type: ignore
        token_is_periodic: Bool["b m"],  # type: ignore
        target_tensors: Dict[str, torch.Tensor],
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
        epsilon: float = 1e-3,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of MFT with loss calculation.

        Args:
            atom_types: Atom types tensor.
            pos: Atom positions tensor.
            frac_coords: Fractional coordinates tensor.
            lengths_scaled: Lattice lengths tensor.
            angles_radians: Lattice angles tensor.
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                ref_pos: Reference atom positions tensor.
                ref_space_uids: Reference space unique IDs tensor.
                atom_to_token: One-hot mapping from atom indices to token indices.
                atom_to_token_idx: Mapping from atom indices to token indices.
                max_num_tokens: Maximum number of unmasked tokens for each batch element.
                token_index: Indices of the tokens in the batch.
            mask: True if valid token, False if padding.
            token_is_periodic: Boolean mask indicating periodic tokens.
            target_tensors: Dictionary containing the following target tensors for loss calculation:
                atom_types: Target atom types tensor (B, N).
                pos: Target positions tensor (B, N, 3).
                frac_coords: Target fractional coordinates tensor (B, N, 3).
                lengths_scaled: Target lattice lengths tensor (B, 1, 3).
                angles_radians: Target lattice angles tensor (B, 1, 3).
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.
            epsilon: Small constant to avoid numerical issues in loss calculation.
            kwargs: Additional keyword arguments (not used).

        Returns:
            Dictionary of loss values.
        """
        device = mask.device
        batch_size, num_tokens = mask.shape

        # Sample time points and corresponding noised inputs for each modality
        modal_input_dict = {}

        modal_t, modal_r = None, None
        if self.logit_normal_time and self.unified_modal_time:
            # Sample time point(s) from a logit-normal distribution for all modalities.
            # See https://arxiv.org/abs/2509.18480 for more details.
            modal_t = 0.98 * sample_logit_normal(
                n=batch_size, m=0.8, s=1.7, device=device
            ) + 0.02 * torch.rand(batch_size, device=device)
            modal_t = modal_t * (1 - 2 * epsilon) + epsilon

            modal_r = 0.98 * sample_logit_normal(
                n=batch_size, m=0.8, s=1.7, device=device
            ) + 0.02 * torch.rand(batch_size, device=device)
            modal_r = modal_r * (1 - 2 * epsilon) + epsilon
        elif self.unified_modal_time:
            # Sample time point(s) from a uniform distribution for all modalities
            modal_t = torch.rand(batch_size, device=device) * (1 - epsilon)
            modal_r = torch.rand(batch_size, device=device) * (1 - epsilon)

        if self.enable_mean_flows and modal_t is not None and modal_r is not None:
            # When using mean flows, ensure start and end times `r` and `t` are properly set
            modal_r = torch.minimum(modal_t, modal_r)  # Ensure `r <= t`

            mf_mask = torch.randperm(batch_size)[: int(batch_size * 0.25)]
            modal_r[mf_mask] = modal_t[mf_mask]  # Set `r = t` for 25% of the batch

        for modal in self.modals:
            path = self.flow.paths[modal]

            x_0 = locals()[modal]  # Noised data
            x_1 = target_tensors[modal]  # Clean data

            t, r = modal_t, modal_r
            if self.logit_normal_time and (t is None or r is None):
                # Sample time point(s) from a logit-normal distribution
                t = 0.98 * sample_logit_normal(
                    n=batch_size, m=0.8, s=1.7, device=device
                ) + 0.02 * torch.rand(batch_size, device=device)
                t = t * (1 - 2 * epsilon) + epsilon

                r = 0.98 * sample_logit_normal(
                    n=batch_size, m=0.8, s=1.7, device=device
                ) + 0.02 * torch.rand(batch_size, device=device)
                r = r * (1 - 2 * epsilon) + epsilon
            elif t is None or r is None:
                # Sample time point(s) from a uniform distribution
                t = torch.rand(batch_size, device=device) * (1 - epsilon)
                r = torch.rand(batch_size, device=device) * (1 - epsilon)

            if self.enable_mean_flows:
                # When using mean flows, ensure start and end times `r` and `t` are properly set
                r = torch.minimum(t, r)  # Ensure `r <= t`

                if not self.unified_modal_time:
                    mf_mask = torch.randperm(batch_size)[: int(batch_size * 0.25)]
                    r[mf_mask] = t[mf_mask]  # Set `r = t` for 25% of the batch

            if modal == "atom_types" and self.treat_discrete_modalities_as_continuous:
                # Maybe convert atom types to (random) continuous (one-hot) vectors
                x_0 = torch.randn(batch_size, num_tokens, self.vocab_size, device=device)
                x_1 = F.one_hot(
                    x_1 * mask, num_classes=self.vocab_size
                ).float()  # Int[B, N] -> Float[B, N, V]

            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            dx_t = getattr(path_sample, "dx_t", None)

            # Apply mask
            if x_t.shape == (batch_size, num_tokens):  # [B, M]
                x_t *= mask
            elif x_t.shape in [
                (batch_size, num_tokens, 3),
                (batch_size, num_tokens, self.vocab_size),
            ]:  # [B, M, C]
                x_t *= mask.unsqueeze(-1)
            elif x_t.shape == (batch_size, 1, 3):  # [B, 1, 3]
                x_t *= mask.any(-1, keepdim=True).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected shape for x_t: {x_t.shape}")

            # Collect inputs
            modal_input_dict[modal] = [
                x_t,
                (t, r) if self.enable_mean_flows else t,
                (
                    dx_t
                    * expand_tensor_like(
                        input_tensor=self.c(t) * self.grad_mul,
                        expand_to=dx_t,
                    )
                    if dx_t is not None
                    else None
                ),
            ]

        # Predict velocity field for each modality
        model_output_dt = None
        if self.enable_mean_flows:
            # Average velocity prediction
            def u_func(
                atom_types,
                pos,
                frac_coords,
                lengths_scaled,
                angles_radians,
                atom_types_t,
                pos_t,
                frac_coords_t,
                lengths_scaled_t,
                angles_radians_t,
                atom_types_r,
                pos_r,
                frac_coords_r,
                lengths_scaled_r,
                angles_radians_r,
            ):
                """Average velocity prediction function."""
                return self.flow.model(
                    x=(
                        atom_types,
                        pos,
                        frac_coords,
                        lengths_scaled,
                        angles_radians,
                    ),
                    t=(
                        (atom_types_t, atom_types_r),
                        (pos_t, pos_r),
                        (frac_coords_t, frac_coords_r),
                        (lengths_scaled_t, lengths_scaled_r),
                        (angles_radians_t, angles_radians_r),
                    ),
                    feats=feats,
                    mask=mask,
                    sdpa_backends=sdpa_backends,
                )

            with torch.amp.autocast(BEST_DEVICE.type, enabled=False):
                model_output, model_output_dt, model_aux_outputs = torch.func.jvp(
                    func=u_func,
                    primals=(
                        modal_input_dict["atom_types"][0],  # atom_types
                        modal_input_dict["pos"][0],  # pos
                        modal_input_dict["frac_coords"][0],  # frac_coords
                        modal_input_dict["lengths_scaled"][0],  # lengths_scaled
                        modal_input_dict["angles_radians"][0],  # angles_radians
                        modal_input_dict["atom_types"][1][0],  # atom_types_t
                        modal_input_dict["pos"][1][0],  # pos_t
                        modal_input_dict["frac_coords"][1][0],  # frac_coords_t
                        modal_input_dict["lengths_scaled"][1][0],  # lengths_scaled_t
                        modal_input_dict["angles_radians"][1][0],  # angles_radians_t
                        modal_input_dict["atom_types"][1][1],  # atom_types_r
                        modal_input_dict["pos"][1][1],  # pos_r
                        modal_input_dict["frac_coords"][1][1],  # frac_coords_r
                        modal_input_dict["lengths_scaled"][1][1],  # lengths_scaled_r
                        modal_input_dict["angles_radians"][1][1],  # angles_radians_r
                    ),
                    tangents=(
                        modal_input_dict["atom_types"][2],  # atom_types_dt
                        modal_input_dict["pos"][2],  # pos_dt
                        modal_input_dict["frac_coords"][2],  # frac_coords_dt
                        modal_input_dict["lengths_scaled"][2],  # lengths_scaled_dt
                        modal_input_dict["angles_radians"][2],  # angles_radians_dt
                        torch.ones_like(modal_input_dict["atom_types"][1][0]),  # atom_types_dt_dt
                        torch.ones_like(modal_input_dict["pos"][1][0]),  # pos_dt_dt
                        torch.ones_like(
                            modal_input_dict["frac_coords"][1][0]
                        ),  # frac_coords_dt_dt
                        torch.ones_like(
                            modal_input_dict["lengths_scaled"][1][0]
                        ),  # lengths_scaled_dt_dt
                        torch.ones_like(
                            modal_input_dict["angles_radians"][1][0]
                        ),  # angles_radians_dt_dt
                        torch.zeros_like(modal_input_dict["atom_types"][1][1]),  # atom_types_dr_dt
                        torch.zeros_like(modal_input_dict["pos"][1][1]),  # pos_dr_dt
                        torch.zeros_like(
                            modal_input_dict["frac_coords"][1][1]
                        ),  # frac_coords_dr_dt
                        torch.zeros_like(
                            modal_input_dict["lengths_scaled"][1][1]
                        ),  # lengths_scaled_dr_dt
                        torch.zeros_like(
                            modal_input_dict["angles_radians"][1][1]
                        ),  # angles_radians_dr_dt
                    ),
                    has_aux=True,
                )
        else:
            # Instantaneous velocity prediction
            model_output, model_aux_outputs = self.flow.model(
                x=(
                    modal_input_dict["atom_types"][0],  # atom_types
                    modal_input_dict["pos"][0],  # pos
                    modal_input_dict["frac_coords"][0],  # frac_coords
                    modal_input_dict["lengths_scaled"][0],  # lengths_scaled
                    modal_input_dict["angles_radians"][0],  # angles_radians
                ),
                t=(
                    modal_input_dict["atom_types"][1],  # atom_types_t, maybe atom_types_r as well
                    modal_input_dict["pos"][1],  # pos_t, maybe pos_r as well
                    modal_input_dict["frac_coords"][
                        1
                    ],  # frac_coords_t, maybe frac_coords_r as well
                    modal_input_dict["lengths_scaled"][
                        1
                    ],  # lengths_scaled_t, maybe lengths_scaled_r as well
                    modal_input_dict["angles_radians"][
                        1
                    ],  # angles_radians_t, maybe angles_radians_r as well
                ),
                feats=feats,
                mask=mask,
                sdpa_backends=sdpa_backends,
            )

        # Preprocess target (velocity) tensors - and unit test for SO(3) equivariance - if requested
        for idx, modal in enumerate(self.modals):
            config = self.flow.modality_configs[idx]
            path = self.flow.paths[modal]

            if not config.get("should_rigid_align", False):
                continue

            # Unit test for SO(3) equivariance
            if self.test_so3_equivariance:
                output_modal = model_output[idx]
                rand_rot_mat = sample_uniform_rotation(
                    shape=output_modal.shape[:-2], dtype=output_modal.dtype, device=device
                )

                # Augment (original) output modality
                expected_output_modal_aug = torch.einsum(
                    "bij,bjk->bik", output_modal, rand_rot_mat.transpose(-2, -1)
                )

                # Augment input modality for new forward pass
                model_output_aug, _ = self.flow.model(
                    x=(
                        modal_input_dict["atom_types"][0],  # atom_types
                        torch.einsum(
                            "bij,bjk->bik",
                            modal_input_dict["pos"][0],
                            rand_rot_mat.transpose(-2, -1),
                        ),  # pos
                        modal_input_dict["frac_coords"][0],  # frac_coords
                        modal_input_dict["lengths_scaled"][0],  # lengths_scaled
                        modal_input_dict["angles_radians"][0],  # angles_radians
                    ),
                    t=(
                        modal_input_dict["atom_types"][
                            1
                        ],  # atom_types_t, maybe atom_types_r as well
                        modal_input_dict["pos"][1],  # pos_t, maybe pos_r as well
                        modal_input_dict["frac_coords"][
                            1
                        ],  # frac_coords_t, maybe frac_coords_r as well
                        modal_input_dict["lengths_scaled"][
                            1
                        ],  # lengths_scaled_t, maybe lengths_scaled_r as well
                        modal_input_dict["angles_radians"][
                            1
                        ],  # angles_radians_t, maybe angles_radians_r as well
                    ),
                    feats=feats,
                    mask=mask,
                    sdpa_backends=sdpa_backends,
                )
                output_modal_aug = model_output_aug[idx]

                assert torch.allclose(
                    output_modal_aug, expected_output_modal_aug, atol=1e-3
                ), "The (Platonic) model's output positions must be SO(3)-equivariant."

            if config.get("x_1_prediction", False):
                # When parametrizing x_1, directly align target modality to predicted modality
                pred_modal = model_output[idx]
                target_tensors[modal] = weighted_rigid_align(
                    pred_modal, target_tensors[modal], mask=mask
                )
                continue  # No need to re-sample if predicting x_1 directly

            # Otherwise, re-sample target modality velocity via one-step Euler iteration
            pred_modal_vel = model_output[idx]
            target_modal = target_tensors[modal]

            x_t = modal_input_dict[modal][0]
            t = modal_input_dict[modal][1]
            modal_t = t[0] if self.enable_mean_flows else t

            # Perform a one-step Euler iteration to get predicted clean data
            pred_modal = (
                x_t - pred_modal_vel
                if self.enable_mean_flows
                else path.velocity_to_target(
                    velocity=pred_modal_vel,
                    x_t=x_t,
                    t=expand_tensor_like(t, x_t),
                )
            )

            # Align target modality to predicted modality
            x_0 = locals()[modal]  # Noised data
            x_1 = target_tensors[modal] = weighted_rigid_align(
                pred_modal, target_modal, mask=mask
            )  # Clean (aligned) data

            # Re-sample probability path - NOTE: alignment of `model_output_dt` is not currently handled
            path_sample = path.sample(t=modal_t, x_0=x_0, x_1=x_1)
            modal_input_dict[modal][2] = path_sample.dx_t

        # Calculate the loss for each modality
        target_atom_types = (
            # Mask out -100 padding to construct target atom types
            F.one_hot(target_tensors["atom_types"] * mask, num_classes=self.vocab_size).float()
            if self.treat_discrete_modalities_as_continuous
            else target_tensors["atom_types"] * mask
        )
        training_loss, training_loss_dict = self.flow.training_loss(
            x_1=[
                target_atom_types,
                target_tensors["pos"],
                target_tensors["frac_coords"],
                target_tensors["lengths_scaled"],
                target_tensors["angles_radians"],
            ],
            x_t=[
                modal_input_dict["atom_types"][0],
                modal_input_dict["pos"][0],
                modal_input_dict["frac_coords"][0],
                modal_input_dict["lengths_scaled"][0],
                modal_input_dict["angles_radians"][0],
            ],
            dx_t=[
                modal_input_dict["atom_types"][2],
                modal_input_dict["pos"][2],
                modal_input_dict["frac_coords"][2],
                modal_input_dict["lengths_scaled"][2],
                modal_input_dict["angles_radians"][2],
            ],
            t=[
                modal_input_dict["atom_types"][1],
                modal_input_dict["pos"][1],
                modal_input_dict["frac_coords"][1],
                modal_input_dict["lengths_scaled"][1],
                modal_input_dict["angles_radians"][1],
            ],
            model_output=model_output,
            model_output_dt=model_output_dt,
            detach_loss_dict=False,
        )
        training_loss.detach_()  # NOTE: Will manually re-aggregate losses below

        # Mask and aggregate losses
        loss_dict = {}
        for idx, modal in enumerate(self.modals):
            modal_time_t = (
                modal_input_dict[modal][1][0]
                if self.enable_mean_flows
                else modal_input_dict[modal][1]
            )
            modal_loss_value = training_loss_dict[modal]

            pred_modal = model_output[idx]
            target_modal = target_tensors[modal]

            loss_mask = mask.float()
            loss_token_is_periodic = token_is_periodic.float()

            target_shape = target_modal.shape

            # Prepare loss masks
            if modal in ("pos", "frac_coords"):
                loss_mask = mask.unsqueeze(-1).float()
                loss_token_is_periodic = token_is_periodic.unsqueeze(-1).float()
            elif modal in ("lengths_scaled", "angles_radians"):
                loss_mask = torch.ones(
                    target_shape, dtype=torch.float32, device=target_modal.device
                )
                loss_token_is_periodic = (
                    token_is_periodic.any(-1, keepdim=True).unsqueeze(-1).float()
                )  # NOTE: A periodic sample is one with any periodic atoms
            elif modal == "atom_types" and self.treat_discrete_modalities_as_continuous:
                loss_mask = mask.unsqueeze(-1)

            # Mask loss values
            modal_loss_value = modal_loss_value * loss_mask

            if modal in (
                "frac_coords",
                "lengths_scaled",
                "angles_radians",
            ):  # Periodic (crystal) losses
                modal_loss_value = modal_loss_value * loss_token_is_periodic
            elif modal == "pos":  # Non-periodic (molecule) losses
                modal_loss_value = modal_loss_value * (1 - loss_token_is_periodic)

            # Collect losses
            time_weighted_modal_loss_value = self.loss_time_fn(
                x=modal_loss_value,
                t=expand_tensor_like(modal_time_t, expand_to=modal_loss_value),
            )
            loss_dict[f"{modal}_loss"] = time_weighted_modal_loss_value.mean()

        # Aggregate losses
        loss_dict["loss"] = sum(loss_dict[f"{modal}_loss"] for modal in self.modals)

        # Add auxiliary losses
        for aux_idx, aux_task in enumerate(self.auxiliary_tasks):
            model_aux_output = model_aux_outputs[aux_idx]
            # Requested auxiliary task → compute loss
            if aux_task in target_tensors:
                mask_aux = (
                    mask.any(-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
                    if model_aux_output.squeeze().ndim == 1
                    else mask.unsqueeze(-1)  # (B, M, 1)
                ).float()
                target_aux = target_tensors[aux_task]
                aux_loss_value = F.l1_loss(
                    model_aux_output * mask_aux,
                    target_aux * mask_aux,
                    reduction="mean",
                )
                loss_dict[f"aux_{aux_task}_loss"] = aux_loss_value
            # Unused auxiliary task → add zero loss to computational graph
            else:
                loss_dict[f"aux_{aux_task}_loss"] = (model_aux_output * 0.0).mean()

            # Maybe log per-atom auxiliary loss
            if aux_task == "global_energy":
                if aux_task in target_tensors:
                    per_atom_aux_loss = F.l1_loss(
                        model_aux_output.squeeze()
                        / (mask.sum(dim=-1).clamp(min=1.0)),  # Avoid division by zero
                        target_tensors[aux_task] / (mask.sum(dim=-1).clamp(min=1.0)),
                        reduction="mean",
                    )
                else:
                    per_atom_aux_loss = (model_aux_output * 0.0).mean()

                loss_dict[f"aux_{aux_task}_per_atom_loss"] = per_atom_aux_loss

            loss_dict["loss"] += loss_dict[f"aux_{aux_task}_loss"]

        return loss_dict
