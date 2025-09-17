import argparse
from .karras_diffusion import KarrasDenoiser, FlowDenoiser, ContinuousKarrasDenoiser
from .unet import UNetModel
import numpy as np
from .network_karras import SongUNet, DhariwalUNet
from .network_dit import DiT_models
from .network_edm2 import EDM2_models
from .network_udit import UDiT_models
from .network_nGPT import nDiT_models
NUM_CLASSES = 1000


def cm_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.1,
        training_mode="consistency_distillation",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=40,
        end_scales=40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="karras",
    )
    return res


def create_model_and_diffusion(args):
    if args.model_type == "openai_unet":
        model = create_model(
            args.image_size,
            args.num_in_channels,
            args.num_channels,
            args.num_res_blocks,
            channel_mult=args.channel_mult,
            learn_sigma=args.learn_sigma,
            num_classes=args.num_classes,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            use_new_attention_order=args.use_new_attention_order,
        )
    elif args.model_type in ["song_unet", "dhariwal_unet"]:
        attention_ds = tuple(int(res) for res in args.attention_resolutions.split(","))
        channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))
        print(f"Atten Res: {attention_ds}")
        print(f"Channel Mult: {channel_mult}")
        if args.model_type == "song_unet":
            unet = SongUNet
        else:
            unet = DhariwalUNet
        model = unet(img_resolution=args.image_size,
                     in_channels=args.num_in_channels,
                     out_channels=(args.num_in_channels if not args.learn_sigma else args.num_in_channels*2),
                     label_dim=args.num_classes,
                     augment_dim=0,
                     model_channels=args.num_channels,
                     channel_mult=channel_mult,
                     channel_mult_emb=4,
                     num_blocks=args.num_res_blocks,
                     attn_resolutions=attention_ds,
                     dropout=args.dropout,
                     label_dropout=0)
    elif "EDM2" in args.model_type:
        model = EDM2_models[args.model_type](img_resolution=args.image_size,
                                             img_channels=args.num_in_channels,
                                             label_dim=args.num_classes,
                                             dropout=args.dropout)
    elif "DiT" in args.model_type and not "U-DiT" in args.model_type and not "nDiT" in args.model_type:
        
        if args.use_repa:
            depth, _, z_dim = parse_repa_enc_info_v2(args.repa_enc_info)
        else:
            depth = None
            z_dim = None
        
        model = DiT_models[args.model_type](input_size=args.image_size,
                                            in_channels=args.num_in_channels,
                                            num_classes=args.num_classes,
                                            learn_sigma=args.learn_sigma,
                                            linear_act=args.linear_act,
                                            norm_type = args.norm_type,
                                            use_rope = args.use_rope,
                                            wo_norm = args.wo_norm,
                                            num_register = args.num_register,
                                            use_repa=args.use_repa,
                                            cond_mapping = args.cond_mapping,
                                            freq_type = args.freq_type,
                                            z_depth=depth,
                                            z_dim=z_dim,
                                            separate_cond=args.separate_cond,
                                            projector_dim=args.projector_dim,
                                            repa_mapper=args.repa_mapper,
                                            mar_mapper_num_res_blocks=args.mar_mapper_num_res_blocks,
                                            uw=args.uw,
                                            cond_mixing = args.cond_mixing)
    elif "U-DiT" in args.model_type:
        model = UDiT_models[args.model_type](input_size=args.image_size,
                                            in_channels=args.num_in_channels,
                                            num_classes=args.num_classes,
                                            learn_sigma=args.learn_sigma,
                                            no_scale = args.no_scale)
    elif "nDiT" in args.model_type:
        model = nDiT_models[args.model_type](input_size=args.image_size,
                                            in_channels=args.num_in_channels,
                                            num_classes=args.num_classes,
                                            learn_sigma=args.learn_sigma)
    else:
        print("No network as define")
        exit(0)
            
    if args.fwd == "ve":
        diffusion = KarrasDenoiser(args=args, sigma_data=args.sigma_data, rho=args.rho)
    elif args.fwd == "ve_cont":
        diffusion = ContinuousKarrasDenoiser(args=args, sigma_data=args.sigma_data)
    elif args.fwd == "flow":
        diffusion = FlowDenoiser(args, sigma_data=args.sigma_data)
    elif args.fwd == "ksd":
        diffusion = KSD_Denoiser(args=args, sigma_data=args.sigma_data)
    return model, diffusion


def create_model(
    image_size,
    num_in_channels,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    num_classes=0,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=num_in_channels,
        model_channels=num_channels,
        out_channels=(num_in_channels if not learn_sigma else num_in_channels*2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if num_classes>0 else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    ict = True,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        else:
            raise NotImplementedError
        return float(target_ema), int(scales)
    
    # Discretization curriculum improvement
    def improve_scale_fn(step):
        temp = np.floor(total_steps/(np.log2(np.floor(end_scales/start_scales))+1))
        scales = min(start_scales*2**np.floor(step/temp), end_scales) + 1
        if target_ema_mode == "adaptive":
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
        elif target_ema_mode == "fix":
            target_ema = start_ema
        return float(target_ema), int(scales)

    if ict: 
        return improve_scale_fn 
    else: 
        return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def parse_repa_enc_info(repa_enc_info):
    """Parse the repa encoder info string into dictionaries mapping depths to encoder types
    and encoder types to their z dimensions.
    
    Args:
        repa_enc_info (str): Format "depth1:encoder1;depth2:encoder2;..."
        Example: "4:dinov2-vit-b;6:dinov2-vit-b" means use dinov2-vit-b at depths 4 and 6
        
    Returns:
        depth_to_encoder (dict): Maps depth to encoder_type, e.g. {4: "dinov2-vit-b", 6: "dinov2-vit-b"}
        encoder_to_z_dim (dict): Maps encoder_type to z_dim, e.g. {"dinov2-vit-b": 768}
    """
    # Parse encoder info into dictionary {depth: encoder_type}
    depth_to_encoder = {}
    for part in repa_enc_info.split(';'):
        depth, enc_type = part.split(':')
        depth_to_encoder[int(depth)] = enc_type
    
    # Define z dimensions for each encoder type
    encoder_to_z_dim = {
        "dinov2-vit-b": 768,
        "clip-vit-L": 1024,
        # Add other encoder types as needed
    }
    
    return depth_to_encoder, encoder_to_z_dim

def parse_repa_enc_info_v2(repa_enc_info):
    depth, enc_type = repa_enc_info.split(':')
    # Define z dimensions for each encoder type
    encoder_to_z_dim = {
        "dinov2-vit-b": 768,
        "clip-vit-L": 1024,
        # Add other encoder types as needed
    }
    return int(depth), enc_type, encoder_to_z_dim[enc_type]