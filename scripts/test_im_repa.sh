export MASTER_PORT=10139

# for epoch in 600 625 650 675
# do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
#                 --ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256/DiT_ve_repa_register_0_B_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict/checkpoints/0000${epoch}.pt \
#                 --seed 42 \
#                 --dataset imagenet_256 \
#                 --image-size 32 \
#                 --num-in-channels 4 \
#                 --num-classes 1000 \
#                 --steps 513 \
#                 --batch-size $((256*1)) \
#                 --num-channels 128 \
#                 --num-head-channels 64 \
#                 --num-res-blocks 4 \
#                 --resblock-updown \
#                 --model-type DiT-B/2 \
#                 --channel-mult 1,2,3,4 \
#                 --attention-resolutions 16,8 \
#                 --sampler onestep \
#                 --ts 0,256,512 \
#                 --normalize-matrix statistic/latent_imagenet256_stat.npy \
#                 --compute-fid \
#                 --ema \
#                 --linear-act gate_relu \
#                 --num-register 0 \
#                 --norm-type rms \
#                 --freq-type prev_mlp \
#                 --use-rope \
#                 --cfg-scale 1.5 \
#                 --c-type edm \
#                 --fwd ve \
#                 --p-mean -0.8 \
#                 --p-std 1.5 \
#                 --sigma-data 0.5 \
#                 --rho 7 \ 
# done


# subset imagenet 256
# for epoch in 675 # 625 650 675 700
# do
#         CUDA_VISIBLE_DEVICES=4 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
#                 --ckpt /research/cbim/medical/qd66/lct_v2_new/subset_imagenet_256/eq_DiT_ve_repa_register_0_B_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_min-snr_22/checkpoints/0000${epoch}.pt \
#                 --seed 42 \
#                 --dataset subset_imagenet_256 \
#                 --image-size 32 \
#                 --num-in-channels 4 \
#                 --num-classes 25 \
#                 --steps 513 \
#                 --batch-size $((48*1)) \
#                 --num-channels 128 \
#                 --num-head-channels 64 \
#                 --num-res-blocks 4 \
#                 --resblock-updown \
#                 --model-type DiT-L/2 \
#                 --channel-mult 1,2,3,4 \
#                 --attention-resolutions 16,8 \
#                 --sampler onestep \
#                 --ts 0,256,512 \
#                 --normalize-matrix statistic/stats_25.npy \
#                 --compute-fid \
#                 --ema \
#                 --linear-act relu \
#                 --num-register 0 \
#                 --norm-type rms \
#                 --freq-type prev_mlp \
#                 --use-rope \
#                 --cfg-scale 1.5 \
#                 --c-type edm \
#                 --fwd ve \
#                 --p-mean -0.8 \
#                 --p-std 1.5 \
#                 --sigma-data 0.5 \
#                 --rho 7 \
#                 --cond-mixing \
#                 # --wo-norm \
#                 # --no-scale \
# done

# for epoch in 600 # 625 650 675 700
# do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
#                 --ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256/DiT_ve_repa_register_0_B_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict/checkpoints/0000${epoch}.pt \
#                 --seed 42 \
#                 --dataset imagenet_256 \
#                 --image-size 32 \
#                 --num-in-channels 4 \
#                 --num-classes 1000 \
#                 --steps 513 \
#                 --batch-size $((256*1)) \
#                 --num-channels 128 \
#                 --num-head-channels 64 \
#                 --num-res-blocks 4 \
#                 --resblock-updown \
#                 --model-type DiT-B/2 \
#                 --channel-mult 1,2,3,4 \
#                 --attention-resolutions 16,8 \
#                 --sampler onestep \
#                 --ts 0,256,512 \
#                 --normalize-matrix statistic/latent_imagenet256_stat.npy \
#                 --compute-fid \
#                 --ema \
#                 --linear-act relu \
#                 --num-register 0 \
#                 --norm-type rms \
#                 --freq-type prev_mlp \
#                 --use-rope \
#                 --cfg-scale 1.75 \
#                 --c-type edm \
#                 --fwd ve \
#                 --p-mean -0.8 \
#                 --p-std 1.5 \
#                 --sigma-data 0.5 \
#                 --rho 7 \
#                 --cond-mixing \

# done

# resume525_mean-0.8_std1.5_diff0.70_lamb50

for epoch in 650
do
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
                --ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256_va/va_DiT_ve_repa_register_0_L_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict_nograd_resume500/checkpoints/0000${epoch}.pt \
                --seed 42 \
                --dataset imagenet_256_va \
                --image-size 16 \
                --num-in-channels 32 \
                --num-classes 1000 \
                --steps 641 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type DiT-L/2 \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,64,128,256,512 \
                --normalize-matrix statistic/latents_stats.pt \
                --compute-fid \
                --ema \
                --linear-act relu \
                --num-register 0 \
                --norm-type rms \
                --freq-type prev_mlp \
                --use-rope \
                --cfg-scale 1.5 \
                --c-type edm \
                --fwd ve \
                --p-mean -0.8 \
                --p-std 1.5 \
                --sigma-data 0.5 \
                --rho 7 \
                --vae va_vae \
                --cond-mixing \

done