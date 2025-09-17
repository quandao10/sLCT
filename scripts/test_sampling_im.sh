export MASTER_PORT=10700

# for epoch in 625 650 675 700
# do
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_cond.py \
        --ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256_va/va_DiT_ve_repa_register_0_XL_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict_resume440/checkpoints/0000650.pt \
        --seed 0 \
        --dataset latent_imagenet256 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 1000 \
        --steps 513 \
        --batch-size $((16)) \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --sampler onestep \
        --ts 0,120,220,320,420,512 \
        --normalize-matrix statistic/stats_25.npy \
        --real-img-dir ../real_samples/celeba_256/ \
        --compute-fid \
        --ema \
        --linear-act relu \
        --norm-type rms \
        --num-register 0 \
        --cfg-scale 3.0 \
        --freq-type prev_mlp \
        --use-rope \
# done