#python train_dreambooth.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#  --output_dir="./output-models/" \
#  --with_prior_preservation --prior_loss_weight=1.0 \
#  --resolution=256 \
#  --train_batch_size=1 \
#  --train_text_encoder \
#  --mixed_precision="fp16" \
#  --use_8bit_adam \
#  --gradient_accumulation_steps=1 \
#  --gradient_checkpointing \
#  --learning_rate=1e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=200 \
#  --num_class_images=300 \
#  --max_train_steps=2000 \
#  --concepts_list="concepts_list.json"\
#  --save_interval=500

python train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir="./output-models/" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=256 \
  --train_batch_size=4 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=200 \
  --not_cache_latents \
  --num_class_images=300 \
  --max_train_steps=10000 \
  --concepts_list="concepts_list.json"\
  --save_interval=500
