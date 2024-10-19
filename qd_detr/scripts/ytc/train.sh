dset_name=ytc
ctx_mode=video_tef
v_feat_types=slowfast
t_feat_type=gpt2
exp_id=exp

######## data paths
train_path=data/ytc/train.jsonl
eval_path=data/ytc/val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../ytc_features


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/ytc_slowfast_features/)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi

# text features
t_feat_dir=${feat_root}/ytc_gpt2_feats/
t_feat_dim=1600

#### training
bsz=16
num_workers=8
n_epoch=100

# lr=0.0001

results_root=results_ytc

seed=9076

list="1e-4 1e-5 1e-6"

for lr in $list
do
  echo $lr

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--eval_bsz 32 \
--results_root ${results_root} \
--exp_id org_16_${lr}_${seed} \
--seed ${seed} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--clip_length 8 \
--max_v_l 10000 \
--max_q_l 500 \
--num_workers 8 \
--max_es_cnt 300 \
${@:1}
done
