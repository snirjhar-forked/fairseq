#!/usr/bin/bash
echo "======================================================"
jobn="$1"
echo "JOB #$jobn"
cfgf=`<c$jobn`
echo "cfgf=$cfgf"
echo "------------------------------------------------------"

nnodes="${2:-1}"
echo "nnodes=$nnodes"
bsize="${3:-2}"
echo "bsize=$bsize"
nrank="${4:-0}"
echo "nrank=$nrank"
echo "------------------------------------------------------"

(( nrank == 0 )) && hostname > "$jobn".maddr

maddr=`<"$jobn".maddr`
echo "maddr=$maddr"
mport="22345"
echo "mport=$mport"
ngpu="$(nvidia-smi -L | wc -l)"
echo "ngpu=$ngpu"
echo "======================================================"

dirname="checkpoints/gt_enwik8_dist_no_ape_alibi_new_f4_hg1015i_w4_4k"
max_update=16000
warmup_updates=4000
max_lr=0.001
min_lr=0.000001

num_windows=4
shuffle_type='$["none"]*4+["half_gaussian"]*12'
shuffle_size="[0.1+0.015*i for i in range(16)]"

lr_period=`expr $max_update - $warmup_updates`
max_tokens=`expr 4096 \* $bsize`

(( nrank == 0 )) && mkdir -p "$dirname"
(( nrank != 0 )) && sleep 1
torchrun   --nproc_per_node="$ngpu"\
           --nnodes="$nnodes" --node_rank="$nrank" --master_addr="$maddr"\
           --master_port="$mport"\
            $(which fairseq-train) --user-dir ./gt/ --task language_modeling \
                data-bin/enwik8\
                --save-dir "$dirname" \
                --log-file "$dirname"/log.log \
                --arch gt_lm_enwik8_new \
                --no-token-positional-embeddings\
                --use-alibi\
                --rpe-embedding-dim 0\
                --num-windows "$num_windows"\
                --shuffle-type "$shuffle_type"\
                --shuffle-size "$shuffle_size"\
                --max-update "$max_update" --warmup-updates "$warmup_updates" --lr-period-updates "$lr_period" \
                --lr "$max_lr" --min-lr "$min_lr" --warmup-init-lr 1e-06 \
                --lr-scheduler cosine --t-mult 2 --lr-shrink 0.75 --stop-min-lr 1e-09 \
                --criterion cross_entropy --optimizer adam --clip-norm 0.1 \
                --max-tokens "$max_tokens" --update-freq 1 --tokens-per-sample 4096 --seed 1 \
                --fp16 --fp16-init-scale 1
                # --amp
                #--max-tokens 9216 
