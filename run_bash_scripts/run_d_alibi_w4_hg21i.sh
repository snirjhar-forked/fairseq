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

dirname="checkpoints/gt_wikitext-103_dist_no_ape_alibi_hg21i_w4_gelu"
max_update=64000
warmup_updates=16000
max_lr=0.001
min_lr=0.00001

num_windows=4
shuffle_type=half_gaussian
shuffle_size="[0.2+0.01*i for i in range(16)]"

lr_period=`expr $max_update - $warmup_updates`
max_tokens=`expr 3072 \* $bsize`

(( nrank == 0 )) && mkdir -p "$dirname"
(( nrank != 0 )) && sleep 1
torchrun   --nproc_per_node="$ngpu"\
           --nnodes="$nnodes" --node_rank="$nrank" --master_addr="$maddr"\
           --master_port="$mport"\
            $(which fairseq-train) --user-dir ./gt/ --task language_modeling \
                data-bin/wikitext-103 \
                --save-dir "$dirname" \
                --log-file "$dirname"/log.log \
                --arch gt_lm_wiki103 \
                --no-token-positional-embeddings\
                --use-alibi\
                --rpe-embedding-dim 0\
                --num-windows "$num_windows"\
                --shuffle-type "$shuffle_type"\
                --shuffle-size "$shuffle_size"\
                --activation-fn gelu\
                --max-update "$max_update" --warmup-updates "$warmup_updates" --lr-period-updates "$lr_period" \
                --lr "$max_lr" --min-lr "$min_lr" --warmup-init-lr 1e-06 \
                --lr-scheduler cosine --t-mult 2 --lr-shrink 0.75 --stop-min-lr 1e-09 \
                --optimizer adam --clip-norm 0.1 \
                --criterion adaptive_loss --max-tokens "$max_tokens" --update-freq 1 --tokens-per-sample 3072 --seed 1 \
                --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp \
                --fp16 --fp16-init-scale 1
                #--max-tokens 9216 
