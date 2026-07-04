cd /home/xiaoqin_feng/workspace/seed_repro

export SR=/home/xiaoqin_feng/workspace/warp_code
export R=/home/xiaoqin_feng/workspace/LongCat-AudioDiT
export PY=/home/xiaoqin_feng/miniconda3/bin/python3.13
export PYTHON_BIN=$PY
export PYTHONPATH=$R
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda:0
export DATA=$SR/data/seedtts_testset
export RES=$SR/results
export SDIR=$SR/scripts
export SEED=$SR/eval/seed-tts-eval

# 1B fp32：

$PY -u scripts/run_paired_hard.py \
--mode fp32 \
--tag fp32 \
--base 1024 \
--sets zh,en,hard \
--model_dir meituan-longcat/LongCat-AudioDiT-1B \
--device cuda:0

cd $SEED

bash $SDIR/run_wer_cer.sh $DATA/zh/meta.lst     $SR/gen/paired/fp32/zh   zh $RES/pf_fp32_zh_cer.txt
bash $SDIR/run_wer_cer.sh $DATA/en/meta.lst     $SR/gen/paired/fp32/en   en $RES/pf_fp32_en_wer.txt
bash $SDIR/run_wer_cer.sh $DATA/zh/hardcase.lst $SR/gen/paired/fp32/hard zh $RES/pf_fp32_hard_cer.txt

$PY $SDIR/sim_eval.py $SR/gen/paired/fp32/zh/wav_res_ref_text   $RES/pf_fp32_zh_sim.txt   cuda:0
$PY $SDIR/sim_eval.py $SR/gen/paired/fp32/en/wav_res_ref_text   $RES/pf_fp32_en_sim.txt   cuda:0
$PY $SDIR/sim_eval.py $SR/gen/paired/fp32/hard/wav_res_ref_text $RES/pf_fp32_hard_sim.txt cuda:0

# 3.5B fp32：

cd $SR

$PY -u scripts/run_paired_hard.py \
--mode fp32 \
--tag fp32_3.5b \
--base 1024 \
--sets zh,en,hard \
--model_dir meituan-longcat/LongCat-AudioDiT-3.5B \
--device cuda:0

cd $SEED

bash $SDIR/run_wer_cer.sh $DATA/zh/meta.lst     $SR/gen/paired/fp32_3.5b/zh   zh $RES/pf_fp32_3.5b_zh_cer.txt
bash $SDIR/run_wer_cer.sh $DATA/en/meta.lst     $SR/gen/paired/fp32_3.5b/en   en $RES/pf_fp32_3.5b_en_wer.txt
bash $SDIR/run_wer_cer.sh $DATA/zh/hardcase.lst $SR/gen/paired/fp32_3.5b/hard zh $RES/pf_fp32_3.5b_hard_cer.txt

$PY $SDIR/sim_eval.py $SR/gen/paired/fp32_3.5b/zh/wav_res_ref_text   $RES/pf_fp32_3.5b_zh_sim.txt   cuda:0
$PY $SDIR/sim_eval.py $SR/gen/paired/fp32_3.5b/en/wav_res_ref_text   $RES/pf_fp32_3.5b_en_sim.txt   cuda:0
$PY $SDIR/sim_eval.py $SR/gen/paired/fp32_3.5b/hard/wav_res_ref_text $RES/pf_fp32_3.5b_hard_sim.txt cuda:0