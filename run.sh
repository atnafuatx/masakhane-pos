export MAX_LENGTH=164
export LABEL=home/atnafuatx/masakhane-pos/data/amh/
export OUTPUT_DIR=amh_pos_${BERT_MODEL}
export BATCH_SIZE=32
export NUM_EPOCHS=5
export SAVE_STEPS=10000
export SEED=1
for model in Atnafu/amh-base-LAFT Atnafu/amh_large-LAFT Atnafu/amh_small-LAFT
do
	export BERT_MODEL=${model}
        CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_pos.py --data_dir data/amh/ \
        --model_type xlmroberta \
        --model_name_or_path $BERT_MODEL \
        --labels  $LABEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --do_train \
        --do_eval \
        --do_predict
done