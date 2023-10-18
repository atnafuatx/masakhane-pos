for LANG in amh orm tir 
do
  for m in EthioNLP/Ethio-xmlr-large-250k-30e EthioNLP/Ethio-xmlr-base-250k EthioNLP/ethio_afro_small
  do
   export BERT_MODEL=${m}
    for j in 1 2 3 4 5
     do
      export MAX_LENGTH=200
      
      export OUTPUT_DIR=/content/drive/MyDrive/POS_ours/result/${m}_${LANG}
      export TEXT_RESULT=test_result$j.txt
      export TEXT_PREDICTION=test_predictions$j.txt
      export BATCH_SIZE=8
      export NUM_EPOCHS=20
      export SAVE_STEPS=10000000
      export SEED=$j

      CUDA_VISIBLE_DEVICES=0 python3 ../train_pos.py --data_dir /home/atnafuatx/POS/${LANG}/ \
      --model_type xlmroberta \
      --model_name_or_path $BERT_MODEL \
      --output_dir $OUTPUT_DIR \
      --test_result_file $TEXT_RESULT \
      --test_prediction_file $TEXT_PREDICTION \
      --max_seq_length  $MAX_LENGTH \
      --num_train_epochs $NUM_EPOCHS \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --save_steps $SAVE_STEPS \
      --learning_rate 1e-5 \
      --gradient_accumulation_steps 2 \
      --seed $SEED \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir
    done
  done
done
