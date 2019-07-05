# train_code
# python main.py --train_GT_path ../dataSet/DIV2K/DIV2K_train_HR --train_LR_path ../dataSet/DIV2K/DIV2K_train_LR_bicubic/X2/ --test_GT_path ../dataSet/benchmark/Set5/HR/ --test_LR_path ../dataSet/benchmark/Set5/LR_bicubic/X2/ --test_with_train True --scale 2 --log_freq 1000

# python main.py --train_GT_path ../dataSet/DIV2K/DIV2K_train_HR --train_LR_path ../dataSet/DIV2K/DIV2K_train_LR_bicubic/X3/ --test_GT_path ../dataSet/benchmark/Set5/HR/ --test_LR_path ../dataSet/benchmark/Set5/LR_bicubic/X3/ --test_with_train True --scale 3 --log_freq 1000 --pre_trained_model ./model/RCAN_X2 --fine_tuning True --load_tail_part False --learning_rate 1e-5 --max_step 600000

# python main.py --train_GT_path ../dataSet/DIV2K/DIV2K_train_HR --train_LR_path ../dataSet/DIV2K/DIV2K_train_LR_bicubic/X4/ --test_GT_path ../dataSet/benchmark/Set5/HR/ --test_LR_path ../dataSet/benchmark/Set5/LR_bicubic/X4/ --test_with_train True --scale 4 --log_freq 1000 --pre_trained_model ./model/RCAN_X2 --fine_tuning True --load_tail_part False --learning_rate 1e-5 --max_step 600000

# test_code

scale=$(seq 2 4)

for s in $scale

do

  python main.py --mode test --pre_trained_model ./model/RCAN_X$s --test_LR_path ../dataSet/benchmark/Set5/LR_bicubic/X$s/ --test_GT_path ../dataSet/benchmark/Set5/HR/ --scale $s --save_test_result True --test_set Set5 

  python main.py --mode test --pre_trained_model ./model/RCAN_X$s --test_LR_path ../dataSet/benchmark/Set14/LR_bicubic/X$s/ --test_GT_path ../dataSet/benchmark/Set14/HR/ --scale $s --save_test_result True --test_set Set14

  python main.py --mode test --pre_trained_model ./model/RCAN_X$s --test_LR_path ../dataSet/benchmark/B100/LR_bicubic/X$s/ --test_GT_path ../dataSet/benchmark/B100/HR/ --scale $s --save_test_result True --test_set B100

  python main.py --mode test --pre_trained_model ./model/RCAN_X$s --test_LR_path ../dataSet/benchmark/Urban100/LR_bicubic/X$s/ --test_GT_path ../dataSet/benchmark/Urban100/HR/ --scale $s --save_test_result True --test_set Urban100

done


