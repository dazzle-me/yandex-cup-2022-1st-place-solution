for fold in 0
do
    python3 train_arcface.py --fold ${fold} --exp exps/exp141 --seed 43
done