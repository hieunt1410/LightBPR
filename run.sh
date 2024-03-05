python train.py -d clothing
python train.py -d food
python train.py -d electronic

python train_gen.py -d clothing
python train_gen.py -d electronic
python train_gen.py -d food

python eval_bundle.py -d clothing
python eval_bundle.py -d electronic
python eval_bundle.py -d food