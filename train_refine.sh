
python3 prepare.py --action gen_val_set \
                  --estimator_cfg configs/gen6d_pretrain.yaml \
                  --que_database linemod/cat \
                  --que_split linemod_val \
                  --ref_database linemod/cat \
                  --ref_split linemod_val

python3 prepare.py --action gen_val_set \
                  --estimator_cfg configs/gen6d_pretrain.yaml \
                  --que_database genmop/tformer-test \
                  --que_split all \
                  --ref_database genmop/tformer-ref \
                  --ref_split all 

python3 train_model.py --cfg configs/detector/detector_train.yaml
python3 train_model.py --cfg configs/selector/selector_train.yaml
# python3 train_model.py --cfg configs/refiner/refiner_train.yaml
python3 train_model.py --cfg configs/refiner/refiner_train.yaml