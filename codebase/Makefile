train:
	# rlaunch --preemptible=no --gpu=4 --memory=128000 --cpu=32 \
	# -- python3 -m torch.distributed.launch --nproc_per_node=4 \
	# main.py --order train --iters 20000 --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --data_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --data_target_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --augment --lr 1e-4 --mixup --mix_mode object --freeze_clip \
	# --max_eval_num 500 --distributed --world_size 4 --log_interval 100 \
	# --tag=1.1 --balance

	# rlaunch --preemptible=no --gpu=4 --memory=128000 --cpu=32 \
	# -- python3 -m torch.distributed.launch --nproc_per_node=4 \
	# main.py --order train --iters 20000 --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --data_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --data_target_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
    # --model_path ../exps1.1/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --augment --lr 2e-6 --mixup --mix_mode object \
	# --max_eval_num 500 --distributed --world_size 4 --log_interval 100 \
	# --tag=1.2 --balance

	rlaunch --preemptible=no --gpu=4 --memory=128000 --cpu=32 \
	-- python3 -m torch.distributed.launch --nproc_per_node=4 \
	main.py --order uda --iters 10000 --bs 1 \
	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	--data_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	--data_target s3://huangailin/uda_datasets/datasets_json/zerowaste-v2-trainval/train.json \
	--data_target_val s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
    --model_path ../exps1.2/weight.pth \
	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	--augment --lr 1e-7 --mixup --mix_mode object \
	--max_eval_num 10 --distributed --world_size 4 --log_interval 50 \
	--tag=2.1_final --uda_version 6 --balance

val:
	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --model_path ../exps1.1/weight.pth \
	# --output_dir ../source_only_train  
	
	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
	# --model_path ../exps1.1/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --output_dir ../source_only_val  

	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/test.json \
	# --model_path ../exps1.1/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --output_dir ../source_only_test  

	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --model_path ../exps1.2/weight.pth \
	# --output_dir ../source_only_train  
	
	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
	# --model_path ../exps1.2/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --output_dir ../source_only_val  

	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order val --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/test.json \
	# --model_path ../exps1.2/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --output_dir ../source_only_test  


	
	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	-- python3 main.py --order val --bs 1 \
	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
	--model_path ../exps2.1/weight.pth \
	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	--output_dir ../uda_val  

	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	-- python3 main.py --order val --bs 1 \
	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	--model_path ../exps2.1/weight.pth \
	--output_dir ../uda_train  

	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	-- python3 main.py --order val --bs 1 \
	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/test.json \
	--model_path ../exps2.1/weight.pth \
	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	--output_dir ../uda_test  

# val:
# 	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
# 	-- python3 main.py --order val --bs 1 \
# 	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/train.json \
# 	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
# 	--model_path ../exps1.2/weight.pth \
# 	--output_dir ../source_only_train  
	
# 	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
# 	-- python3 main.py --order val --bs 1 \
# 	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/val.json \
# 	--model_path ../exps1.2/weight.pth \
# 	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
# 	--output_dir ../source_only_val

# 	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
# 	-- python3 main.py --order val --bs 1 \
# 	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-f/test.json \
# 	--model_path ../exps1.2/weight.pth \
# 	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
# 	--output_dir ../source_only_test 




test:
	# rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	# -- python3 main.py --order test --bs 1 \
	# --data s3://huangailin/uda_datasets/datasets_json/zerowaste-v2-trainval/test.json \
	# --model_path ../exps1.2/weight.pth \
	# --model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	# --output_dir ../source_only --morphology --kernel_shape circle

	rlaunch --preemptible=no --gpu=1 --memory=64000 --cpu=8 \
	-- python3 main.py --order test  --bs 1 \
	--data s3://huangailin/uda_datasets/datasets_json/zerowaste-v2-trainval/test.json \
	--model_path ../exps2.1/weight.pth \
	--model segclip --crop --crop_size 352 --resize --resize_ratio 0.5 \
	--output_dir ../uda_version --morphology --kernel_shape circle