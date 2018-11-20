## 登录机器17，启动ps:0
CUDA_VISIBLE_DEVICES=-1 python dist_tf_mnist_async.py --ps_hosts 192.168.40.17:2223 --worker_hosts 192.168.40.17:2224,192.168.40.17:2225,192.168.40.18:2223,192.168.40.18:2224 --job_name ps --task_id 0
## 登录机器17，启动worker:0
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist_async.py --ps_hosts 192.168.40.17:2223 --worker_hosts 192.168.40.17:2224,192.168.40.17:2225,192.168.40.18:2223,192.168.40.18:2224 --job_name worker --task_id 0
## 登录机器17，启动worker:1
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist_async.py --ps_hosts 192.168.40.17:2223 --worker_hosts 192.168.40.17:2224,192.168.40.17:2225,192.168.40.18:2223,192.168.40.18:2224 --job_name worker --task_id 1
## 登录机器18，启动worker:0
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist_async.py --ps_hosts 192.168.40.17:2223 --worker_hosts 192.168.40.17:2224,192.168.40.17:2225,192.168.40.18:2223,192.168.40.18:2224 --job_name worker --task_id 2
## 登录机器18，启动worker:1
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist_async.py --ps_hosts 192.168.40.17:2223 --worker_hosts 192.168.40.17:2224,192.168.40.17:2225,192.168.40.18:2223,192.168.40.18:2224 --job_name worker --task_id 3
