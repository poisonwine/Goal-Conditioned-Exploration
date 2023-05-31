# DEST
# nohup mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --critic_type 'monolithic' --agent 'DDPG' --n_epochs 100 --seed 5  --explore_alpha 0.5 --alg DEST --goal_teacher --teacher_method AGE --sample_stratage MEGA_MinV --goal_shift   --num_rollouts_per_mpi 2   --state_discover_method mine --state_discover --reward_teacher --reward_method mine  --age_lambda 0.2  >pushnew_5.log 2>&1 &
# nohup mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --critic_type 'monolithic' --agent 'DDPG' --n_epochs 100 --seed 6  --explore_alpha 0.5 --alg DEST --goal_teacher --teacher_method AGE --sample_stratage MEGA_MinV --goal_shift   --num_rollouts_per_mpi 2   --state_discover_method mine --state_discover --reward_teacher --reward_method mine --age_lambda 0.2  >pushnew_6.log 2>&1 &
# nohup mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --critic_type 'monolithic' --agent 'DDPG' --n_epochs 100 --seed 7  --explore_alpha 0.5 --alg DEST --goal_teacher --teacher_method AGE --sample_stratage MEGA_MinV --goal_shift   --num_rollouts_per_mpi 2   --state_discover_method mine --state_discover --reward_teacher --reward_method mine --age_lambda 0.2  >pushnew_7.log 2>&1 &

# HGG
# nohup mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --critic_type 'monolithic' --agent 'DDPG' --n_epochs 100 --seed 5   --alg HGG --goal_teacher --teacher_method HGG >pushnew_5.log 2>&1 &

# VDS
# nohup mpirun -np 6 python -u train.py --env_name FetchPushMiddleGap-v1 --critic_type 'monolithic' --agent 'DDPG' --n_epochs 100 --seed 5   --alg VDS --goal_teacher --teacher_method VDS >pushnew_5.log 2>&1 &
