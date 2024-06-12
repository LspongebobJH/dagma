version=5
log_file_global=11
log_file_column=12

python gen_copies.py --gen_type X --version $version
for i in {1..10}; do
    python gen_copies.py --gen_type knockoff --seed_knockoff $i --version $version &
done

wait

for i in {1..10}; do
    python gen_copies.py --gen_type W  --seed_knockoff $i --version $version &
done

wait

if [ $? -eq 0 ]; then
    python multi_main.py --control_type=type_3_global --dagma_type=dagma_1 \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --version=$version \
    --log_file=log_${log_file_global} &

    python multi_main.py --control_type=type_3 --dagma_type=dagma_1 \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --version=$version \
    --log_file=log_${log_file_column} &
fi