source activate kvswap
cd LongBench
python /root/KVswap/LongBench/pred.py --model_name llama-3-8b-262k --method quest --name quest_prefetch_cache_fixed_k=4_2048 --quest_page_size 32 --sparse_budget 2048
python /root/KVswap/LongBench/pred.py --model_name llama-3-8b-262k --method quest --name quest_prefetch_cache_threshold_0.75_2048 --quest_page_size 32 --sparse_budget 2048