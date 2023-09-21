# pip3 install -e .
# python3 examples/run_rl.py --algorithm=nfsp --env=no-limit-holdem --log_dir=experiments/no-limit_nfsp_result/
# python3 examples/run_rl.py --algorithm=ppo --env=no-limit-holdem --log_dir=experiments/no-limit_ppo_result/

# 2L-8:                     2+1 layers, 8*8,     SGD,  clip=0.2, lr=0.0001 15000
# 2L-8-AdamW:               2+1 layers, 8*8,     AdamW,clip=0.2, lr=0.0001 15000
# 2L-8-AdamW-lr0.001:       2+1 layers, 8*8,     AdamW,clip=0.2, lr=0.001  15000
# 2L-8-clip0.5:             2+1 layers, 8*8,     SGD,  clip=0.5, lr=0.0001 30000
# 2L-8-clip0.5-Adam:        2+1 layers, 8*8,     Adam, clip=0.5, lr=0.0001 30000
# 2L-16-AdamW-lr0.001:      2+1 layers, 16*16,   AdamW,clip=0.2, lr=0.001  15000 #Best
# 2L-16-clip0.5:            2+1 layers, 16*16,   SGD,  clip=0.5, lr=0.0001 15000
# 3L-4-clip0.5:             3+2 layers, 4*4,     SGD,  clip=0.5, lr=0.0001 30000
# 3L-4_2-clip0.5:           3+2 layers, 4*2,     SGD,  clip=0.5, lr=0.0001 30000
# 3L-8-clip0.5:             3+2 layers, 8*4,     SGD,  clip=0.5, lr=0.0001 30000
# 3L-8-clip0.5-lr0.001:     3+2 layers, 8*4,     SGD,  clip=0.5, lr=0.001  15000
# 3L-8-clip0.5-lr0.0005:    3+2 layers, 8*4,     SGD,  clip=0.5, lr=0.0005 30000
# 3L-512-clip0.5:           3+2 layers, 512*256, SGD,  clip=0.5, lr=0.0001 15000
# 4L:                       4+3 layers, 32*32,   SGD,  clip=0.2, lr=0.0001 30000
# 4L-512-clip0.5:           4+3 layers, 512*256, SGD,  clip=0.5, lr=0.0001 15000
# 4L-clip0.5:               4+3 layers, 32*32,   SGD,  clip=0.5, lr=0.0001 30000

