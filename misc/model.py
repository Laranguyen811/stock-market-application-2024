gemini_model = Gemini(
    num_tokens=10000, #number of tokens, every 100 tokens represent 60-80 words
    max_seq_len=1024,#maximum sequence length, controlling the number of tokens of the input
    dim=320,#dimension, representing the number of features
    depth=8,#the number of layers in a neural network
    dim_head=32,#referring the dimensionality of key,query, and value vectors in the self-attention mechanism
    heads=6,#A head = the part of the model that makes predictions. Every model has a backbone and a head (backbone network) and a certain amount of prediction heads
    use_abs_pos_emb=False,#use absolute positional embedding. Positional embeddings inform about the position of tokens in input sequence
    attn_flash=True,#using flash attention, an efficient and precise Transformer acceleration technique,perceiving memory read and write ops,
    #changing blocks of query,key,and value from the GPU's HBM(main memory) to SRAM.
    attn_kv_heads=2,#key-value caching pair with attention heads
    qk_norm=True,#normalisation of the query and key vectors
    attn_qk_norm=True,#normalisation of the query and key vectors of attention mechanism
    attn_qk_norm_dim_scale=True,# the attention score computed as a dot product of query and key vectors,scaled down by the square root of
    #the dimension of these vectors => applying learned scale across the dimension of features, then normalising the query and key vectors

)