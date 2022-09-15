#######################L1################
L1_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":6, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

L1_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1}]

#######################L2################
L2_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":6, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

L2_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1}]

#######################L3################
L3_attention_mse=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse", "weight":1}]

L3_attention_ce=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1}]

L3_attention_mse_sum=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse_sum", "weight":1}]

L3_attention_ce_mean=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1}]

L3_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1}]

L3n_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]}]

L3_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

#######################L4################
L4_attention_mse=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_mse", "weight":1}]

L4_attention_ce=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_ce", "weight":1}]

L4_attention_mse_sum=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_mse_sum", "weight":1}]

L4_attention_ce_mean=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_ce_mean", "weight":1}]

L4_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[9,9],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1}]

L4_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":3, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":6, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":9, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":12,"layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1}]



L4t_hidden_sgram=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"gram", "weight":1, "proj":["linear",312,768]},
                  {"layer_T":[3,3],  "layer_S":[1,1], "feature":"hidden", "loss":"gram", "weight":1, "proj":["linear",312,768]},
                  {"layer_T":[6,6],  "layer_S":[2,2], "feature":"hidden", "loss":"gram", "weight":1, "proj":["linear",312,768]},
                  {"layer_T":[9,9],  "layer_S":[3,3], "feature":"hidden", "loss":"gram", "weight":1, "proj":["linear",312,768]},
                  {"layer_T":[12,12],"layer_S":[4,4], "feature":"hidden", "loss":"gram", "weight":1, "proj":["linear",312,768]}]

L4t_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":3, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":6, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":9, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":12,"layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]}]


#######################L5################
L5_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[9,9],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1}]

L5_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":3, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":5, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":7, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":9, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1},
                {"layer_T":12,"layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1}]


###########L6#############
L6_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],  "layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[10,10],"layer_S":[5,5], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[6,6], "feature":"hidden", "loss":"mmd", "weight":1}]

L6_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":4, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":6, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":8, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":10,"layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":12,"layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

init_layers_L7=[{"layer_S": 0, "layer_T": 0, "type": "all"},
               {"layer_S": 1, "layer_T": 2, "type": "all"},
               {"layer_S": 2, "layer_T": 4, "type": "all"},
               {"layer_S": 3, "layer_T": 5, "type": "all"},
               {"layer_S": 4, "layer_T": 6, "type": "all"},
               {"layer_S": 5, "layer_T": 8, "type": "all"},
               {"layer_S": 6, "layer_T": 10, "type": "all"},
               {"layer_S": 7, "layer_T": 12, "type": "all"}]




###########L8#############
L8_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[5,5],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[7,7],"layer_S":[5,5], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],"layer_S":[6,6], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[10,10],"layer_S":[7,7], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[8,8], "feature":"hidden", "loss":"mmd", "weight":1}]

L8_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":4, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":5, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":6, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":7,"layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":8,"layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":10,"layer_S":7, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":12,"layer_S":8, "feature":"hidden", "loss":"hidden_mse", "weight":1}]




###########L10#############
L10_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[5,5],  "layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],"layer_S":[5,5], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[7,7],"layer_S":[6,6], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],"layer_S":[7,7], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[9,9],"layer_S":[8,8], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[10,10],"layer_S":[9,9], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[10,10], "feature":"hidden", "loss":"mmd", "weight":1}]

L10_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":3, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":4, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":5, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":6,"layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":7,"layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":8,"layer_S":7, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":9,"layer_S":8, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":10,"layer_S":9, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":10, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

###########L12#############
L12_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[1,1],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[5,5],"layer_S":[5,5], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],"layer_S":[6,6], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[7,7],"layer_S":[7,7], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],"layer_S":[8,8], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[9,9],"layer_S":[9,9], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[10,10],"layer_S":[10,10], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[11,11],"layer_S":[11,11], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[12,12], "feature":"hidden", "loss":"mmd", "weight":1}]

L12_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":1, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":2, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":3, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":4, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":5,"layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1}, 
               {"layer_T":6,"layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":7,"layer_S":7, "feature":"hidden", "loss":"hidden_mse", "weight":1},
              {"layer_T":8,"layer_S":8, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":9,"layer_S":9, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":10,"layer_S":10, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":11,"layer_S":11, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":12, "feature":"hidden", "loss":"hidden_mse", "weight":1}]


matches={'L3_attention_mse':L3_attention_mse,'L3_attention_mse_sum':L3_attention_mse_sum,
         'L3_attention_ce' :L3_attention_ce, 'L3_attention_ce_mean':L3_attention_ce_mean,
         'L3n_hidden_mse'  :L3n_hidden_mse,  'L3_hidden_smmd'      :L3_hidden_smmd,       'L3_hidden_mse': L3_hidden_mse,
         'L4_attention_mse':L4_attention_mse,'L4_attention_mse_sum':L4_attention_mse_sum,
         'L4_attention_ce' :L4_attention_ce, 'L4_attention_ce_mean':L4_attention_ce_mean,
         'L4t_hidden_mse'  :L4t_hidden_mse,  'L4_hidden_smmd'      :L4_hidden_smmd,       'L4t_hidden_sgram': L4t_hidden_sgram,
         'L6_hidden_mse'   :L6_hidden_mse,   'L6_hidden_smmd'      :L6_hidden_smmd,
         "L1_hidden_mse" : L1_hidden_mse, "L1_hidden_smmd" : L1_hidden_smmd,
         "L2_hidden_mse" : L2_hidden_mse, "L2_hidden_smmd" : L2_hidden_smmd, "L4_hidden_mse": L4_hidden_mse,
         "L5_hidden_mse" : L5_hidden_mse, "L5_hidden_smmd" : L5_hidden_smmd,
         "L8_hidden_mse" : L8_hidden_mse, "L8_hidden_smmd" : L8_hidden_smmd,
         "L10_hidden_mse" : L10_hidden_mse, "L10_hidden_smmd" : L10_hidden_smmd,
         "L12_hidden_mse" : L12_hidden_mse, "L12_hidden_smmd" : L12_hidden_smmd
        }
