#!/bin/bash

# Set up path to CUDA library
#source ~/.profile

export PYTHONPATH=`pwd`'/..:'${PYTHONPATH}

ARGS=${@:1}

# num_epochs was 100, num_samples was 256
python3 -m encoder_decoder.translate \
    --rnn_cell gru \
    --encoder_topology birnn \
    --num_epochs 1 \
    --num_samples 32 \
    --variational_recurrent_dropout \
    --token_decoding_algorithm beam_search \
    --beam_size 100 \
    --alpha 1.0 \
    --num_nn_slot_filling 10 \
    ${ARGS}
