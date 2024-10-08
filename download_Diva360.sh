#!/bin/bash
set -e 
set -x
DATA_PATH="./data"

# Download the BRICS dataset
scenes=("battery" "blue_car" "bunny" "chess" "clock" "dog" "drum" "flip_book" "horse" "hour_glass" "jenga" "k1_double_punch" "k1_hand_stand" "k1_push_up" "keyboard_mouse" "kindle" "maracas" "music_box" "pan" "peel_apple" "penguin" "piano" "plasma_ball" "plasma_ball_clip" "poker" "pour_salt" "pour_tea" "put_candy" "put_fruit" "red_car" "scissor" "slice_apple" "soda" "stirling" "tambourine" "tea" "tornado" "trex" "truck" "unlock" "wall_e" "wolf" "world_globe" "writing_1" "writing_2" "xylophone")

scenes_short=("xylophone" "blue_car" "dog" "flip_book" "keyboard_mouse" "kindle" "maracas" "pan" "peel_apple" "plasma_ball_clip" "poker" "put_candy" "pour_tea" "put_fruit" "red_car" "slice_apple" "soda" "stirling" "tambourine" "tea" "tornado" "trex" "truck" "unlock" "wall_e")

scenes_short_default=("pour_tea" "put_fruit" "red_car" "slice_apple" "soda" "stirling" "tambourine" "tea" "tornado" "trex" "truck" "unlock" "wall_e")
for scene in "${scenes_short_default[@]}"
do
    mkdir -p $DATA_PATH/$scene
    echo "Downloading $scene"
    aws s3 cp s3://diva360/processed_data/$scene/frames_1.tar.gz $DATA_PATH/$scene/ --no-sign-request
    tar -xvzf $DATA_PATH/$scene/frames_1.tar.gz -C $DATA_PATH/$scene
    rm $DATA_PATH/$scene/frames_1.tar.gz

    aws s3 cp s3://diva360/processed_data/$scene/transforms_test.json $DATA_PATH/$scene/ --no-sign-request
    aws s3 cp s3://diva360/processed_data/$scene/transforms_train.json $DATA_PATH/$scene/ --no-sign-request
    aws s3 cp s3://diva360/processed_data/$scene/transforms_val.json $DATA_PATH/$scene/ --no-sign-request
done