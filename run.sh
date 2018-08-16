#python atari.py -e Pong-v4 -t --replay-start-size 1 \
#	--initial-random-actions 1 --max-episodes 10 --max-frames-number 100 \
#	--test-freq 10 --validation-frames 500 --learning-rate 0.0005 \
#	--minibatch-size 128 
python atari.py -e Pong-v4 -t --replay-start-size 1 \
	--initial-random-actions 30 --max-episodes 4000 --max-frames-number 400000 \
	--test-freq 25000 --validation-frames 70000 --learning-rate 0.0005 \
	--minibatch-size 128 