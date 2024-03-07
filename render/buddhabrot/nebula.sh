SAMPLES=1000000000

./a.out 5000 $SAMPLES
python convert.py out.img red.png

./a.out 500 $SAMPLES
python convert.py out.img green.png

./a.out 50 $SAMPLES
python convert.py out.img blue.png

python nebula.py
