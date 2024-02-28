echo "heavy resnet model"

for i in {1..100}
do
	python3 test_performance.py $i ../vai_c_output/rfClassification.xmodel 1000
done

echo "loght convolutional model"

for i in {1..100}
do
	python3 test_performance.py $i ../vai_c_output/light_conv_updated_weights.xmodel 1000
done



