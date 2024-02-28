run(){
	model=$1
	echo "Working on:" $model
	./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/test_performance_yolov4 $model ./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/images_val.list -t4 -s30 #2>&1 | tee -a output_perf_all_sizes.txt
}

cd ..
cd ..
rm -f output_perf_all_sizes.txt	

for j in  {1..5}
# for j in  {1..1}
do
echo running the ${j}th experiment

	for i in 64 128 192 256 320 384 448 512  576 640
	do
		run ./models/05_image_aircraft_object_detection_zcu102_s1/dpu_yolov4-${i}.xmodel
	done
done
