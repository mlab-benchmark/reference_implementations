run(){
	threads=$1
	echo "Working on:" $threads
	./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/test_performance_yolov4 ./models/05_image_aircraft_object_detection_zcu102_s1/dpu_yolov4-512.xmodel ./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/images_val.list -t${threads} -s30 # 2>&1 | tee -a output_perf_all_threads.txt
}

cd ..
cd ..
rm -f output_perf_all_threads.txt	

for j in  {1..5}
#for j in  {1..1}
do
	echo running the ${j}th experiment
	for i in 1 2 3 4 5 6 7 8
	do
		run ${i}
	done
done
