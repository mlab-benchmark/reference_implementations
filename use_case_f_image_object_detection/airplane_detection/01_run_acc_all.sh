run(){
	model=$1
	echo "Working on:" $model
	predictions=${model/xmodel/json}
	#echo $predictions
	./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/test_accuracy_yolov4_mt $model ./04_submissions/05_image_aircraft_object_detection_zcu102_s1/airplane_detection/images_val.list $predictions
}
cd ..
cd ..
for i in 64 128 192 256 320 384 448 512  576 640 
do
	run ./models/05_image_aircraft_object_detection_zcu102_s1/dpu_yolov4-${i}.xmodel 2>&1 | tee -a ./04_submissions/05_image_aircraft_object_detection_zcu102_s1/accuracy.txt
done
